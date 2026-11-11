// Element-wise steps the MTP draft's torch code runs as ATen TensorIterator kernels whose results
// are fixed by IEEE arithmetic alone: each output is one correctly rounded fp32 add or multiply (or
// an exact square, or a round-to-nearest-even bf16 conversion) of its inputs, so any kernel computing
// the same single operation per element writes the same bits. Built with -fmad=false so no multiply
// and add ever contract into an FMA. (Non-IEEE steps - rsqrt, sigmoid, the reductions - replay
// ATen's own kernels instead.)
#include <cuda_bf16.h>
#include <stdint.h>

#define GRID_STRIDE(i, n) for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < (n); i += (int64_t)gridDim.x * blockDim.x)

extern "C" {

// x.square(): ATen pow with exponent 2 is base * base.
__global__ void dsv4_square_f32(const float* a, float* out, int64_t n) {
  GRID_STRIDE(i, n) { float v = a[i]; out[i] = v * v; }
}

// x + scalar (a Python float added to an fp32 tensor: the scalar rounded to fp32, alpha 1).
__global__ void dsv4_add_scalar_f32(const float* a, float s, float* out, int64_t n) {
  GRID_STRIDE(i, n) { out[i] = a[i] + s; }
}

// out[r, j] = a[r, j] * b[r]   (a [rows, cols] times a [rows, 1] column).
__global__ void dsv4_mul_rows_f32(const float* a, const float* b, float* out, int64_t rows, int cols) {
  GRID_STRIDE(i, rows * cols) { out[i] = a[i] * b[i / cols]; }
}

// out[i] = a[i] * s[0]   (times a one-element device tensor).
__global__ void dsv4_mul_dev_scalar_f32(const float* a, const float* s, float* out, int64_t n) {
  float v = s[0];
  GRID_STRIDE(i, n) { out[i] = a[i] * v; }
}

// out[r, j] = a[r, j] + b[j]   (plus a [cols] row).
__global__ void dsv4_add_cols_f32(const float* a, const float* b, float* out, int64_t rows, int cols) {
  GRID_STRIDE(i, rows * cols) { out[i] = a[i] + b[i % cols]; }
}

// out[q, k] = p[q] * x[q, k]   (p [rc, 1] times x [rc, d]: hc_head's pre.unsqueeze(-1) * x).
__global__ void dsv4_mul_bcast_last_f32(const float* p, const float* x, float* out, int64_t rc, int d) {
  GRID_STRIDE(i, rc * d) { out[i] = p[i / d] * x[i]; }
}

// out[r, c, k] = bf16(float(e[r, k]) + float(h[r, c, k]))   (e[:, None, :] + h, bf16 tensors: ATen
// adds in fp32 and rounds to nearest even).
__global__ void dsv4_add_bcast_hc_bf16(const __nv_bfloat16* e, const __nv_bfloat16* h, __nv_bfloat16* out, int64_t rows, int c, int d) {
  GRID_STRIDE(i, rows * c * d) {
    int64_t r = i / ((int64_t)c * d), k = i % d;
    out[i] = __float2bfloat16_rn(__bfloat162float(e[r * d + k]) + __bfloat162float(h[i]));
  }
}

// x.to(bf16) from fp32 (round to nearest even).
__global__ void dsv4_f32_to_bf16(const float* a, __nv_bfloat16* out, int64_t n) {
  GRID_STRIDE(i, n) { out[i] = __float2bfloat16_rn(a[i]); }
}

// Data movement for the MTP loop's per-request state (no arithmetic): dst row dst_idx[r] =
// src row src_idx[r] for r < n, rows of row_words 8-byte words; a null index array is the identity.
__global__ void dsv4_copy_rows(const uint64_t* src, const int32_t* src_idx, uint64_t* dst, const int32_t* dst_idx, int n, int row_words) {
  GRID_STRIDE(i, (int64_t)n * row_words) {
    int64_t r = i / row_words, w = i % row_words;
    int64_t from = src_idx ? src_idx[r] : r, to = dst_idx ? dst_idx[r] : r;
    dst[to * row_words + w] = src[from * row_words + w];
  }
}

// dst[i * stride + offset] = src[i] (int64, i < n).
__global__ void dsv4_scatter_strided_i64(const int64_t* src, int64_t* dst, int n, int stride, int offset) {
  GRID_STRIDE(i, n) { dst[i * stride + offset] = src[i]; }
}

// After a speculative verify (SGLang's clear_unaccepted_c128_draft_states, every C128 layer and
// request in one launch): request r's ring rows rid[r] * ring + (seq[r] + d) % ring for d in
// [accept[r], drafts) get kv 0.0 and score -inf. Grid: x over the half row, y = r * drafts + d,
// z = the layer (layer_ids[z]); `state` is the cache's C128 state, `layer_stride` floats a layer.
__global__ void dsv4_clear_c128_rejected(float* state, const int32_t* layer_ids, int64_t layer_stride, const int32_t* rid, const int32_t* seq,
                                         const int32_t* accept, int ring, int drafts, int half) {
  int r = blockIdx.y / drafts, d = blockIdx.y % drafts;
  if (d < accept[r]) return;
  int64_t row = (int64_t)rid[r] * ring + (seq[r] + d) % ring;
  float* base = state + layer_ids[blockIdx.z] * layer_stride + row * 2 * half;
  for (int w = blockIdx.x * blockDim.x + threadIdx.x; w < half; w += gridDim.x * blockDim.x) {
    base[w] = 0.0f;
    base[half + w] = __int_as_float(0xff800000);
  }
}

// The decode inputs' compressed-entry lists and page table, from per-row parameters
// fill[b, 5] = (c4 base, c4 count, c128 base, c128 count, page0): c4_ids[r, e] = base + e for
// e < count else -1 (count 0 when the request holds more than 512 entries: the indexer picks),
// c128_ids likewise over 8256, table[r, p] = page0 + p. The values the host would build.
__global__ void dsv4_fill_decode_ids(const int32_t* fill, int32_t* c4_ids, int32_t* c128_ids, int32_t* table, int c4_width, int c128_width,
                                     int w) {
  int r = blockIdx.y;
  const int32_t* f = fill + 5 * r;
  int c4b = f[0], n4 = f[1], c128b = f[2], n128 = f[3], page0 = f[4];
  int n = max(max(c4_width, c128_width), w);
  for (int e = blockIdx.x * blockDim.x + threadIdx.x; e < n; e += gridDim.x * blockDim.x) {
    if (e < c4_width) c4_ids[(int64_t)r * c4_width + e] = e < n4 ? c4b + e : -1;
    if (e < c128_width) c128_ids[(int64_t)r * c128_width + e] = e < n128 ? c128b + e : -1;
    if (e < w) table[(int64_t)r * w + e] = page0 + e;
  }
}

}  // extern "C"
