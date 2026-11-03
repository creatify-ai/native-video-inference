// The official Wan 2.1 VAE (wan/modules/vae.py) in fp32, as WanVAE runs it: layout moves are exact
// copies; arithmetic follows the eager fp32 kernels one op at a time. A tensor [C, T, H, W] is held
// either contiguous ("CT") or, after `rearrange('(b t) c h w -> b c t h w')`, as the permuted view
// of [T, C, H, W] memory ("TC"); `tc` flags the second.
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>

namespace {
constexpr int kThreads = 256;
inline unsigned blocks(int64_t n) { return unsigned((n + kThreads - 1) / kThreads); }
__device__ __forceinline__ int64_t at(int tc, int64_t c, int64_t f, int64_t C, int64_t T, int64_t hw, int64_t s) {
    return tc ? (f * C + c) * hw + s : (c * T + f) * hw + s;
}

// CausalConv3d input: cat([cache, x], dim=2) then F.pad (W, H symmetric, `front` zero frames), into
// contiguous [C, out_frames, H+ph0+ph1, W+pw0+pw1]. The cache is contiguous; x may be CT or TC.
__global__ void pad_cat(const float* cache, int64_t cache_frames, const float* x, int x_tc, int64_t frames, float* out,
                        int64_t channels, int64_t front, int64_t out_frames, int64_t h, int64_t w, int64_t ph0, int64_t ph1,
                        int64_t pw0, int64_t pw1) {
    const int64_t oh = h + ph0 + ph1, ow = w + pw0 + pw1;
    const int64_t n = channels * out_frames * oh * ow;
    int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int64_t xx = i % ow, yy = (i / ow) % oh, f = (i / (ow * oh)) % out_frames, c = i / (ow * oh * out_frames);
    const int64_t sy = yy - ph0, sx = xx - pw0, sf = f - front;
    float v = 0.f;
    if (sy >= 0 && sy < h && sx >= 0 && sx < w && sf >= 0) {
        if (sf < cache_frames) v = cache[((c * cache_frames + sf) * h + sy) * w + sx];
        else if (sf - cache_frames < frames) v = x[at(x_tc, c, sf - cache_frames, channels, frames, h * w, sy * w + sx)];
    }
    out[i] = v;
}

// F.normalize(x, dim=1) * scale * gamma + 0. (then nn.SiLU when `silu`): x / clamp_min(norm, 1e-12),
// each op rounded; the output keeps x's layout. norm: contiguous [T, H*W].
__global__ void normalize(const float* x, const float* norm, const float* gamma, float* out, float scale, int silu, int tc,
                          int64_t C, int64_t T, int64_t hw, int64_t n) {
    int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int64_t c, f, s;
    if (tc) { s = i % hw; c = (i / hw) % C; f = i / (hw * C); }
    else { s = i % hw; f = (i / hw) % T; c = i / (hw * T); }
    const float d = ::max(norm[f * hw + s], 1e-12f);
    float v = __fadd_rn(__fmul_rn(__fmul_rn(__fdiv_rn(x[i], d), scale), gamma[c]), 0.0f);
    if (silu) v = v / (1.0f + ::exp(-v));
    out[i] = v;
}

// conv bias add_ on contiguous [C, inner].
__global__ void bias(float* x, const float* b, int64_t channels, int64_t inner, int64_t n) {
    int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) x[i] = __fadd_rn(x[i], b[(i / inner) % channels]);
}

// out (CT) = x (CT) + h (CT or TC): ResidualBlock's `x + h` and AttentionBlock's `x + identity`.
__global__ void add(const float* x, const float* h, int h_tc, float* out, int64_t C, int64_t T, int64_t hw, int64_t n) {
    int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int64_t s = i % hw, f = (i / hw) % T, c = i / (hw * T);
    out[i] = __fadd_rn(x[i], h[at(h_tc, c, f, C, T, hw, s)]);
}

// [A, B, S] -> [B, A, S] (rearrange between [C, T] and [T, C] orders, or a [3C, HW] transpose with S = 1).
__global__ void swap01(const float* x, float* out, int64_t a, int64_t b, int64_t s) {
    const int64_t n = a * b * s;
    int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int64_t e = i % s, ia = (i / s) % a, ib = i / (s * a);
    out[i] = x[(ia * b + ib) * s + e];
}

// [N, H, W] -> [N, 2H, 2W] nearest-exact (scale 2: source = out // 2).
__global__ void upsample2x(const float* x, float* out, int64_t nc, int64_t h, int64_t w) {
    const int64_t n = nc * 4 * h * w;
    int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int64_t ow = 2 * w, ox = i % ow, oy = (i / ow) % (2 * h), p = i / (ow * 2 * h);
    out[i] = x[(p * h + oy / 2) * w + ox / 2];
}

// time_conv output [2C, T, H, W] -> reshape(2, C, T) -> stack(dim 3) -> [C, 2T, H, W].
__global__ void time_interleave(const float* x, float* out, int64_t c, int64_t t, int64_t hw) {
    const int64_t n = c * 2 * t * hw;
    int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int64_t s = i % hw, f2 = (i / hw) % (2 * t), ch = i / (hw * 2 * t);
    out[i] = x[(((f2 % 2) * c + ch) * t + f2 / 2) * hw + s];
}

// out[c, f, s] = x[c, start + f, s] of a CT or TC tensor (contiguous CT out).
__global__ void slice_time(const float* x, int tc, float* out, int64_t C, int64_t T, int64_t start, int64_t count, int64_t hw) {
    const int64_t n = C * count * hw;
    int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int64_t s = i % hw, f = (i / hw) % count, ch = i / (hw * count);
    out[i] = x[at(tc, ch, start + f, C, T, hw, s)];
}
// out[c, start + f, s] = x[c, f, s] (x CT or TC; torch.cat along time into a contiguous tensor).
__global__ void place_time(const float* x, int tc, float* out, int64_t C, int64_t t_out, int64_t start, int64_t count, int64_t hw) {
    const int64_t n = C * count * hw;
    int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int64_t s = i % hw, f = (i / hw) % count, ch = i / (hw * count);
    out[(ch * t_out + start + f) * hw + s] = x[at(tc, ch, f, C, count, hw, s)];
}

// encode: (mu - mean) * inv_std; decode: z / inv_std + mean (fp32, one rounding per op).
__global__ void scale_encode(const float* mu, const float* mean, const float* inv, float* out, int64_t inner, int64_t n) {
    int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) { const int64_t c = i / inner; out[i] = __fmul_rn(__fsub_rn(mu[i], mean[c]), inv[c]); }
}
__global__ void scale_decode(const float* z, const float* mean, const float* inv, float* out, int64_t inner, int64_t n) {
    int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) { const int64_t c = i / inner; out[i] = __fadd_rn(__fdiv_rn(z[i], inv[c]), mean[c]); }
}
// .float().clamp_(-1, 1): NaN propagates.
__global__ void clamp1(float* x, int64_t n) {
    int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) { const float v = x[i]; x[i] = isnan(v) ? v : ::min(::max(v, -1.0f), 1.0f); }
}

// The conditioning frame: F.interpolate(img[None].cpu(), (h, w), mode='bicubic') as torch's
// AVX2 / AVX-512 CPU build evaluates it (InterpolateNonSeparable, GCC FMA contraction): each
// row tap t_j = fma(x0, w0, x1 * w1) -> fma(x2, w2, .) -> fma(x3, w3, .) over the column taps,
// then the same combination of t_0..t_3 with the row weights. Taps (clamped indices and weights)
// come from frontend::cubic_taps. Writes frame 0 of the contiguous [3, frames, h, w] VAE input.
__global__ void bicubic(const float* src, int64_t in_h, int64_t in_w, const int* iy, const float* wy, const int* ix,
                        const float* wx, float* out, int64_t out_h, int64_t out_w, int64_t channel_stride, int64_t n) {
    int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int64_t x = i % out_w, y = (i / out_w) % out_h, c = i / (out_w * out_h);
    const float* plane = src + c * in_h * in_w;
    const int* cx = ix + 4 * x;
    const float* kx = wx + 4 * x;
    float t[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float* row = plane + int64_t(iy[4 * y + j]) * in_w;
        float a = __fmaf_rn(row[cx[0]], kx[0], __fmul_rn(row[cx[1]], kx[1]));
        a = __fmaf_rn(row[cx[2]], kx[2], a);
        t[j] = __fmaf_rn(row[cx[3]], kx[3], a);
    }
    const float* ky = wy + 4 * y;
    float o = __fmaf_rn(t[0], ky[0], __fmul_rn(t[1], ky[1]));
    o = __fmaf_rn(t[2], ky[2], o);
    out[c * channel_stride + y * out_w + x] = __fmaf_rn(t[3], ky[3], o);
}

// ---- row forms ----------------------------------------------------------------------------------
// pad_cat, normalize, bias, add, slice_time and place_time over rows of their innermost extent (W for
// pad_cat, H*W or `inner` otherwise): the grid's y strides over rows, whose outer coordinates are
// decomposed once, and its x covers kRowChunk-element chunks of a row. Each element's expression is
// the flat kernel's.
constexpr int kRowChunk = 4 * kThreads;
inline dim3 rows_grid(int64_t rows, int64_t inner) {
    return dim3(unsigned((inner + kRowChunk - 1) / kRowChunk), unsigned(rows < 65535 ? rows : 65535));
}
#define W21_ROW_LOOP(rows, inner, body)                                                                  \
    for (int64_t r = blockIdx.y; r < (rows); r += gridDim.y) {                                          \
        _Pragma("unroll") for (int k = 0; k < 4; ++k) {                                                 \
            const int64_t j = int64_t(blockIdx.x) * kRowChunk + k * kThreads + threadIdx.x;             \
            if (j < (inner)) { body }                                                                   \
        }                                                                                               \
    }

__global__ void pad_cat_rows(const float* cache, int64_t cache_frames, const float* x, int x_tc, int64_t frames, float* out,
                             int64_t channels, int64_t front, int64_t out_frames, int64_t h, int64_t w, int64_t ph0,
                             int64_t ph1, int64_t pw0, int64_t pw1) {
    const int64_t oh = h + ph0 + ph1, ow = w + pw0 + pw1, rows = channels * out_frames * oh;
    for (int64_t r = blockIdx.y; r < rows; r += gridDim.y) {
        const int64_t yy = r % oh, f = (r / oh) % out_frames, c = r / (oh * out_frames);
        const int64_t sy = yy - ph0, sf = f - front;
        const float* src = nullptr;  // the source row, or none (zero padding)
        if (sy >= 0 && sy < h && sf >= 0) {
            if (sf < cache_frames) src = cache + ((c * cache_frames + sf) * h + sy) * w;
            else if (sf - cache_frames < frames) src = x + at(x_tc, c, sf - cache_frames, channels, frames, h * w, sy * w);
        }
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            const int64_t xx = int64_t(blockIdx.x) * kRowChunk + k * kThreads + threadIdx.x;
            if (xx >= ow) continue;
            const int64_t sx = xx - pw0;
            out[r * ow + xx] = src && sx >= 0 && sx < w ? src[sx] : 0.f;
        }
    }
}
__global__ void normalize_rows(const float* x, const float* norm, const float* gamma, float* out, float scale, int silu, int tc,
                               int64_t C, int64_t T, int64_t hw) {
    for (int64_t r = blockIdx.y; r < C * T; r += gridDim.y) {
        const int64_t c = tc ? r % C : r / T, f = tc ? r / C : r % T;
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            const int64_t s = int64_t(blockIdx.x) * kRowChunk + k * kThreads + threadIdx.x;
            if (s >= hw) continue;
            const int64_t i = r * hw + s;
            const float d = ::max(norm[f * hw + s], 1e-12f);
            float v = __fadd_rn(__fmul_rn(__fmul_rn(__fdiv_rn(x[i], d), scale), gamma[c]), 0.0f);
            if (silu) v = v / (1.0f + ::exp(-v));
            out[i] = v;
        }
    }
}
__global__ void bias_rows(float* x, const float* b, int64_t channels, int64_t inner, int64_t rows) {
    W21_ROW_LOOP(rows, inner, { x[r * inner + j] = __fadd_rn(x[r * inner + j], b[r % channels]); })
}
__global__ void add_rows(const float* x, const float* h, int h_tc, float* out, int64_t C, int64_t T, int64_t hw) {
    W21_ROW_LOOP(C * T, hw, {
        const int64_t i = r * hw + j;
        out[i] = __fadd_rn(x[i], h[at(h_tc, r / T, r % T, C, T, hw, j)]);
    })
}
__global__ void slice_time_rows(const float* x, int tc, float* out, int64_t C, int64_t T, int64_t start, int64_t count, int64_t hw) {
    W21_ROW_LOOP(C * count, hw, { out[r * hw + j] = x[at(tc, r / count, start + r % count, C, T, hw, j)]; })
}
__global__ void place_time_rows(const float* x, int tc, float* out, int64_t C, int64_t t_out, int64_t start, int64_t count,
                                int64_t hw) {
    W21_ROW_LOOP(C * count, hw, { out[((r / count) * t_out + start + r % count) * hw + j] = x[at(tc, r / count, r % count, C, count, hw, j)]; })
}
#undef W21_ROW_LOOP
}  // namespace

extern "C" {
int w21_vae_pad_cat(const void* cache, int64_t cache_frames, const void* x, int x_tc, int64_t frames, void* out, int64_t channels,
                    int64_t front, int64_t out_frames, int64_t h, int64_t w, int64_t ph0, int64_t ph1, int64_t pw0, int64_t pw1,
                    cudaStream_t s) {
    pad_cat_rows<<<rows_grid(channels * out_frames * (h + ph0 + ph1), w + pw0 + pw1), kThreads, 0, s>>>(
        (const float*)cache, cache_frames, (const float*)x, x_tc, frames, (float*)out, channels, front, out_frames, h, w, ph0, ph1,
        pw0, pw1);
    return cudaGetLastError();
}
int w21_vae_normalize(const void* x, const void* norm, const void* gamma, void* out, float scale, int silu, int tc, int64_t c,
                      int64_t t, int64_t hw, cudaStream_t s) {
    normalize_rows<<<rows_grid(c * t, hw), kThreads, 0, s>>>((const float*)x, (const float*)norm, (const float*)gamma, (float*)out, scale, silu, tc,
                                                             c, t, hw);
    return cudaGetLastError();
}
int w21_vae_bias(void* x, const void* b, int64_t channels, int64_t inner, int64_t batch, cudaStream_t s) {
    bias_rows<<<rows_grid(batch * channels, inner), kThreads, 0, s>>>((float*)x, (const float*)b, channels, inner, batch * channels);
    return cudaGetLastError();
}
int w21_vae_add(const void* x, const void* h, int h_tc, void* out, int64_t c, int64_t t, int64_t hw, cudaStream_t s) {
    add_rows<<<rows_grid(c * t, hw), kThreads, 0, s>>>((const float*)x, (const float*)h, h_tc, (float*)out, c, t, hw);
    return cudaGetLastError();
}
int w21_vae_swap01(const void* x, void* out, int64_t a, int64_t b, int64_t s_, cudaStream_t s) {
    swap01<<<blocks(a * b * s_), kThreads, 0, s>>>((const float*)x, (float*)out, a, b, s_);
    return cudaGetLastError();
}
int w21_vae_upsample2x(const void* x, void* out, int64_t nc, int64_t h, int64_t w, cudaStream_t s) {
    upsample2x<<<blocks(nc * 4 * h * w), kThreads, 0, s>>>((const float*)x, (float*)out, nc, h, w);
    return cudaGetLastError();
}
int w21_vae_time_interleave(const void* x, void* out, int64_t c, int64_t t, int64_t hw, cudaStream_t s) {
    time_interleave<<<blocks(c * 2 * t * hw), kThreads, 0, s>>>((const float*)x, (float*)out, c, t, hw);
    return cudaGetLastError();
}
int w21_vae_slice_time(const void* x, int tc, void* out, int64_t c, int64_t t, int64_t start, int64_t count, int64_t hw,
                       cudaStream_t s) {
    slice_time_rows<<<rows_grid(c * count, hw), kThreads, 0, s>>>((const float*)x, tc, (float*)out, c, t, start, count, hw);
    return cudaGetLastError();
}
int w21_vae_place_time(const void* x, int tc, void* out, int64_t c, int64_t t_out, int64_t start, int64_t count, int64_t hw,
                       cudaStream_t s) {
    place_time_rows<<<rows_grid(c * count, hw), kThreads, 0, s>>>((const float*)x, tc, (float*)out, c, t_out, start, count, hw);
    return cudaGetLastError();
}
int w21_vae_scale_encode(const void* mu, const void* mean, const void* inv, void* out, int64_t channels, int64_t inner,
                         cudaStream_t s) {
    scale_encode<<<blocks(channels * inner), kThreads, 0, s>>>((const float*)mu, (const float*)mean, (const float*)inv,
                                                               (float*)out, inner, channels * inner);
    return cudaGetLastError();
}
int w21_vae_scale_decode(const void* z, const void* mean, const void* inv, void* out, int64_t channels, int64_t inner,
                         cudaStream_t s) {
    scale_decode<<<blocks(channels * inner), kThreads, 0, s>>>((const float*)z, (const float*)mean, (const float*)inv,
                                                               (float*)out, inner, channels * inner);
    return cudaGetLastError();
}
int w21_vae_clamp(void* x, int64_t n, cudaStream_t s) {
    clamp1<<<blocks(n), kThreads, 0, s>>>((float*)x, n);
    return cudaGetLastError();
}
int w21_bicubic(const void* src, int64_t in_h, int64_t in_w, const void* iy, const void* wy, const void* ix, const void* wx,
                void* out, int64_t out_h, int64_t out_w, int64_t channel_stride, cudaStream_t s) {
    const int64_t n = 3 * out_h * out_w;
    bicubic<<<blocks(n), kThreads, 0, s>>>((const float*)src, in_h, in_w, (const int*)iy, (const float*)wy, (const int*)ix,
                                           (const float*)wx, (float*)out, out_h, out_w, channel_stride, n);
    return cudaGetLastError();
}
}
