// Request-encoder operators (CLIP visual, umt5 encoder) with pinned PyTorch 2.13 semantics on
// B200: torch's own device code from the wheel headers (persistent softmax, Welford and
// block-reduce helpers), kernels copied verbatim from the .cu sources that are not headers
// (GroupNorm, weight norm, linear upsample), and the cuBLAS calls torch's bmm makes.
#include <ATen/cuda/CUDAContext.h>
#include <ATen/native/cuda/PersistentSoftmax.cuh>
#include <c10/cuda/CUDAMathCompat.h>
#include <ATen/native/cuda/block_reduce.cuh>
#include <ATen/native/SharedReduceOps.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <utility>

namespace {
constexpr int kThreads = 256;
inline unsigned blocks(int64_t n) { return unsigned((n + kThreads - 1) / kThreads); }

// ---- GroupNorm (group_norm_kernel.cu, forward with gamma and beta) -------------------------------

template <typename T, typename T_ACC>
__global__ void RowwiseMomentsCUDAKernel(int64_t N, T eps, const T* X, T* mean, T* rstd, T_ACC* mean_acc,
                                         T_ACC* rstd_acc) {
    using WelfordType = at::native::WelfordData<T_ACC, int64_t>;
    using WelfordOp = at::native::WelfordOps<T_ACC, T_ACC, int64_t, std::pair<T_ACC, T_ACC>>;
    const int64_t i = blockIdx.x;
    WelfordOp welford_op = {/*correction=*/0, /*take_sqrt=*/false};
    WelfordType val(0, 0, 0, 0);
    for (int64_t j = threadIdx.x; j < N; j += blockDim.x) {
        const int64_t index = i * N + j;
        val = welford_op.reduce(val, static_cast<T_ACC>(X[index]), index);
    }
    if (blockDim.x <= C10_WARP_SIZE) {
        val = at::native::cuda_utils::WarpReduce(val, welford_op);
    } else {
        alignas(WelfordType) __shared__ char val_shared[sizeof(WelfordType) * C10_WARP_SIZE_UPPER_BOUND];
        WelfordType* val_shared_ptr = reinterpret_cast<WelfordType*>(val_shared);
        val = at::native::cuda_utils::BlockReduce(val, welford_op, /*identity_element=*/WelfordType(0, 0, 0, 0),
                                                  val_shared_ptr);
    }
    if (threadIdx.x == 0) {
        auto [m2, m1] = welford_op.project(val);
        T_ACC rstd_val = c10::cuda::compat::rsqrt(m2 + static_cast<T_ACC>(eps));
        mean[i] = m1;
        rstd[i] = rstd_val;
        if constexpr (!std::is_same_v<T, T_ACC>) {
            mean_acc[i] = m1;
            rstd_acc[i] = rstd_val;
        }
    }
}

template <typename T, typename T_ACC>
__global__ void ComputeFusedParamsCUDAKernel(int64_t N, int64_t C, int64_t group, const T_ACC* mean, const T_ACC* rstd,
                                             const T* gamma, const T* beta, T_ACC* a, T_ACC* b) {
    const int64_t index = ((int64_t)blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < N * C) {
        const int64_t ng = index / (C / group);
        const int64_t c = index % C;
        const T_ACC scale = (gamma == nullptr) ? rstd[ng] : rstd[ng] * static_cast<T_ACC>(gamma[c]);
        a[index] = scale;
        b[index] = -scale * mean[ng] + ((beta == nullptr) ? 0 : static_cast<T_ACC>(beta[c]));
    }
}

// gpu_kernel(iter, [](T x, T_ACC a, T_ACC b) -> T { return a * static_cast<T_ACC>(x) + b; })
__global__ void group_norm_apply(const float* x, const float* a, const float* b, float* y, int64_t hw, int64_t n) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int64_t nc = i / hw;
    const float av = a[nc], bv = b[nc];
    y[i] = av * static_cast<float>(x[i]) + bv;
}

// ---- weight norm over all dims but the last (WeightNorm.cu) --------------------------------------

#define TILE_W 16
#define TILE_H 64

template <typename T, typename ReduceOp>
__device__ __forceinline__ void reduce_block_into_lanes(T* x, T val, int lanes, ReduceOp reduceOp) {
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int blockSize = blockDim.x * blockDim.y;
    if (blockSize >= 64) {
        x[tid] = val;
        __syncthreads();
    }
#pragma unroll
    for (int i = (blockSize >> 1); i >= 64; i >>= 1) {
        if (tid < i) x[tid] = reduceOp(x[tid], x[tid + i]);
        __syncthreads();
    }
    if (tid < 32) {
        T final;
        if (blockSize >= 64) final = reduceOp(x[tid], x[tid + 32]);
        else final = val;
#pragma unroll
        for (int i = 16; i >= lanes; i >>= 1) final = reduceOp(final, WARP_SHFL_DOWN(final, i));
        if (tid < lanes) x[tid] = final;
    }
    __syncthreads();
}

template <typename T> struct ReduceAdd {
    inline __device__ T operator()(const T a, const T b) const { return a + b; }
};

template <typename scalar_t, typename accscalar_t>
__global__ void weight_norm_fwd_last_dim_kernel(scalar_t* __restrict__ w, accscalar_t* __restrict__ norms,
                                                const scalar_t* __restrict__ v, const scalar_t* __restrict__ g,
                                                const int fast_dim_size, const int slower_dims_size) {
    const int fast_dim_location = threadIdx.x + blockIdx.x * blockDim.x;
    extern __shared__ char buf[];
    accscalar_t* alloc = (accscalar_t*)buf;
    accscalar_t* s = &alloc[0];
    accscalar_t* rnorms_this_block = &alloc[blockDim.x * blockDim.y];
    accscalar_t thread_sum = 0.f;
    int slower_dims_location = threadIdx.y;
    int currentIdx = fast_dim_location + fast_dim_size * slower_dims_location;
    if (fast_dim_location < fast_dim_size)
        while (slower_dims_location < slower_dims_size) {
            accscalar_t val_f = static_cast<accscalar_t>(v[currentIdx]);
            thread_sum += val_f * val_f;
            currentIdx += blockDim.y * fast_dim_size;
            slower_dims_location += blockDim.y;
        }
    reduce_block_into_lanes(s, thread_sum, blockDim.x, ReduceAdd<accscalar_t>());
    if (threadIdx.y == 0) {
        accscalar_t result = s[threadIdx.x];
        accscalar_t norm_this_col = sqrtf(result);
        norms[fast_dim_location] = norm_this_col;
        rnorms_this_block[threadIdx.x] = 1.f / norm_this_col;
    }
    __syncthreads();
    accscalar_t g_this_col = static_cast<accscalar_t>(g[fast_dim_location]);
    accscalar_t rnorm = rnorms_this_block[threadIdx.x];
    slower_dims_location = threadIdx.y;
    currentIdx = fast_dim_location + fast_dim_size * slower_dims_location;
    if (fast_dim_location < fast_dim_size)
        while (slower_dims_location < slower_dims_size) {
            accscalar_t val_f = static_cast<accscalar_t>(v[currentIdx]);
            w[currentIdx] = static_cast<scalar_t>(g_this_col * val_f * rnorm);
            currentIdx += blockDim.y * fast_dim_size;
            slower_dims_location += blockDim.y;
        }
}

// ---- F.interpolate(mode="linear") (UpSampleLinear1d.cu) on contiguous [N, C, W] ------------------

template <typename scalar_t, typename accscalar_t>
__global__ void upsample_linear1d_out_frame(const int n, const accscalar_t rwidth, const bool align_corners,
                                            const scalar_t* idata, scalar_t* odata, int batchsize, int channels,
                                            int width1, int width2) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < n) {
        const int w2 = index % width2;
        if (width1 == width2) {
            const int w1 = w2;
            for (int n = 0; n < batchsize; n++)
                for (int c = 0; c < channels; ++c)
                    odata[(int64_t(n) * channels + c) * width2 + w2] = idata[(int64_t(n) * channels + c) * width1 + w1];
            return;
        }
        accscalar_t w1r;
        if (align_corners) {
            w1r = rwidth * w2;
        } else {
            accscalar_t src_idx = rwidth * (w2 + static_cast<accscalar_t>(0.5)) - static_cast<accscalar_t>(0.5);
            w1r = src_idx < static_cast<accscalar_t>(0) ? static_cast<accscalar_t>(0) : src_idx;
        }
        const int w1 = w1r;
        const int w1p = (w1 < width1 - 1) ? 1 : 0;
        const accscalar_t w1lambda = w1r - w1;
        const accscalar_t w0lambda = static_cast<accscalar_t>(1) - w1lambda;
        for (int n = 0; n < batchsize; n++) {
            for (int c = 0; c < channels; ++c) {
                const scalar_t* row = idata + (int64_t(n) * channels + c) * width1;
                const accscalar_t val = w0lambda * row[w1] + w1lambda * row[w1 + w1p];
                odata[(int64_t(n) * channels + c) * width2 + w2] = static_cast<scalar_t>(val);
            }
        }
    }
}

// ---- elementwise (GeluCUDAKernelImpl none; layout copies) ---------------------------------------

// GELU (erf) on rows of `cols` elements read with `in_stride`, written contiguous.
template <typename T>
__global__ void gelu_erf(const T* x, T* y, int64_t rows, int64_t cols, int64_t in_stride) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= rows * cols) return;
    using opmath_t = float;
    constexpr opmath_t kAlpha = M_SQRT1_2;
    const opmath_t v = static_cast<opmath_t>(x[(i / cols) * in_stride + i % cols]);
    y[i] = static_cast<T>(v * opmath_t(0.5) * (opmath_t(1) + ::erf(v * kAlpha)));
}

// out (contiguous, sizes[0..4]) = src viewed with element strides[0..4].
template <typename T>
__global__ void strided_copy(const T* src, T* out, int64_t s0, int64_t s1, int64_t s2, int64_t s3, int64_t d1, int64_t d2,
                             int64_t d3, int64_t n) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int64_t i3 = i % d3, i2 = (i / d3) % d2, i1 = (i / (d3 * d2)) % d1, i0 = i / (d3 * d2 * d1);
    out[i] = src[i0 * s0 + i1 * s1 + i2 * s2 + i3 * s3];
}

// out[c, l] = x[c, l] + bias[c] (conv bias add_, alpha 1) for fp32.
__global__ void channel_bias_f32(float* x, const float* bias, int64_t inner, int64_t n) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float alpha = 1.0f;
    x[i] = x[i] + alpha * bias[i / inner];
}

__global__ void add_f32(const float* a, const float* b, float* out, int64_t n) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float alpha = 1.0f;
    out[i] = a[i] + alpha * b[i];
}

__global__ void mul_scalar_f32(float* x, float s, int64_t n) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) x[i] = x[i] * s;
}

// ---- CLIP visual preprocessing (UpSampleBicubic2d.cu, torchvision Normalize) -------------------

template <typename accscalar_t>
__device__ __forceinline__ accscalar_t cubic_convolution1(accscalar_t x, accscalar_t A) {
    return ((A + 2) * x - (A + 3)) * x * x + 1;
}
template <typename accscalar_t>
__device__ __forceinline__ accscalar_t cubic_convolution2(accscalar_t x, accscalar_t A) {
    return ((A * x - 5 * A) * x + 8 * A) * x - 4 * A;
}
template <typename accscalar_t>
__device__ __forceinline__ void get_cubic_upsampling_coefficients(accscalar_t coeffs[4], accscalar_t t) {
    accscalar_t A = -0.75;
    accscalar_t x1 = t;
    coeffs[0] = cubic_convolution2<accscalar_t>(x1 + 1.0, A);
    coeffs[1] = cubic_convolution1<accscalar_t>(x1, A);
    accscalar_t x2 = 1.0 - t;
    coeffs[2] = cubic_convolution1<accscalar_t>(x2, A);
    coeffs[3] = cubic_convolution2<accscalar_t>(x2 + 1.0, A);
}
template <typename scalar_t, typename accscalar_t>
__device__ __forceinline__ accscalar_t cubic_interp1d(scalar_t x0, scalar_t x1, scalar_t x2, scalar_t x3, accscalar_t t) {
    accscalar_t coeffs[4];
    get_cubic_upsampling_coefficients<accscalar_t>(coeffs, t);
    return x0 * coeffs[0] + x1 * coeffs[1] + x2 * coeffs[2] + x3 * coeffs[3];
}
template <typename scalar_t>
__device__ __forceinline__ scalar_t bounded(const scalar_t* plane, int height, int width, int y, int x) {
    int access_y = max(min(y, height - 1), 0);
    int access_x = max(min(x, width - 1), 0);
    return plane[int64_t(access_y) * width + access_x];
}

// upsample_bicubic2d_out_frame on contiguous [planes, H, W] (align_corners False).
template <typename scalar_t, typename accscalar_t>
__global__ void upsample_bicubic2d_out_frame(const int num_elements, const accscalar_t height_scale,
                                             const accscalar_t width_scale, const scalar_t* idata, scalar_t* odata,
                                             int planes, int input_height, int input_width, int output_height,
                                             int output_width) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= num_elements) return;
    const int output_x = index % output_width;
    const int output_y = index / output_width;
    accscalar_t real_x = width_scale * (output_x + static_cast<accscalar_t>(0.5)) - static_cast<accscalar_t>(0.5);
    int in_x = floorf(real_x);
    accscalar_t t_x = real_x - in_x;
    accscalar_t real_y = height_scale * (output_y + static_cast<accscalar_t>(0.5)) - static_cast<accscalar_t>(0.5);
    int in_y = floorf(real_y);
    accscalar_t t_y = real_y - in_y;
    for (int p = 0; p < planes; p++) {
        const scalar_t* plane = idata + int64_t(p) * input_height * input_width;
        accscalar_t coefficients[4];
        for (int k = 0; k < 4; k++) {
            coefficients[k] = cubic_interp1d(bounded(plane, input_height, input_width, in_y - 1 + k, in_x - 1),
                                             bounded(plane, input_height, input_width, in_y - 1 + k, in_x + 0),
                                             bounded(plane, input_height, input_width, in_y - 1 + k, in_x + 1),
                                             bounded(plane, input_height, input_width, in_y - 1 + k, in_x + 2), t_x);
        }
        odata[(int64_t(p) * output_height + output_y) * output_width + output_x] = static_cast<scalar_t>(
            cubic_interp1d(coefficients[0], coefficients[1], coefficients[2], coefficients[3], t_y));
    }
}

// videos.mul_(0.5).add_(0.5), Normalize(mean, std) (sub_, div_ by fp32 [3,1,1]), then the
// autocast fp16 cast of the conv input; x: [3, hw] fp32.
__global__ void clip_normalize(const float* x, __half* out, int64_t hw, float m0, float m1, float m2, float s0, float s1,
                               float s2) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= 3 * hw) return;
    const int c = int(i / hw);
    const float mean = c == 0 ? m0 : c == 1 ? m1 : m2, std = c == 0 ? s0 : c == 1 ? s1 : s2;
    float v = __fmul_rn(x[i], 0.5f);
    v = __fadd_rn(v, 0.5f);
    v = __fsub_rn(v, mean);
    v = __fdiv_rn(v, std);
    out[i] = __float2half_rn(v);
}

// x[t, c] = cat(cls, patches^T)[t, c] + pos[t, c] (fp16 add in fp32 opmath); patches [C, P].
__global__ void clip_embed(const __half* patches, const __half* cls, const __half* pos, __half* out, int64_t c, int64_t p) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= (p + 1) * c) return;
    const int64_t t = i / c, ch = i % c;
    const __half src = t == 0 ? cls[ch] : patches[ch * p + (t - 1)];
    out[i] = __float2half_rn(__half2float(src) + 1.0f * __half2float(pos[i]));
}

__global__ void add_f16(const __half* a, const __half* b, __half* out, int64_t n) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2half_rn(__half2float(a[i]) + 1.0f * __half2float(b[i]));
}
__global__ void f16_to_f32(const __half* x, float* y, int64_t n) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) y[i] = __half2float(x[i]);
}
__global__ void f32_to_f16(const float* x, __half* y, int64_t n) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) y[i] = __float2half_rn(x[i]);
}

// ---- umt5 encoder (wan/modules/t5.py, bf16 eager ops) --------------------------------------------

using bf16 = __nv_bfloat16;
__device__ __forceinline__ float bf(bf16 v) { return __bfloat162float(v); }
__device__ __forceinline__ bf16 rb(float v) { return __float2bfloat16_rn(v); }

// x[b, i] = token_embedding[ids[b, i]]
__global__ void t5_embed(const int64_t* ids, const bf16* table, bf16* x, int64_t tokens, int64_t dim) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= tokens * dim) return;
    x[i] = table[ids[i / dim] * dim + i % dim];
}

// T5RelativeEmbedding(bidirectional, 32 buckets, max_dist 128) of an [L, L] grid -> bias [H, L, L]:
// the bucket arithmetic of _relative_position_bucket (int64 / fp32 eager ops), then the gather.
__global__ void t5_position_bias(const bf16* table, bf16* bias, int64_t len, int64_t heads, float inv_exact,
                                 float inv_log_ratio, float span) {
    const int64_t n = heads * len * len;
    const int64_t idx = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    const int64_t j = idx % len, i = (idx / len) % len, h = idx / (len * len);
    int64_t rel = j - i;
    int64_t bucket = (rel > 0 ? int64_t(1) : int64_t(0)) * 16;
    rel = rel < 0 ? -rel : rel;
    const float scaled = static_cast<float>(rel) * inv_exact;
    const float logged = ::log(scaled);
    const float ratio = logged * inv_log_ratio;
    const int64_t large_raw = static_cast<int64_t>(ratio * span) + 8;
    const int64_t large = large_raw < 15 ? large_raw : 15;
    bucket += rel < 8 ? rel : large;
    bias[idx] = table[bucket * heads + h];
}

// x.float().pow(2)
__global__ void t5_square(const bf16* x, float* sq, int64_t n) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = bf(x[i]);
    sq[i] = v * v;
}

// weight * (x * rsqrt(mean + eps)).type_as(weight)
__global__ void t5_norm_apply(const bf16* x, const float* mean, const bf16* weight, bf16* out, int64_t dim, float eps, int64_t n) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float alpha = 1.0f;
    const float r = ::rsqrt(mean[i / dim] + alpha * eps);
    const bf16 y = rb(bf(x[i]) * r);
    out[i] = rb(bf(weight[i % dim]) * bf(y));
}

// [B, L, H, C] -> [B, H, L, C] (t = 0) or [B, H, C, L] (t = 1), contiguous.
__global__ void t5_heads(const bf16* x, bf16* out, int64_t b, int64_t len, int64_t heads, int64_t c, int t) {
    const int64_t n = b * len * heads * c;
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int64_t bb, hh, ll, cc;
    if (t == 0) { cc = i % c; ll = (i / c) % len; hh = (i / (c * len)) % heads; bb = i / (c * len * heads); }
    else { ll = i % len; cc = (i / len) % c; hh = (i / (len * c)) % heads; bb = i / (len * c * heads); }
    out[i] = x[((bb * len + ll) * heads + hh) * c + cc];
}

// [B*H, L, C] -> [B, L, H*C]
__global__ void t5_merge(const bf16* x, bf16* out, int64_t b, int64_t len, int64_t heads, int64_t c) {
    const int64_t n = b * len * heads * c;
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int64_t cc = i % c, hh = (i / c) % heads, ll = (i / (c * heads)) % len, bb = i / (c * heads * len);
    out[i] = x[((bb * heads + hh) * len + ll) * c + cc];
}

// (scores + attn_bias).float(): attn_bias = zeros + pos_bias, masked_fill(mask == 0, bf16 min).
__global__ void t5_scores(const bf16* scores, const bf16* pos, const int64_t* mask, float* out, int64_t b, int64_t heads,
                          int64_t len) {
    const int64_t n = b * heads * len * len;
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int64_t j = i % len, hij = i % (heads * len * len), bb = i / (heads * len * len);
    const bf16 zero = rb(0.0f);
    bf16 bias = rb(bf(zero) + 1.0f * bf(pos[hij]));
    if (mask[bb * len + j] == 0) bias = rb(-3.3895313892515355e+38f);
    out[i] = bf(rb(bf(scores[i]) + 1.0f * bf(bias)));
}

__global__ void f32_to_bf16(const float* x, bf16* y, int64_t n) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) y[i] = rb(x[i]);
}

__global__ void add_bf16(const bf16* a, const bf16* b, bf16* out, int64_t n) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) out[i] = rb(bf(a[i]) + 1.0f * bf(b[i]));
}

// fc1(x) * GELU(gate(x)) with GELU = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x**3))), each
// eager op rounded to bf16 (pow(x, 3) multiplies in BFloat16 arithmetic).
__global__ void t5_gated_gelu(const bf16* fc1, const bf16* gate, bf16* out, float c_tanh, int64_t n) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const bf16 g = gate[i];
    const bf16 half = rb(bf(g) * 0.5f);
    const bf16 cube = rb(bf(rb(bf(g) * bf(g))) * bf(g));
    const bf16 scaled = rb(bf(cube) * 0.044715f);
    const bf16 inner = rb(bf(g) + 1.0f * bf(scaled));
    const bf16 arg = rb(bf(inner) * c_tanh);
    const bf16 th = rb(::tanhf(bf(arg)));
    const bf16 one_plus = rb(bf(th) + 1.0f);
    const bf16 act = rb(bf(half) * bf(one_plus));
    out[i] = rb(bf(fc1[i]) * bf(act));
}

// upsample_bilinear2d_out_frame (UpSampleBilinear2d.cu, NCHW) on contiguous [planes, H, W].
template <typename scalar_t, typename accscalar_t>
__launch_bounds__(1024)  // C10_LAUNCH_BOUNDS_1(1024)
__global__ void upsample_bilinear2d_out_frame(const int n, const accscalar_t rheight, const accscalar_t rwidth,
                                              const bool align_corners, const scalar_t* idata, scalar_t* odata, int planes,
                                              int height1, int width1, int height2, int width2) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < n) {
        const int w2 = index % width2;
        const int h2 = index / width2;
        accscalar_t h1r, w1r;
        if (align_corners) {
            h1r = rheight * h2;
            w1r = rwidth * w2;
        } else {
            accscalar_t hs = rheight * (h2 + static_cast<accscalar_t>(0.5)) - static_cast<accscalar_t>(0.5);
            h1r = hs < static_cast<accscalar_t>(0) ? static_cast<accscalar_t>(0) : hs;
            accscalar_t ws = rwidth * (w2 + static_cast<accscalar_t>(0.5)) - static_cast<accscalar_t>(0.5);
            w1r = ws < static_cast<accscalar_t>(0) ? static_cast<accscalar_t>(0) : ws;
        }
        const int h1 = h1r;
        const int h1p = (h1 < height1 - 1) ? 1 : 0;
        const accscalar_t h1lambda = h1r - h1;
        const accscalar_t h0lambda = static_cast<accscalar_t>(1) - h1lambda;
        const int w1 = w1r;
        const int w1p = (w1 < width1 - 1) ? 1 : 0;
        const accscalar_t w1lambda = w1r - w1;
        const accscalar_t w0lambda = static_cast<accscalar_t>(1) - w1lambda;
        for (int c = 0; c < planes; ++c) {
            const scalar_t* plane = idata + int64_t(c) * height1 * width1;
            auto at = [&](int y, int x) { return static_cast<accscalar_t>(plane[int64_t(y) * width1 + x]); };
            const accscalar_t val = h0lambda * (w0lambda * at(h1, w1) + w1lambda * at(h1, w1 + w1p)) +
                                    h1lambda * (w0lambda * at(h1 + h1p, w1) + w1lambda * at(h1 + h1p, w1 + w1p));
            odata[(int64_t(c) * height2 + h2) * width2 + w2] = static_cast<scalar_t>(val);
        }
    }
}

template <typename T, int L2E>
void launch_softmax(T* dst, const T* src, int elements, int stride, int batch, cudaStream_t stream) {
    constexpr int next_power_of_two = 1 << L2E;
    const int warp_size = next_power_of_two < 32 ? next_power_of_two : 32;
    const int batches_per_warp = next_power_of_two <= 128 ? 2 : 1;
    constexpr int threads_per_block = 128;
    const int warps_per_block = threads_per_block / warp_size;
    const int batches_per_block = warps_per_block * batches_per_warp;
    const int nblocks = (batch + batches_per_block - 1) / batches_per_block;
    dim3 threads(warp_size, warps_per_block, 1);
    softmax_warp_forward<T, T, float, L2E, false, false, 32><<<nblocks, threads, 0, stream>>>(dst, src, batch, stride,
                                                                                             elements, nullptr, -1, false);
}

}  // namespace

extern "C" {

int wan_softmax_long_f32(const float* x, float* y, int64_t rows, int64_t cols, void* stream);

// host_softmax(dim=-1) for a contiguous fp32 [rows, cols]: the persistent warp softmax up to 2048
// columns, torch's block kernels (softmax.cu) beyond.
int wan_softmax_f32(const float* x, float* y, int64_t rows, int64_t cols, void* stream) {
    if (cols > 2048) return wan_softmax_long_f32(x, y, rows, cols, stream);
    if (cols <= 0) return -1;
    auto st = static_cast<cudaStream_t>(stream);
    const int64_t chunk = (int64_t(1) << 30) / cols;
    for (int64_t done = 0; done < rows; done += chunk) {
        const int batch = int(std::min<int64_t>(rows - done, chunk));
        const float* src = x + done * cols;
        float* dst = y + done * cols;
        const int c = int(cols);
        switch (log2_ceil(c)) {
        case 0: launch_softmax<float, 0>(dst, src, c, c, batch, st); break;
        case 1: launch_softmax<float, 1>(dst, src, c, c, batch, st); break;
        case 2: launch_softmax<float, 2>(dst, src, c, c, batch, st); break;
        case 3: launch_softmax<float, 3>(dst, src, c, c, batch, st); break;
        case 4: launch_softmax<float, 4>(dst, src, c, c, batch, st); break;
        case 5: launch_softmax<float, 5>(dst, src, c, c, batch, st); break;
        case 6: launch_softmax<float, 6>(dst, src, c, c, batch, st); break;
        case 7: launch_softmax<float, 7>(dst, src, c, c, batch, st); break;
        case 8: launch_softmax<float, 8>(dst, src, c, c, batch, st); break;
        case 9: launch_softmax<float, 9>(dst, src, c, c, batch, st); break;
        case 10: launch_softmax<float, 10>(dst, src, c, c, batch, st); break;
        default: launch_softmax<float, 11>(dst, src, c, c, batch, st);
        }
    }
    return cudaGetLastError();
}

// native_group_norm(X [N, C, HxW] fp32, gamma, beta) with `group` groups; scratch: 2 * N * group
// + 2 * N * C floats.
int wan_group_norm_f32(const float* x, const float* gamma, const float* beta, float* y, int64_t n, int64_t c,
                       int64_t hw, int64_t group, float eps, float* scratch, void* stream) {
    auto st = static_cast<cudaStream_t>(stream);
    const int64_t d = c / group;
    float* mean = scratch;
    float* rstd = mean + n * group;
    float* a = rstd + n * group;
    float* b = a + n * c;
    const int threads = d * hw < at::native::cuda_utils::kCUDABlockReduceNumThreads ? 32
                                                                                     : at::native::cuda_utils::kCUDABlockReduceNumThreads;
    RowwiseMomentsCUDAKernel<float, float><<<unsigned(n * group), threads, 0, st>>>(d * hw, eps, x, mean, rstd, mean, rstd);
    ComputeFusedParamsCUDAKernel<float, float><<<unsigned((n * c + 255) / 256), 256, 0, st>>>(n, c, group, mean, rstd, gamma,
                                                                                            beta, a, b);
    group_norm_apply<<<blocks(n * c * hw), kThreads, 0, st>>>(x, a, b, y, hw, n * c * hw);
    return cudaGetLastError();
}

// torch._weight_norm(v, g, dim = last): w = g * v / ||v|| with norms over the other dims.
int wan_weight_norm_last_dim_f32(const float* v, const float* g, float* w, float* norms, int fast_dim,
                                 int slower_dims, void* stream) {
    weight_norm_fwd_last_dim_kernel<float, float><<<(fast_dim + TILE_W - 1) / TILE_W, dim3(TILE_W, TILE_H),
                                                    (TILE_W * TILE_H + TILE_W) * sizeof(float),
                                                    static_cast<cudaStream_t>(stream)>>>(w, norms, v, g, fast_dim,
                                                                                         slower_dims);
    return cudaGetLastError();
}

int wan_upsample_linear1d_f32(const float* x, float* y, int batch, int channels, int width1, int width2,
                              int align_corners, void* stream) {
    float rwidth;
    if (align_corners) rwidth = width2 > 1 ? (float)(width1 - 1) / (width2 - 1) : 0.f;
    else rwidth = (float)width1 / width2;
    const int num_threads = 512;
    upsample_linear1d_out_frame<float, float><<<(width2 + num_threads - 1) / num_threads, num_threads, 0,
                                                static_cast<cudaStream_t>(stream)>>>(
        width2, rwidth, align_corners != 0, x, y, batch, channels, width1, width2);
    return cudaGetLastError();
}

// dtype: 0 fp32, 1 fp16, 2 bf16.
int wan_gelu_erf(const void* x, void* y, int64_t rows, int64_t cols, int64_t in_stride, int dtype, void* stream) {
    auto st = static_cast<cudaStream_t>(stream);
    const unsigned g = blocks(rows * cols);
    if (dtype == 0) gelu_erf<<<g, kThreads, 0, st>>>((const float*)x, (float*)y, rows, cols, in_stride);
    else if (dtype == 1) gelu_erf<<<g, kThreads, 0, st>>>((const __half*)x, (__half*)y, rows, cols, in_stride);
    else gelu_erf<<<g, kThreads, 0, st>>>((const __nv_bfloat16*)x, (__nv_bfloat16*)y, rows, cols, in_stride);
    return cudaGetLastError();
}

// Contiguous copy of a 4-d strided view (element strides); elem: 2 or 4 bytes.
int wan_strided_copy(const void* src, void* out, const int64_t* sizes, const int64_t* strides, int elem, void* stream) {
    const int64_t n = sizes[0] * sizes[1] * sizes[2] * sizes[3];
    auto st = static_cast<cudaStream_t>(stream);
    if (elem == 4)
        strided_copy<<<blocks(n), kThreads, 0, st>>>((const float*)src, (float*)out, strides[0], strides[1], strides[2],
                                                     strides[3], sizes[1], sizes[2], sizes[3], n);
    else
        strided_copy<<<blocks(n), kThreads, 0, st>>>((const uint16_t*)src, (uint16_t*)out, strides[0], strides[1],
                                                     strides[2], strides[3], sizes[1], sizes[2], sizes[3], n);
    return cudaGetLastError();
}

int wan_channel_bias_f32(float* x, const float* bias, int64_t channels, int64_t inner, void* stream) {
    channel_bias_f32<<<blocks(channels * inner), kThreads, 0, static_cast<cudaStream_t>(stream)>>>(x, bias, inner,
                                                                                                  channels * inner);
    return cudaGetLastError();
}

int wan_add_f32(const float* a, const float* b, float* out, int64_t n, void* stream) {
    add_f32<<<blocks(n), kThreads, 0, static_cast<cudaStream_t>(stream)>>>(a, b, out, n);
    return cudaGetLastError();
}

int wan_mul_scalar_f32(float* x, double s, int64_t n, void* stream) {
    mul_scalar_f32<<<blocks(n), kThreads, 0, static_cast<cudaStream_t>(stream)>>>(x, float(s), n);
    return cudaGetLastError();
}

// at::bmm(batch1 [B, M, K], batch2 [B, K, N]) -> out contiguous [B, M, N] (baddbmm_out_cuda_impl
// with beta 0): strides in elements; dtype 0 fp32 (cublasSgemmStridedBatched), 2 bf16
// (cublasGemmStridedBatchedEx, compute 32F).
int wan_bmm(void* handle, int dtype, const void* b1, const int64_t* s1, const void* b2, const int64_t* s2, void* out,
            int64_t batch, int64_t m_rows, int64_t k_inner, int64_t n_cols) {
    // Contiguous result: transpose_result = true, leading_dim = 1.
    const int64_t m = n_cols, n = m_rows, k = k_inner;
    auto prepare = [](const int64_t* strides, int64_t rows, int64_t cols, bool& trans, int64_t& ld) {
        // fast_dim = 2, leading_dim = 1 (transpose_result).
        if (strides[2] == 1 && strides[1] >= std::max<int64_t>(1, rows)) {
            trans = false;
            ld = strides[1];
            return true;
        }
        if (strides[1] == 1 && strides[2] >= std::max<int64_t>(1, cols)) {
            trans = true;
            ld = strides[2];
            return true;
        }
        return false;  // torch would copy to contiguous; callers pass views that never need it
    };
    bool ta = false, tb = false;
    int64_t lda = 0, ldb = 0;
    if (!prepare(s2, m, k, ta, lda) || !prepare(s1, k, n, tb, ldb)) return -2;
    int64_t ldc = n_cols;
    // _cublasAdjustLdLevel3
    if (n <= 1) ldc = std::max<int64_t>(m, 1);
    if (ta) { if (m <= 1) lda = std::max<int64_t>(k, 1); }
    else { if (k <= 1) lda = std::max<int64_t>(m, 1); }
    if (tb) { if (k <= 1) ldb = std::max<int64_t>(n, 1); }
    else { if (n <= 1) ldb = std::max<int64_t>(k, 1); }
    const cublasOperation_t opa = ta ? CUBLAS_OP_T : CUBLAS_OP_N, opb = tb ? CUBLAS_OP_T : CUBLAS_OP_N;
    const float alpha = 1.0f, beta = 0.0f;
    auto h = static_cast<cublasHandle_t>(handle);
    const int64_t stride_c = m_rows * n_cols;
    if (dtype == 0)
        return int(cublasSgemmStridedBatched(h, opa, opb, int(m), int(n), int(k), &alpha, (const float*)b2, int(lda), s2[0],
                                             (const float*)b1, int(ldb), s1[0], &beta, (float*)out, int(ldc), stride_c,
                                             int(batch)));
    return int(cublasGemmStridedBatchedEx(h, opa, opb, int(m), int(n), int(k), &alpha, b2, CUDA_R_16BF, int(lda), s2[0], b1,
                                          CUDA_R_16BF, int(ldb), s1[0], &beta, out, CUDA_R_16BF, int(ldc), stride_c,
                                          int(batch), CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

// F.interpolate(mode="bicubic", align_corners=False) on contiguous fp32 [planes, H, W].
int wan_upsample_bicubic2d_f32(const float* x, float* y, int planes, int in_h, int in_w, int out_h, int out_w,
                               void* stream) {
    const float rheight = (float)in_h / out_h, rwidth = (float)in_w / out_w;
    const int n = out_h * out_w, threads = 1024;
    upsample_bicubic2d_out_frame<float, float><<<(n + threads - 1) / threads, threads, 0, static_cast<cudaStream_t>(stream)>>>(
        n, rheight, rwidth, x, y, planes, in_h, in_w, out_h, out_w);
    return cudaGetLastError();
}

int wan_clip_normalize(const float* x, void* out, int64_t hw, const float* mean, const float* std, void* stream) {
    clip_normalize<<<blocks(3 * hw), kThreads, 0, static_cast<cudaStream_t>(stream)>>>(
        x, (__half*)out, hw, mean[0], mean[1], mean[2], std[0], std[1], std[2]);
    return cudaGetLastError();
}

int wan_clip_embed(const void* patches, const void* cls, const void* pos, void* out, int64_t c, int64_t p, void* stream) {
    clip_embed<<<blocks((p + 1) * c), kThreads, 0, static_cast<cudaStream_t>(stream)>>>(
        (const __half*)patches, (const __half*)cls, (const __half*)pos, (__half*)out, c, p);
    return cudaGetLastError();
}

int wan_add_f16(const void* a, const void* b, void* out, int64_t n, void* stream) {
    add_f16<<<blocks(n), kThreads, 0, static_cast<cudaStream_t>(stream)>>>((const __half*)a, (const __half*)b, (__half*)out, n);
    return cudaGetLastError();
}

int wan_f16_to_f32(const void* x, float* y, int64_t n, void* stream) {
    f16_to_f32<<<blocks(n), kThreads, 0, static_cast<cudaStream_t>(stream)>>>((const __half*)x, y, n);
    return cudaGetLastError();
}

int wan_f32_to_f16(const float* x, void* y, int64_t n, void* stream) {
    f32_to_f16<<<blocks(n), kThreads, 0, static_cast<cudaStream_t>(stream)>>>(x, (__half*)y, n);
    return cudaGetLastError();
}

int wan_t5_embed(const int64_t* ids, const void* table, void* x, int64_t tokens, int64_t dim, void* stream) {
    t5_embed<<<blocks(tokens * dim), kThreads, 0, static_cast<cudaStream_t>(stream)>>>(ids, (const bf16*)table, (bf16*)x,
                                                                                        tokens, dim);
    return cudaGetLastError();
}

int wan_t5_position_bias(const void* table, void* bias, int64_t len, int64_t heads, void* stream) {
    // rel_pos.float() / max_exact (8) -> * float(1/8); / math.log(max_dist / max_exact) -> * float(1/log(16));
    // * (num_buckets - max_exact) (8).
    const float inv_exact = float(1.0 / 8), inv_log_ratio = float(1.0 / std::log(128.0 / 8)), span = 8.0f;
    t5_position_bias<<<blocks(heads * len * len), kThreads, 0, static_cast<cudaStream_t>(stream)>>>(
        (const bf16*)table, (bf16*)bias, len, heads, inv_exact, inv_log_ratio, span);
    return cudaGetLastError();
}

int wan_t5_square(const void* x, float* sq, int64_t n, void* stream) {
    t5_square<<<blocks(n), kThreads, 0, static_cast<cudaStream_t>(stream)>>>((const bf16*)x, sq, n);
    return cudaGetLastError();
}

int wan_t5_norm_apply(const void* x, const float* mean, const void* weight, void* out, int64_t dim, double eps, int64_t n,
                      void* stream) {
    t5_norm_apply<<<blocks(n), kThreads, 0, static_cast<cudaStream_t>(stream)>>>((const bf16*)x, mean, (const bf16*)weight,
                                                                                (bf16*)out, dim, float(eps), n);
    return cudaGetLastError();
}

int wan_t5_heads(const void* x, void* out, int64_t b, int64_t len, int64_t heads, int64_t c, int transpose, void* stream) {
    t5_heads<<<blocks(b * len * heads * c), kThreads, 0, static_cast<cudaStream_t>(stream)>>>((const bf16*)x, (bf16*)out, b,
                                                                                             len, heads, c, transpose);
    return cudaGetLastError();
}

int wan_t5_merge(const void* x, void* out, int64_t b, int64_t len, int64_t heads, int64_t c, void* stream) {
    t5_merge<<<blocks(b * len * heads * c), kThreads, 0, static_cast<cudaStream_t>(stream)>>>((const bf16*)x, (bf16*)out, b,
                                                                                             len, heads, c);
    return cudaGetLastError();
}

int wan_t5_scores(const void* scores, const void* pos, const int64_t* mask, float* out, int64_t b, int64_t heads,
                  int64_t len, void* stream) {
    t5_scores<<<blocks(b * heads * len * len), kThreads, 0, static_cast<cudaStream_t>(stream)>>>(
        (const bf16*)scores, (const bf16*)pos, mask, out, b, heads, len);
    return cudaGetLastError();
}

int wan_f32_to_bf16_enc(const float* x, void* y, int64_t n, void* stream) {
    f32_to_bf16<<<blocks(n), kThreads, 0, static_cast<cudaStream_t>(stream)>>>(x, (bf16*)y, n);
    return cudaGetLastError();
}

int wan_add_bf16_enc(const void* a, const void* b, void* out, int64_t n, void* stream) {
    add_bf16<<<blocks(n), kThreads, 0, static_cast<cudaStream_t>(stream)>>>((const bf16*)a, (const bf16*)b, (bf16*)out, n);
    return cudaGetLastError();
}

int wan_t5_gated_gelu(const void* fc1, const void* gate, void* out, int64_t n, void* stream) {
    const float c_tanh = float(std::sqrt(2.0 / M_PI));
    t5_gated_gelu<<<blocks(n), kThreads, 0, static_cast<cudaStream_t>(stream)>>>((const bf16*)fc1, (const bf16*)gate,
                                                                                (bf16*)out, c_tanh, n);
    return cudaGetLastError();
}

// F.interpolate(mode="bilinear", align_corners=False) on contiguous bf16 [planes, H, W].
int wan_upsample_bilinear2d_bf16(const void* x, void* y, int planes, int in_h, int in_w, int out_h, int out_w, void* stream) {
    const float rheight = (float)in_h / out_h, rwidth = (float)in_w / out_w;
    const int n = out_h * out_w, threads = 1024;
    upsample_bilinear2d_out_frame<__nv_bfloat16, float><<<(n + threads - 1) / threads, threads, 0,
                                                          static_cast<cudaStream_t>(stream)>>>(
        n, rheight, rwidth, false, (const __nv_bfloat16*)x, (__nv_bfloat16*)y, planes, in_h, in_w, out_h, out_w);
    return cudaGetLastError();
}
}
