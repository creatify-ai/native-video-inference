// Eager PyTorch 2.13 elementwise operators used by the Wan DiT, with the pinned
// source expressions verbatim so nvcc's default contraction matches ATen's build
// (no fast math). Per-element results do not depend on ATen's vectorization.
// Source: pytorch cf30153c4c131c8164ee7798e5022d810682e2cb
//   aten/src/ATen/native/cuda/ActivationGeluKernel.cu (GeluType::Tanh)
//   aten/src/ATen/native/cuda/BinaryAddSubKernel (AddFunctor: a + alpha * b, alpha = 1)
//   aten/src/ATen/native/cuda/ActivationSiluKernel.cu, PowKernel.cu/Pow.cuh (CPU-scalar base),
//   UnaryGeometricCos/SinKernel.cu, layer_norm_kernel.cu (vectorized_layer_norm_kernel)
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <mutex>

namespace {
constexpr int kThreads = 256;
inline unsigned blocks(int64_t n) { return unsigned((n + kThreads - 1) / kThreads); }

__device__ __forceinline__ __nv_bfloat16 gelu_tanh_one(__nv_bfloat16 x) {
    using opmath_t = float;
    constexpr opmath_t kBeta = M_SQRT2 * M_2_SQRTPI * opmath_t(0.5);
    constexpr opmath_t kKappa = 0.044715;
    auto x_cube = static_cast<opmath_t>(x) * static_cast<opmath_t>(x) * static_cast<opmath_t>(x);
    auto inner = kBeta * (static_cast<opmath_t>(x) + kKappa * x_cube);
    return __float2bfloat16_rn(opmath_t(0.5) * static_cast<opmath_t>(x) * (opmath_t(1) + ::tanhf(inner)));
}
__global__ void gelu_tanh_bf16(const __nv_bfloat16* x_in, __nv_bfloat16* out, int64_t n) {
    int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = gelu_tanh_one(x_in[i]);
}
// The same per-element expression, 8 elements per thread through 16-byte loads and stores
// (ATen vectorizes too; per-element results do not depend on it).
__global__ void gelu_tanh_bf16_vec8(const uint4* x_in, uint4* out, int64_t n8) {
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n8; i += int64_t(gridDim.x) * blockDim.x) {
        uint4 v = x_in[i];
        __nv_bfloat16* e = reinterpret_cast<__nv_bfloat16*>(&v);
        #pragma unroll
        for (int k = 0; k < 8; ++k) e[k] = gelu_tanh_one(e[k]);
        out[i] = v;
    }
}

// GELU of a bf16 depends only on its 16 bits: g_gelu_lut[bits] holds gelu_tanh_one of every bf16
// (filled once per device with that function), and the table kernel looks each element up instead
// of evaluating tanhf; the values are gelu_tanh_one's by construction.
__device__ uint16_t g_gelu_lut[65536];
__global__ void gelu_lut_fill() {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < 65536) g_gelu_lut[i] = __bfloat16_as_ushort(gelu_tanh_one(__ushort_as_bfloat16(uint16_t(i))));
}
// One block per SM with the table in shared memory; each thread keeps four 16-byte loads in flight.
constexpr int kLutThreads = 1024, kLutDepth = 4;
__global__ void __launch_bounds__(kLutThreads) gelu_tanh_bf16_lut(const uint4* x, uint4* out, int64_t n8) {
    extern __shared__ uint4 table4[];
    for (int i = threadIdx.x; i < 65536 / 8; i += blockDim.x) table4[i] = reinterpret_cast<const uint4*>(g_gelu_lut)[i];
    __syncthreads();
    const uint16_t* table = reinterpret_cast<const uint16_t*>(table4);
    const int64_t step = int64_t(gridDim.x) * blockDim.x;
    for (int64_t i0 = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i0 < n8; i0 += step * kLutDepth) {
        uint4 v[kLutDepth];
        #pragma unroll
        for (int d = 0; d < kLutDepth; ++d) if (i0 + d * step < n8) v[d] = x[i0 + d * step];
        #pragma unroll
        for (int d = 0; d < kLutDepth; ++d) {
            if (i0 + d * step >= n8) break;
            uint16_t* e = reinterpret_cast<uint16_t*>(&v[d]);
            #pragma unroll
            for (int k = 0; k < 8; ++k) e[k] = table[e[k]];
            out[i0 + d * step] = v[d];
        }
    }
}
// The table on the current device, filled (stream-ordered) before its first use there.
bool gelu_lut_ready(cudaStream_t stream) {
    static std::mutex m;
    static bool filled[64] = {};
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess || device < 0 || device >= 64) return false;
    std::lock_guard<std::mutex> lock(m);
    if (!filled[device]) {
        if (cudaFuncSetAttribute(gelu_tanh_bf16_lut, cudaFuncAttributeMaxDynamicSharedMemorySize, 65536 * 2) != cudaSuccess) {
            cudaGetLastError(); return false;
        }
        gelu_lut_fill<<<256, 256, 0, stream>>>();
        if (cudaGetLastError() != cudaSuccess) return false;
        filled[device] = true;
    }
    return true;
}

// bf16 + bf16 (AddFunctor in fp32 opmath, alpha = 1), 8 elements per thread through 16-byte
// accesses; the per-element expression is the scalar one.
__device__ __forceinline__ __nv_bfloat16 add_one(__nv_bfloat16 a, __nv_bfloat16 b) {
    const float alpha = 1.0f;
    return __float2bfloat16_rn(static_cast<float>(a) + alpha * static_cast<float>(b));
}
__global__ void add_bf16(const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* out, int64_t n) {
    int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) out[i] = add_one(a[i], b[i]);
}
__global__ void add_bf16_vec8(const uint4* a, const uint4* b, uint4* out, int64_t n8) {
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n8; i += int64_t(gridDim.x) * blockDim.x) {
        uint4 va = a[i];
        const uint4 vb = b[i];
        __nv_bfloat16* ea = reinterpret_cast<__nv_bfloat16*>(&va);
        const __nv_bfloat16* eb = reinterpret_cast<const __nv_bfloat16*>(&vb);
        #pragma unroll
        for (int k = 0; k < 8; ++k) ea[k] = add_one(ea[k], eb[k]);
        out[i] = va;
    }
}

// bf16 tensor + fp32 tensor promotes to fp32: a + alpha * b with alpha = 1.
__global__ void add_bf16_f32(const __nv_bfloat16* a, const float* b, float* out, int64_t n) {
    int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float alpha = 1.0f;
    out[i] = static_cast<float>(a[i]) + alpha * b[i];
}
__global__ void silu_f32(const float* x, float* out, int64_t n) {
    int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    using opmath_t = float;
    const opmath_t x_acc = static_cast<opmath_t>(x[i]);
    out[i] = x_acc / (opmath_t(1) + ::exp(-x_acc));
}

// relu is clamp_min(self, 0) (TensorCompare.cu launch_clamp_scalar): NaN propagates,
// otherwise ::max(opmath v, 0) converted back.
__global__ void relu_bf16(const __nv_bfloat16* x, __nv_bfloat16* out, int64_t n) {
    int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    using opmath_t = float;
    const __nv_bfloat16 v = x[i];
    const opmath_t lim0_val = 0;
    if (isnan(static_cast<opmath_t>(v))) { out[i] = v; return; }
    out[i] = __float2bfloat16_rn(::max(static_cast<opmath_t>(v), lim0_val));
}

__global__ void bf16_to_f32(const __nv_bfloat16* x, float* out, int64_t n) {
    int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) out[i] = static_cast<float>(x[i]);
}

// sinusoidal_embedding_1d(dim, t): exponent = -(arange(half) * (1 / half)) (div by a CPU
// scalar multiplies by its reciprocal), pow(10000 (CPU scalar), exponent), outer product,
// then [cos | sin]. One position.
__global__ void sinusoidal(const float* position, float* out, int half) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= half) return;
    const float inv = 1.0f / float(half);
    const float exponent = -(float(i) * inv);
    const float freq = ::pow(10000.0f, exponent);
    const float arg = position[0] * freq;
    out[i] = ::cos(arg);
    out[half + i] = ::sin(arg);
}

// out[r, c] = x[r, c] + bias[channel(r)] for [channels, inner] blocks (conv bias, add_).
__global__ void add_channel_bias_bf16(__nv_bfloat16* x, const __nv_bfloat16* bias, int64_t inner, int64_t n) {
    int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float alpha = 1.0f;
    x[i] = __float2bfloat16_rn(static_cast<float>(x[i]) + alpha * static_cast<float>(bias[i / inner]));
}

// layer_norm_kernel.cu, vectorized path (vec_size 4, threads (32, 4)), float data.
struct WelfordDataLN { float mean, sigma2, count; };
__device__ WelfordDataLN cuWelfordOnlineSum(const float val, const WelfordDataLN& curr_sum) {
    float delta = val - curr_sum.mean;
    float new_count = curr_sum.count + 1.f;
    auto fn_rcp_mul = [](auto a, auto b) { return a * (1.0f / b); };
    float new_mean = curr_sum.mean + fn_rcp_mul(delta, new_count);
    return {new_mean, curr_sum.sigma2 + delta * (val - new_mean), new_count};
}
__device__ WelfordDataLN cuWelfordCombine(const WelfordDataLN dataB, const WelfordDataLN dataA) {
    float delta = dataB.mean - dataA.mean;
    float count = dataA.count + dataB.count;
    float mean, sigma2;
    if (count > 0.f) {
        auto fn_rcp = [](auto a) { return 1.0f / a; };
        auto coef = fn_rcp(count);
        auto nA = dataA.count * coef;
        auto nB = dataB.count * coef;
        mean = nA * dataA.mean + nB * dataB.mean;
        sigma2 = dataA.sigma2 + dataB.sigma2 + delta * delta * dataA.count * nB;
    } else {
        mean = 0.f;
        sigma2 = 0.f;
    }
    return {mean, sigma2, count};
}
struct alignas(16) float4v { float val[4]; };
__global__ void layer_norm_f32(const float* X, const float* gamma, const float* beta, float* Y, int N, float eps) {
    __shared__ float s_data[6];
    const int64_t row_offset = int64_t(blockIdx.x) * int64_t(N);
    const float4v* X_vec = reinterpret_cast<const float4v*>(X + row_offset);
    const int numx = blockDim.x * blockDim.y;
    const int thrx = threadIdx.x + threadIdx.y * blockDim.x;
    const int n_vec_to_read = N / 4;
    WelfordDataLN wd{0.f, 0.f, 0.f};
    for (int i = thrx; i < n_vec_to_read; i += numx) {
        float4v data = X_vec[i];
#pragma unroll
        for (int ii = 0; ii < 4; ii++) wd = cuWelfordOnlineSum(data.val[ii], wd);
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        WelfordDataLN wdB{__shfl_down_sync(0xffffffff, wd.mean, offset), __shfl_down_sync(0xffffffff, wd.sigma2, offset),
                          __shfl_down_sync(0xffffffff, wd.count, offset)};
        wd = cuWelfordCombine(wd, wdB);
    }
    float* meansigmabuf = s_data;
    float* countbuf = s_data + blockDim.y;
    for (int offset = blockDim.y / 2; offset > 0; offset /= 2) {
        if (threadIdx.x == 0 && threadIdx.y >= offset && threadIdx.y < 2 * offset) {
            const int wrt_y = threadIdx.y - offset;
            meansigmabuf[2 * wrt_y] = wd.mean;
            meansigmabuf[2 * wrt_y + 1] = wd.sigma2;
            countbuf[wrt_y] = wd.count;
        }
        __syncthreads();
        if (threadIdx.x == 0 && threadIdx.y < offset) {
            WelfordDataLN wdB{meansigmabuf[2 * threadIdx.y], meansigmabuf[2 * threadIdx.y + 1], countbuf[threadIdx.y]};
            wd = cuWelfordCombine(wd, wdB);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        meansigmabuf[0] = wd.mean;
        meansigmabuf[1] = wd.sigma2 / float(N);
    }
    __syncthreads();
    const float mean = meansigmabuf[0];
    const float rstd_val = ::rsqrt(meansigmabuf[1] + eps);
    const float4v* gamma_vec = reinterpret_cast<const float4v*>(gamma);
    const float4v* beta_vec = reinterpret_cast<const float4v*>(beta);
    float4v* Y_vec = reinterpret_cast<float4v*>(Y + row_offset);
    for (int i = thrx; i < n_vec_to_read; i += numx) {
        float4v data = X_vec[i];
        float4v out;
#pragma unroll
        for (int ii = 0; ii < 4; ii++)
            out.val[ii] = gamma_vec[i].val[ii] * (rstd_val * (data.val[ii] - mean)) + beta_vec[i].val[ii];
        Y_vec[i] = out;
    }
}
// x = cat([latent, y], dim=0).to(bf16): latent is bf16 (step 0) or fp32, y is bf16.
template <typename L>
__global__ void cat_latent_y(const L* latent, const __nv_bfloat16* y, __nv_bfloat16* out, int64_t latent_n, int64_t n) {
    int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (i < latent_n) {
        if constexpr (sizeof(L) == 4) out[i] = __float2bfloat16_rn(static_cast<float>(latent[i]));
        else out[i] = latent[i];
    } else {
        out[i] = y[i - latent_n];
    }
}

// [channels, tokens] -> [tokens, channels] (flatten(2).transpose(1, 2), materialized by cat).
// Tiles through shared memory so both the reads along tokens and the writes along channels are
// coalesced. Pure data movement.
constexpr int kTile = 32, kTileRows = 8;
__global__ void transpose_bf16(const __nv_bfloat16* x, __nv_bfloat16* out, int64_t channels, int64_t tokens,
                               int64_t token_tiles) {
    __shared__ __nv_bfloat16 tile[kTile][kTile + 1];
    const int64_t t0 = (int64_t(blockIdx.x) % token_tiles) * kTile, c0 = (int64_t(blockIdx.x) / token_tiles) * kTile;
    for (int r = threadIdx.y; r < kTile; r += kTileRows) {
        const int64_t c = c0 + r, t = t0 + threadIdx.x;
        if (c < channels && t < tokens) tile[r][threadIdx.x] = x[c * tokens + t];
    }
    __syncthreads();
    for (int r = threadIdx.y; r < kTile; r += kTileRows) {
        const int64_t t = t0 + r, c = c0 + threadIdx.x;
        if (t < tokens && c < channels) out[t * channels + c] = tile[threadIdx.x][r];
    }
}
// Even sizes: 64x64 tiles, two adjacent elements per 4-byte access on both sides.
constexpr int kWide = 64;
__global__ void transpose_bf16x2(const __nv_bfloat162* x, __nv_bfloat162* out, int64_t channels, int64_t tokens,
                                 int64_t token_tiles) {
    __shared__ __nv_bfloat16 tile[kWide][kWide + 2];
    const int64_t t0 = (int64_t(blockIdx.x) % token_tiles) * kWide, c0 = (int64_t(blockIdx.x) / token_tiles) * kWide;
    const int lane2 = 2 * threadIdx.x;
    for (int r = threadIdx.y; r < kWide; r += kTileRows) {
        const int64_t c = c0 + r, t = t0 + lane2;
        if (c < channels && t < tokens) *reinterpret_cast<__nv_bfloat162*>(&tile[r][lane2]) = x[(c * tokens + t) / 2];
    }
    __syncthreads();
    for (int r = threadIdx.y; r < kWide; r += kTileRows) {
        const int64_t t = t0 + r, c = c0 + lane2;
        if (t < tokens && c < channels) {
            __nv_bfloat162 v;
            v.x = tile[lane2][r];
            v.y = tile[lane2 + 1][r];
            out[(t * channels + c) / 2] = v;
        }
    }
}

// cat([fp32 rows, bf16 rows], dim=0) promotes to fp32.
__global__ void cat_f32_bf16(const float* a, const __nv_bfloat16* b, float* out, int64_t a_n, int64_t n) {
    int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = i < a_n ? a[i] : static_cast<float>(b[i - a_n]);
}
}  // namespace

extern "C" int wan_cat_latent_y(const void* latent, int latent_f32, const void* y, void* out, int64_t latent_n, int64_t y_n,
                                cudaStream_t stream) {
    const int64_t n = latent_n + y_n;
    if (latent_f32) cat_latent_y<float><<<blocks(n), kThreads, 0, stream>>>((const float*)latent, (const __nv_bfloat16*)y, (__nv_bfloat16*)out, latent_n, n);
    else cat_latent_y<__nv_bfloat16><<<blocks(n), kThreads, 0, stream>>>((const __nv_bfloat16*)latent, (const __nv_bfloat16*)y, (__nv_bfloat16*)out, latent_n, n);
    return cudaGetLastError();
}
extern "C" int wan_transpose_bf16(const void* x, void* out, int64_t channels, int64_t tokens, cudaStream_t stream) {
    if (channels <= 0 || tokens <= 0) return cudaSuccess;
    const bool pairs = channels % 2 == 0 && tokens % 2 == 0 && reinterpret_cast<uintptr_t>(x) % 4 == 0 &&
                       reinterpret_cast<uintptr_t>(out) % 4 == 0;
    const int tile = pairs ? kWide : kTile;
    const int64_t token_tiles = (tokens + tile - 1) / tile, tiles = token_tiles * ((channels + tile - 1) / tile);
    if (tiles > 2147483647) return cudaErrorInvalidValue;
    if (pairs)
        transpose_bf16x2<<<unsigned(tiles), dim3(32, kTileRows), 0, stream>>>((const __nv_bfloat162*)x, (__nv_bfloat162*)out,
                                                                              channels, tokens, token_tiles);
    else
        transpose_bf16<<<unsigned(tiles), dim3(kTile, kTileRows), 0, stream>>>((const __nv_bfloat16*)x, (__nv_bfloat16*)out,
                                                                              channels, tokens, token_tiles);
    return cudaGetLastError();
}
extern "C" int wan_cat_f32_bf16(const void* a, const void* b, void* out, int64_t a_n, int64_t b_n, cudaStream_t stream) {
    cat_f32_bf16<<<blocks(a_n + b_n), kThreads, 0, stream>>>((const float*)a, (const __nv_bfloat16*)b, (float*)out, a_n, a_n + b_n);
    return cudaGetLastError();
}

extern "C" int wan_silu_f32(const void* x, void* out, int64_t n, cudaStream_t stream) {
    silu_f32<<<blocks(n), kThreads, 0, stream>>>((const float*)x, (float*)out, n);
    return cudaGetLastError();
}
extern "C" int wan_relu_bf16(const void* x, void* out, int64_t n, cudaStream_t stream) {
    relu_bf16<<<blocks(n), kThreads, 0, stream>>>((const __nv_bfloat16*)x, (__nv_bfloat16*)out, n);
    return cudaGetLastError();
}
extern "C" int wan_bf16_to_f32(const void* x, void* out, int64_t n, cudaStream_t stream) {
    bf16_to_f32<<<blocks(n), kThreads, 0, stream>>>((const __nv_bfloat16*)x, (float*)out, n);
    return cudaGetLastError();
}
extern "C" int wan_sinusoidal(const void* position, void* out, int dim, cudaStream_t stream) {
    if (dim <= 0 || dim % 2) return cudaErrorInvalidValue;
    sinusoidal<<<blocks(dim / 2), kThreads, 0, stream>>>((const float*)position, (float*)out, dim / 2);
    return cudaGetLastError();
}
extern "C" int wan_add_channel_bias_bf16(void* x, const void* bias, int64_t channels, int64_t inner, cudaStream_t stream) {
    const int64_t n = channels * inner;
    add_channel_bias_bf16<<<blocks(n), kThreads, 0, stream>>>((__nv_bfloat16*)x, (const __nv_bfloat16*)bias, inner, n);
    return cudaGetLastError();
}
extern "C" int wan_layer_norm_f32(const void* x, const void* gamma, const void* beta, void* out, int64_t rows, int cols,
                                  float eps, cudaStream_t stream) {
    if (rows < 1 || cols < 4 || cols % 4) return cudaErrorInvalidValue;
    for (const void* p : {x, gamma, beta, static_cast<const void*>(out)})
        if (reinterpret_cast<uintptr_t>(p) % 16) return cudaErrorMisalignedAddress;
    layer_norm_f32<<<unsigned(rows), dim3(32, 4), 0, stream>>>((const float*)x, (const float*)gamma, (const float*)beta,
                                                             (float*)out, cols, eps);
    return cudaGetLastError();
}

extern "C" int wan_gelu_tanh_bf16(const void* x, void* out, int64_t n, cudaStream_t stream) {
    const bool aligned = reinterpret_cast<uintptr_t>(x) % 16 == 0 && reinterpret_cast<uintptr_t>(out) % 16 == 0;
    const int64_t n8 = aligned ? n / 8 : 0;
    int sms = 0, device = 0;
    if (n8 >= (int64_t(1) << 20) && cudaGetDevice(&device) == cudaSuccess &&
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device) == cudaSuccess && gelu_lut_ready(stream)) {
        gelu_tanh_bf16_lut<<<unsigned(sms), kLutThreads, 65536 * 2, stream>>>((const uint4*)x, (uint4*)out, n8);
    } else if (n8 > 0) {
        const int64_t want = (n8 + kThreads - 1) / kThreads;
        gelu_tanh_bf16_vec8<<<unsigned(want < 65536 * 8 ? want : 65536 * 8), kThreads, 0, stream>>>(
            (const uint4*)x, (uint4*)out, n8);
    }
    const int64_t done = n8 * 8;
    if (done < n)
        gelu_tanh_bf16<<<blocks(n - done), kThreads, 0, stream>>>((const __nv_bfloat16*)x + done, (__nv_bfloat16*)out + done,
                                                                   n - done);
    return cudaGetLastError();
}

extern "C" int wan_add_bf16(const void* a, const void* b, void* out, int64_t n, cudaStream_t stream) {
    const bool aligned = (reinterpret_cast<uintptr_t>(a) | reinterpret_cast<uintptr_t>(b) | reinterpret_cast<uintptr_t>(out)) % 16 == 0;
    const int64_t n8 = aligned ? n / 8 : 0;
    if (n8 > 0) {
        const int64_t want = (n8 + kThreads - 1) / kThreads;
        add_bf16_vec8<<<unsigned(want < 65536 * 8 ? want : 65536 * 8), kThreads, 0, stream>>>((const uint4*)a, (const uint4*)b,
                                                                                            (uint4*)out, n8);
    }
    const int64_t done = n8 * 8;
    if (done < n)
        add_bf16<<<blocks(n - done), kThreads, 0, stream>>>((const __nv_bfloat16*)a + done, (const __nv_bfloat16*)b + done,
                                                            (__nv_bfloat16*)out + done, n - done);
    return cudaGetLastError();
}

extern "C" int wan_add_bf16_f32(const void* a, const void* b, void* out, int64_t n, cudaStream_t stream) {
    add_bf16_f32<<<blocks(n), kThreads, 0, stream>>>((const __nv_bfloat16*)a, (const float*)b, (float*)out, n);
    return cudaGetLastError();
}
