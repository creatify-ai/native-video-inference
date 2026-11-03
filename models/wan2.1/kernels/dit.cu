// Official Wan 2.1 DiT (wan/modules/model.py at 9737cba) elementwise numerics, as the eager PyTorch
// 2.13 kernels compute them under bf16 autocast with the bf16-converted parameters. Each kernel
// fuses a chain of eager ops but keeps every op's rounding: fp32 ops with __f*_rn (no
// contraction across ops), casts at the points where the Python code casts.
//   aten/src/ATen/native/cuda/layer_norm_kernel.cu (vectorized_layer_norm_kernel, vec 4, 32x4 threads)
//   aten/src/ATen/native/cuda/PowKernel.cu (pow(x, 2) = x * x), UnaryOpsKernel.cu (rsqrt)
//   c10/util/complex.h (complex<double> operator*)
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <initializer_list>

using B = __nv_bfloat16;
namespace {
constexpr int kThreads = 256;
inline unsigned blocks(int64_t n) { return unsigned((n + kThreads - 1) / kThreads); }
__device__ __forceinline__ float f(B v) { return __bfloat162float(v); }
__device__ __forceinline__ B b(float v) { return __float2bfloat16_rn(v); }
__device__ __forceinline__ float round_bf16(float v) { return f(b(v)); }

// cat([latent, y], dim=0) [C, T, H, W] fp32 each, then the autocast cast to bf16 (patch embedding input).
__global__ void cat_cast_bf16(const float* a, const float* y, B* out, int64_t a_n, int64_t n) {
    int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) out[i] = b(i < a_n ? a[i] : y[i - a_n]);
}

// sinusoidal_embedding_1d(256, t): float64 throughout, then .float().
//   exponent = -(arange(128).double() / 128)   (CUDA div by a CPU scalar: x * (1.0 / 128), exact)
//   sinusoid = position * pow(10000, exponent); x = cat([cos, sin])
__global__ void sinusoidal_f64(const int64_t* t, float* out, int half) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= half) return;
    const double position = static_cast<double>(t[0]);
    const double exponent = -(static_cast<double>(i) * (1.0 / static_cast<double>(half)));
    const double arg = position * ::pow(10000.0, exponent);
    out[i] = static_cast<float>(::cos(arg));
    out[half + i] = static_cast<float>(::sin(arg));
}

// ---- LayerNorm ----------------------------------------------------------------------------------
// torch's vectorized_layer_norm_kernel on fp32 rows (the Python code calls x.float() first, or
// autocast casts the input; the conversion is exact, so it is fused into the load).
struct Welford { float mean, sigma2, count; };
__device__ __forceinline__ Welford online_sum(float val, const Welford& s) {
    float delta = val - s.mean;
    float new_count = s.count + 1.f;
    float new_mean = s.mean + delta * (1.0f / new_count);
    return {new_mean, s.sigma2 + delta * (val - new_mean), new_count};
}
__device__ __forceinline__ Welford combine(const Welford b_, const Welford a) {
    float delta = b_.mean - a.mean;
    float count = a.count + b_.count;
    if (count > 0.f) {
        float coef = 1.0f / count;
        float na = a.count * coef, nb = b_.count * coef;
        return {na * a.mean + nb * b_.mean, a.sigma2 + b_.sigma2 + delta * delta * a.count * nb, count};
    }
    return {0.f, 0.f, 0.f};
}
template <typename T> __device__ __forceinline__ float load_f(const T* p, int64_t i);
template <> __device__ __forceinline__ float load_f<float>(const float* p, int64_t i) { return p[i]; }
template <> __device__ __forceinline__ float load_f<B>(const B* p, int64_t i) { return f(p[i]); }
template <> __device__ __forceinline__ float load_f<__half>(const __half* p, int64_t i) { return __half2float(p[i]); }
// 4 consecutive elements in one access (V kernels: 16-byte-aligned fp32, 8-byte-aligned 16-bit rows).
template <typename T> __device__ __forceinline__ float4 load4(const T* p);
template <> __device__ __forceinline__ float4 load4<float>(const float* p) { return *reinterpret_cast<const float4*>(p); }
template <> __device__ __forceinline__ float4 load4<B>(const B* p) {
    const uint2 u = *reinterpret_cast<const uint2*>(p);
    const B* e = reinterpret_cast<const B*>(&u);
    return make_float4(f(e[0]), f(e[1]), f(e[2]), f(e[3]));
}
template <> __device__ __forceinline__ float4 load4<__half>(const __half* p) {
    const uint2 u = *reinterpret_cast<const uint2*>(p);
    const __half* e = reinterpret_cast<const __half*>(&u);
    return make_float4(__half2float(e[0]), __half2float(e[1]), __half2float(e[2]), __half2float(e[3]));
}

// Row statistics exactly as compute_stats<float> with blockDim (32, 4) and vec_size 4: each thread's
// Welford state over its 4-element vectors thrx, thrx + 128, ... (ln_accumulate), then the warp and
// block combine (ln_reduce). V loads each vector in one access (same values, same order).
// ITERS > 0: every thread of the 128 accumulates exactly ITERS vectors (n == 512 * ITERS); the loop
// is unrolled, so its loads issue together and each 1/count is a compile-time constant (the same
// correctly rounded value). The vectors and their order are the runtime loop's.
template <typename T, bool V, int ITERS = 0>
__device__ __forceinline__ Welford ln_accumulate(const T* row, int n, float4* held = nullptr) {
    const int numx = blockDim.x * blockDim.y, thrx = threadIdx.x + threadIdx.y * blockDim.x, nvec = n / 4;
    Welford wd{0.f, 0.f, 0.f};
    auto step = [&](int i) {
        if constexpr (V) {
            const float4 v = load4(row + int64_t(i) * 4);
            wd = online_sum(v.x, wd);
            wd = online_sum(v.y, wd);
            wd = online_sum(v.z, wd);
            wd = online_sum(v.w, wd);
        } else {
            #pragma unroll
            for (int ii = 0; ii < 4; ++ii) wd = online_sum(load_f(row, int64_t(i) * 4 + ii), wd);
        }
    };
    if constexpr (ITERS > 0 && V) {
        // The fixed form keeps its vectors for ln_apply (`held`), which then reads no memory.
        #pragma unroll
        for (int it = 0; it < ITERS; ++it) {
            const float4 v = load4(row + int64_t(thrx + it * 128) * 4);
            held[it] = v;
            wd = online_sum(v.x, wd);
            wd = online_sum(v.y, wd);
            wd = online_sum(v.z, wd);
            wd = online_sum(v.w, wd);
        }
    } else if constexpr (ITERS > 0) {
        #pragma unroll
        for (int it = 0; it < ITERS; ++it) step(thrx + it * 128);
    } else {
        for (int i = thrx; i < nvec; i += numx) step(i);
    }
    return wd;
}
__device__ void ln_reduce(Welford wd, int n, float* buf, float& mean, float& var) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        Welford o{__shfl_down_sync(0xffffffff, wd.mean, offset), __shfl_down_sync(0xffffffff, wd.sigma2, offset),
                  __shfl_down_sync(0xffffffff, wd.count, offset)};
        wd = combine(wd, o);
    }
    float* ms = buf;
    float* cnt = buf + blockDim.y;
    for (int offset = blockDim.y / 2; offset > 0; offset /= 2) {
        if (threadIdx.x == 0 && threadIdx.y >= offset && threadIdx.y < 2 * offset) {
            const int w = threadIdx.y - offset;
            ms[2 * w] = wd.mean;
            ms[2 * w + 1] = wd.sigma2;
            cnt[w] = wd.count;
        }
        __syncthreads();
        if (threadIdx.x == 0 && threadIdx.y < offset) wd = combine(wd, Welford{ms[2 * threadIdx.y], ms[2 * threadIdx.y + 1], cnt[threadIdx.y]});
        __syncthreads();
    }
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        ms[0] = wd.mean;
        ms[1] = wd.sigma2 / float(n);
    }
    __syncthreads();
    mean = ms[0];
    var = ms[1];
}

// Modes (what the Python code does with the LayerNorm output):
//   kModulate:  (norm(x).type_as(x)).float() * (1 + e1) + e0  -> bf16 (the next Linear's autocast cast);
//               `round_out` rounds the LN output to bf16 first (block 0, whose x is bf16).
//   kAffineBf16: affine LN (fp32) -> bf16 (the next Linear's autocast cast).
//   kAffineF32:  affine LN -> fp32.
//   kModulateF32: norm(x) * (1 + e1) + e0 -> fp32 (Head, inside its fp32 autocast region).
enum Mode { kModulate = 0, kAffineBf16 = 1, kAffineF32 = 2, kModulateF32 = 3 };
// The normalized row: thread thrx writes elements 4 * (thrx + k * 128) .. + 3, the vectors whose
// statistics it accumulated. V: 4-element loads and stores (row and out aligned to 4 elements).
template <typename T, int M, bool V, int ITERS = 0>
__device__ __forceinline__ void ln_apply(const T* row, int64_t o0, int n, float mean, float rstd, const float* g, const float* be,
                         const float* e0, const float* e1, void* out, int round_out, const float4* held = nullptr) {
    const int numx = blockDim.x * blockDim.y, thrx = threadIdx.x + threadIdx.y * blockDim.x;
    constexpr bool kBf16Out = M == kModulate || M == kAffineBf16;
    #pragma unroll
    for (int i = thrx * 4, it = 0; ITERS > 0 ? it < ITERS : i < n; i += numx * 4, ++it) {
        float data[4], y4[4];
        if constexpr (V) {
            const float4 v = ITERS > 0 && held ? held[it] : load4(row + i);
            data[0] = v.x, data[1] = v.y, data[2] = v.z, data[3] = v.w;
        } else {
            #pragma unroll
            for (int ii = 0; ii < 4; ++ii) data[ii] = load_f(row, i + ii);
        }
        #pragma unroll
        for (int ii = 0; ii < 4; ++ii) {
            const int c = i + ii;
            if constexpr (M == kAffineBf16 || M == kAffineF32) {
                y4[ii] = g[c] * (rstd * (data[ii] - mean)) + be[c];
            } else {
                float y = rstd * (data[ii] - mean);
                if (round_out) y = round_bf16(y);
                y4[ii] = __fadd_rn(__fmul_rn(y, __fadd_rn(1.0f, e1[c])), e0[c]);
            }
        }
        const int64_t o = o0 + i;
        if constexpr (V && kBf16Out) {
            uint2 u;
            B* e = reinterpret_cast<B*>(&u);
            #pragma unroll
            for (int ii = 0; ii < 4; ++ii) e[ii] = b(y4[ii]);
            *reinterpret_cast<uint2*>(static_cast<B*>(out) + o) = u;
        } else if constexpr (V) {
            *reinterpret_cast<float4*>(static_cast<float*>(out) + o) = make_float4(y4[0], y4[1], y4[2], y4[3]);
        } else {
            #pragma unroll
            for (int ii = 0; ii < 4; ++ii) {
                if constexpr (kBf16Out) static_cast<B*>(out)[o + ii] = b(y4[ii]);
                else static_cast<float*>(out)[o + ii] = y4[ii];
            }
        }
    }
}
template <typename T, int M, bool V, int ITERS = 0>
__global__ void layer_norm(const T* x, const float* g, const float* be, const float* e0, const float* e1, void* out, int n,
                           float eps, int round_out) {
    __shared__ float buf[6];
    const T* row = x + int64_t(blockIdx.x) * n;
    float mean, var;
    float4 held[ITERS > 0 ? ITERS : 1];
    ln_reduce(ln_accumulate<T, V, ITERS>(row, n, held), n, buf, mean, var);
    ln_apply<T, M, V, ITERS>(row, int64_t(blockIdx.x) * n, n, mean, ::rsqrtf(var + eps), g, be, e0, e1, out, round_out,
                             ITERS > 0 && V ? held : nullptr);
}
// x = x + y * eg (Gate, residual_gate's expression) or x + y (residual_add's), in place, then
// layer_norm<float, M, true> of the new row. Each thread writes its own vectors during the Welford
// pass and reads only those back, so the row needs no barrier between the passes. ITERS > 0 fixes
// the vectors per thread (n = 4 * blockDim * ITERS) at compile time: the same loop, unrolled, whose
// Welford counts (and so their reciprocals) are constants.
template <int M, bool Gate, int ITERS = 0>
__global__ void residual_layer_norm(float* x, const B* y, const float* eg, const float* g, const float* be, const float* e0,
                                    const float* e1, void* out, int n, float eps) {
    __shared__ float buf[6];
    const int numx = blockDim.x * blockDim.y, thrx = threadIdx.x + threadIdx.y * blockDim.x, nvec = ITERS > 0 ? ITERS * numx : n / 4;
    float* row = x + int64_t(blockIdx.x) * n;
    const B* yr = y + int64_t(blockIdx.x) * n;
    Welford wd{0.f, 0.f, 0.f};
    #pragma unroll
    for (int i = thrx; i < nvec; i += numx) {
        const float4 xv = load4(row + int64_t(i) * 4), yv = load4(yr + int64_t(i) * 4);
        float4 r;
        if constexpr (Gate) {
            const float4 ev = load4(eg + int64_t(i) * 4);
            r = make_float4(__fadd_rn(xv.x, __fmul_rn(yv.x, ev.x)), __fadd_rn(xv.y, __fmul_rn(yv.y, ev.y)),
                            __fadd_rn(xv.z, __fmul_rn(yv.z, ev.z)), __fadd_rn(xv.w, __fmul_rn(yv.w, ev.w)));
        } else {
            r = make_float4(__fadd_rn(xv.x, yv.x), __fadd_rn(xv.y, yv.y), __fadd_rn(xv.z, yv.z), __fadd_rn(xv.w, yv.w));
        }
        *reinterpret_cast<float4*>(row + int64_t(i) * 4) = r;
        wd = online_sum(r.x, wd);
        wd = online_sum(r.y, wd);
        wd = online_sum(r.z, wd);
        wd = online_sum(r.w, wd);
    }
    float mean, var;
    ln_reduce(wd, n, buf, mean, var);
    ln_apply<float, M, true>(row, int64_t(blockIdx.x) * n, n, mean, ::rsqrtf(var + eps), g, be, e0, e1, out, 0);
}

// ---- WanRMSNorm ---------------------------------------------------------------------------------
// mean(x.float().pow(2), -1) is torch's Reduce port with the square at the load (post.cu).
// (x * rsqrt(mean + eps)).type_as(x) * weight: fp32 mul, bf16 round, bf16 x bf16 -> bf16.
__global__ void rms_apply(const B* x, const float* mean, const B* w, B* out, int64_t cols, float eps, int64_t n) {
    int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float r = ::rsqrtf(__fadd_rn(mean[i / cols], eps));
    const B t = b(__fmul_rn(f(x[i]), r));
    out[i] = b(__fmul_rn(f(t), f(w[i % cols])));
}

// ---- RoPE ---------------------------------------------------------------------------------------
// rope_apply: x (bf16 [L, heads, 128]) to float64 pairs, times the complex128 factors of the token's
// (f, h, w) cell (pairs 0..21 by frame, 22..42 by row, 43..63 by column), back to real, .float();
// flash_attention then casts to bf16. `out_f32` (optional) keeps the fp32 tensor for comparison.
__global__ void rope(const B* x, const double* freqs, B* out, float* out_f32, int heads, int hh, int ww, int64_t tokens) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;  // one complex pair
    const int64_t pairs = int64_t(heads) * 64;
    if (i >= tokens * pairs) return;
    const int64_t l = i / pairs;
    const int j = int(i % 64);
    const int64_t fi = l / (int64_t(hh) * ww), hi = (l / ww) % hh, wi = l % ww;
    const int64_t row = j < 22 ? fi : (j < 43 ? hi : wi);
    const double c = freqs[(row * 64 + j) * 2], d = freqs[(row * 64 + j) * 2 + 1];
    const double a = static_cast<double>(f(x[2 * i])), bb = static_cast<double>(f(x[2 * i + 1]));
    const double re = a * c - bb * d;
    const double im = a * d + bb * c;
    const float rf = static_cast<float>(re), imf = static_cast<float>(im);
    out[2 * i] = b(rf);
    out[2 * i + 1] = b(imf);
    if (out_f32) { out_f32[2 * i] = rf; out_f32[2 * i + 1] = imf; }
}

// ---- residual stream ----------------------------------------------------------------------------
// x + y * e (fp32 autocast region): bf16 y promotes; mul then add, each rounded. X is bf16 (block 0)
// or fp32 (may alias out).
template <typename X>
__global__ void residual_gate(const X* x, const B* y, const float* e, float* out, int64_t cols, int64_t n) {
    int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __fadd_rn(load_f(x, i), __fmul_rn(f(y[i]), e[i % cols]));
}
// x + cross_attn(...) (fp32 + bf16 -> fp32, alpha 1).
__global__ void residual_add(const float* x, const B* y, float* out, int64_t n) {
    int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __fadd_rn(x[i], f(y[i]));
}
// modulation (bf16 param) + e (fp32) -> fp32.
__global__ void add_bf16_f32(const B* a, const float* e, float* out, int64_t n, int64_t e_period) {
    int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __fadd_rn(f(a[i]), e[i % e_period]);
}
template <typename I, typename O>
__global__ void cast(const I* x, O* out, int64_t n) {
    int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = load_f(x, i);
    if constexpr (sizeof(O) == 4) out[i] = v; else out[i] = b(v);
}
// unpatchify: u[:L].view(F, H, W, 1, 2, 2, C) -> einsum('fhwpqrc->cfphqwr') -> [C, F, 2H, 2W].
// torch.einsum runs under the outer bf16 autocast, so its output is bf16; WanModel.forward then
// returns u.float().
__global__ void unpatchify(const float* u, float* out, int fr, int hh, int ww, int c, int64_t n) {
    int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;  // output index
    if (i >= n) return;
    const int64_t W2 = 2 * ww, H2 = 2 * hh;
    const int64_t x2 = i % W2, y2 = (i / W2) % H2, t = (i / (W2 * H2)) % fr, ch = i / (W2 * H2 * fr);
    const int64_t src = ((((t * hh + y2 / 2) * ww + x2 / 2) * 2 + y2 % 2) * 2 + x2 % 2) * c + ch;
    out[i] = round_bf16(u[src]);
}

// ---- 8-wide forms -------------------------------------------------------------------------------
// The RMSNorm, RoPE and residual kernels above, 8 consecutive elements per thread through 16-byte
// accesses, for row-major [rows, cols] tensors with cols % 8 == 0 and 16-byte-aligned pointers. The
// grid's x covers a row's 8-element vectors and its y strides over rows, so per-column operands load
// once per thread and no element index is divided. Every per-element expression is the scalar
// kernel's; results do not depend on the vectorization.
constexpr int kVecThreads = 128;
inline dim3 row_grid(int64_t rows, int64_t cols) {
    return dim3(unsigned((cols / 8 + kVecThreads - 1) / kVecThreads), unsigned(rows < 8192 ? rows : 8192));
}
inline bool vec8(int64_t cols, std::initializer_list<const void*> ptrs) {
    for (const void* q : ptrs)
        if (reinterpret_cast<uintptr_t>(q) % 16) return false;
    return cols % 8 == 0;
}
struct F8 { float v[8]; };
__device__ __forceinline__ F8 load8(const float* q) {
    const float4 a = reinterpret_cast<const float4*>(q)[0], c = reinterpret_cast<const float4*>(q)[1];
    return {{a.x, a.y, a.z, a.w, c.x, c.y, c.z, c.w}};
}
__device__ __forceinline__ F8 load8(const B* q) {
    const uint4 u = *reinterpret_cast<const uint4*>(q);
    const B* e = reinterpret_cast<const B*>(&u);
    F8 r;
    #pragma unroll
    for (int k = 0; k < 8; ++k) r.v[k] = f(e[k]);
    return r;
}
__device__ __forceinline__ void store8(float* q, const F8& x) {
    reinterpret_cast<float4*>(q)[0] = make_float4(x.v[0], x.v[1], x.v[2], x.v[3]);
    reinterpret_cast<float4*>(q)[1] = make_float4(x.v[4], x.v[5], x.v[6], x.v[7]);
}
// Rounds each element to bf16.
__device__ __forceinline__ void store8(B* q, const F8& x) {
    uint4 u;
    B* e = reinterpret_cast<B*>(&u);
    #pragma unroll
    for (int k = 0; k < 8; ++k) e[k] = b(x.v[k]);
    *reinterpret_cast<uint4*>(q) = u;
}

__global__ void rms_apply8(const B* x, const float* mean, const B* w, B* out, int64_t cols, float eps, int64_t rows) {
    const int64_t c = (int64_t(blockIdx.x) * blockDim.x + threadIdx.x) * 8;
    if (c >= cols) return;
    const F8 wv = load8(w + c);
    for (int64_t r = blockIdx.y; r < rows; r += gridDim.y) {
        const float rs = ::rsqrtf(__fadd_rn(mean[r], eps));
        F8 v = load8(x + r * cols + c);
        #pragma unroll
        for (int k = 0; k < 8; ++k) v.v[k] = __fmul_rn(round_bf16(__fmul_rn(v.v[k], rs)), wv.v[k]);
        store8(out + r * cols + c, v);
    }
}

// rope on 4 consecutive complex pairs j0..j0+3 of one head (they may straddle a section edge) of
// token (fi, hi, wi): rope's per-pair expression.
__device__ __forceinline__ void rope4(const B* e, B* eo, int j0, int fi, int hi, int wi, const double* freqs) {
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        const int j = j0 + k;
        const int row = j < 22 ? fi : (j < 43 ? hi : wi);
        const double c = freqs[(row * 64 + j) * 2], d = freqs[(row * 64 + j) * 2 + 1];
        const double a = static_cast<double>(f(e[2 * k])), bb = static_cast<double>(f(e[2 * k + 1]));
        const double re = a * c - bb * d;
        const double im = a * d + bb * c;
        const float rf = static_cast<float>(re), imf = static_cast<float>(im);
        eo[2 * k] = b(rf);
        eo[2 * k + 1] = b(imf);
    }
}
// Token l's (frame, row, column) cell; 32-bit (a request has far fewer than 2^31 tokens).
__device__ __forceinline__ void cell(int64_t l, int hh, int ww, int& fi, int& hi, int& wi) {
    const int li = int(l), hw = hh * ww;
    fi = li / hw;
    const int r = li - fi * hw;
    hi = r / ww;
    wi = r - hi * ww;
}

// cols = heads * 128.
__global__ void rope8(const B* x, const double* freqs, B* out, int64_t cols, int hh, int ww, int64_t tokens) {
    const int64_t col = (int64_t(blockIdx.x) * blockDim.x + threadIdx.x) * 8;
    if (col >= cols) return;
    const int j0 = int(col % 128) / 2;
    for (int64_t l = blockIdx.y; l < tokens; l += gridDim.y) {
        int fi, hi, wi;
        cell(l, hh, ww, fi, hi, wi);
        const uint4 u = *reinterpret_cast<const uint4*>(x + l * cols + col);
        uint4 o;
        rope4(reinterpret_cast<const B*>(&u), reinterpret_cast<B*>(&o), j0, fi, hi, wi, freqs);
        *reinterpret_cast<uint4*>(out + l * cols + col) = o;
    }
}

// rms_apply8 then rope8 on the same 8 elements: the bf16 normalized values rms_apply8 would write
// go straight into the rotation.
__global__ void rms_rope8(const B* x, const float* mean, const B* w, const double* freqs, B* out, int64_t cols, float eps,
                          int hh, int ww, int64_t tokens) {
    const int64_t col = (int64_t(blockIdx.x) * blockDim.x + threadIdx.x) * 8;
    if (col >= cols) return;
    const F8 wv = load8(w + col);
    const int j0 = int(col % 128) / 2;
    for (int64_t l = blockIdx.y; l < tokens; l += gridDim.y) {
        int fi, hi, wi;
        cell(l, hh, ww, fi, hi, wi);
        const float rs = ::rsqrtf(__fadd_rn(mean[l], eps));
        const F8 v = load8(x + l * cols + col);
        B qn[8];
        #pragma unroll
        for (int k = 0; k < 8; ++k) qn[k] = b(__fmul_rn(round_bf16(__fmul_rn(v.v[k], rs)), wv.v[k]));
        uint4 o;
        rope4(qn, reinterpret_cast<B*>(&o), j0, fi, hi, wi, freqs);
        *reinterpret_cast<uint4*>(out + l * cols + col) = o;
    }
}
// A bf16 as a double (exact): a normal value by placing its sign, rebiased exponent and 7 mantissa
// bits; zero, subnormals, inf and NaN through the conversion.
__device__ __forceinline__ double bf16_to_double(B v) {
    const unsigned bits = __bfloat16_as_ushort(v), e = (bits >> 7) & 0xFF;
    if (e - 1u < 254u) return __hiloint2double(int(((bits & 0x8000u) << 16) | ((e + 896u) << 20) | ((bits & 0x7Fu) << 13)), 0);
    return static_cast<double>(f(v));
}
// rms_rope8 with the token's factors shared across heads: a block of 128 threads takes one token
// row of 8 * H heads; thread t takes elements 8 (t % 16) .. + 7 of heads t / 16 + 8 k (k < H), so its
// four complex factors (16-byte loads) serve H heads and its H weight vectors stay in registers.
// Every element is rms_rope8's expression on the same operands.
template <int H>
__global__ void rms_rope8_heads(const B* x, const float* mean, const B* w, const double* freqs, B* out, int64_t cols,
                                float eps, int hh, int ww, int64_t tokens) {
    const int g = threadIdx.x % 16, hb = threadIdx.x / 16, j0 = 4 * g;
    uint4 wv[H];
    #pragma unroll
    for (int k = 0; k < H; ++k) wv[k] = *reinterpret_cast<const uint4*>(w + (hb + 8 * k) * 128 + g * 8);
    for (int64_t l = blockIdx.x; l < tokens; l += gridDim.x) {
        int fi, hi, wi;
        cell(l, hh, ww, fi, hi, wi);
        double2 cd[4];
        #pragma unroll
        for (int q = 0; q < 4; ++q) {
            const int j = j0 + q;
            const int row = j < 22 ? fi : (j < 43 ? hi : wi);
            cd[q] = *reinterpret_cast<const double2*>(freqs + (row * 64 + j) * 2);
        }
        const float rs = ::rsqrtf(__fadd_rn(mean[l], eps));
        uint4 u[H];
        #pragma unroll
        for (int k = 0; k < H; ++k) u[k] = *reinterpret_cast<const uint4*>(x + l * cols + (hb + 8 * k) * 128 + g * 8);
        #pragma unroll
        for (int k = 0; k < H; ++k) {
            const B* e = reinterpret_cast<const B*>(&u[k]);
            const B* we = reinterpret_cast<const B*>(&wv[k]);
            B qn[8];
            #pragma unroll
            for (int m = 0; m < 8; ++m) qn[m] = b(__fmul_rn(round_bf16(__fmul_rn(f(e[m]), rs)), f(we[m])));
            uint4 o;
            B* eo = reinterpret_cast<B*>(&o);
            #pragma unroll
            for (int q = 0; q < 4; ++q) {
                const double c = cd[q].x, d = cd[q].y;
                const double a = bf16_to_double(qn[2 * q]), bb = bf16_to_double(qn[2 * q + 1]);
                const double re = a * c - bb * d;
                const double im = a * d + bb * c;
                eo[2 * q] = b(static_cast<float>(re));
                eo[2 * q + 1] = b(static_cast<float>(im));
            }
            *reinterpret_cast<uint4*>(out + l * cols + (hb + 8 * k) * 128 + g * 8) = o;
        }
    }
}

// Two floats rounded to bf16 in one packed conversion (round to nearest even, as b()); lo_f / hi_f
// read the halves back as floats.
__device__ __forceinline__ unsigned pack_bf16x2(float lo, float hi) {
    const __nv_bfloat162 p = __floats2bfloat162_rn(lo, hi);
    return *reinterpret_cast<const unsigned*>(&p);
}
__device__ __forceinline__ float lo_f(unsigned u) { return __uint_as_float(u << 16); }
__device__ __forceinline__ float hi_f(unsigned u) { return __uint_as_float(u & 0xFFFF0000u); }
// WanRMSNorm (then rope when Rope) of bf16 rows of 5120, a warp per row. Lane l holds columns
// 128 j + 4 l .. + 3 (j < 40): square_mean_rows' lanes and order, so the mean of squares is that
// kernel's sum of the same values in registers. The normalized values are rms_apply8's expressions
// and the rotation rms_rope8's (lane l's two pairs per head take the factors of pairs 2 l, 2 l + 1).
template <bool Rope>
__global__ void __launch_bounds__(128) rms_rows(const B* x, const B* w, const double* freqs, B* out, int64_t rows, float eps,
                                                float factor, int hh, int ww) {
    const int lane = threadIdx.x;
    const int64_t l = int64_t(blockIdx.x) * blockDim.y + threadIdx.y;
    if (l >= rows) return;
    const B* xr = x + l * 5120 + 4 * lane;
    uint2 v[40];
    #pragma unroll
    for (int j = 0; j < 40; ++j) v[j] = *reinterpret_cast<const uint2*>(xr + 128 * j);
    double2 cd[2];
    if (Rope) {
        int fi, hi, wi;
        cell(l, hh, ww, fi, hi, wi);
        #pragma unroll
        for (int q = 0; q < 2; ++q) {
            const int j = 2 * lane + q;
            const int row = j < 22 ? fi : (j < 43 ? hi : wi);
            cd[q] = *reinterpret_cast<const double2*>(freqs + (row * 64 + j) * 2);
        }
    }
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    #pragma unroll
    for (int j = 0; j < 40; ++j) {
        const float e[4] = {lo_f(v[j].x), hi_f(v[j].x), lo_f(v[j].y), hi_f(v[j].y)};
        #pragma unroll
        for (int i = 0; i < 4; ++i) acc[i] = __fadd_rn(acc[i], __fmul_rn(e[i], e[i]));
    }
    float s = acc[0];
    s = __fadd_rn(s, acc[1]);
    s = __fadd_rn(s, acc[2]);
    s = __fadd_rn(s, acc[3]);
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) s = __fadd_rn(s, __shfl_down_sync(0xffffffff, s, offset));
    const float rs = ::rsqrtf(__fadd_rn(__fmul_rn(__shfl_sync(0xffffffff, s, 0), factor), eps));
    B* orow = out + l * 5120 + 4 * lane;
    const B* wl = w + 4 * lane;
    #pragma unroll 8
    for (int j = 0; j < 40; ++j) {
        const uint2 wu = __ldg(reinterpret_cast<const uint2*>(wl + 128 * j));
        unsigned o[2];
        #pragma unroll
        for (int q = 0; q < 2; ++q) {
            const unsigned xv = q ? v[j].y : v[j].x, wv = q ? wu.y : wu.x;
            const unsigned t = pack_bf16x2(__fmul_rn(lo_f(xv), rs), __fmul_rn(hi_f(xv), rs));
            const unsigned n = pack_bf16x2(__fmul_rn(lo_f(t), lo_f(wv)), __fmul_rn(hi_f(t), hi_f(wv)));
            if (!Rope) { o[q] = n; continue; }
            const double a = static_cast<double>(lo_f(n)), bb = static_cast<double>(hi_f(n));
            const double c = cd[q].x, d = cd[q].y;
            const double re = a * c - bb * d;
            const double im = a * d + bb * c;
            o[q] = pack_bf16x2(static_cast<float>(re), static_cast<float>(im));
        }
        *reinterpret_cast<uint2*>(orow + 128 * j) = make_uint2(o[0], o[1]);
    }
}

template <typename X>
__global__ void residual_gate8(const X* x, const B* y, const float* e, float* out, int64_t cols, int64_t rows) {
    const int64_t c = (int64_t(blockIdx.x) * blockDim.x + threadIdx.x) * 8;
    if (c >= cols) return;
    const F8 ev = load8(e + c);
    for (int64_t r = blockIdx.y; r < rows; r += gridDim.y) {
        const int64_t o = r * cols + c;
        const F8 xv = load8(x + o), yv = load8(y + o);
        F8 v;
        #pragma unroll
        for (int k = 0; k < 8; ++k) v.v[k] = __fadd_rn(xv.v[k], __fmul_rn(yv.v[k], ev.v[k]));
        store8(out + o, v);
    }
}

__global__ void residual_add8(const float* x, const B* y, float* out, int64_t n8) {
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n8; i += int64_t(gridDim.x) * blockDim.x) {
        const F8 xv = load8(x + 8 * i), yv = load8(y + 8 * i);
        F8 v;
        #pragma unroll
        for (int k = 0; k < 8; ++k) v.v[k] = __fadd_rn(xv.v[k], yv.v[k]);
        store8(out + 8 * i, v);
    }
}
}  // namespace

extern "C" {
int w21_cat_cast_bf16(const void* a, const void* y, void* out, int64_t a_n, int64_t y_n, cudaStream_t s) {
    cat_cast_bf16<<<blocks(a_n + y_n), kThreads, 0, s>>>((const float*)a, (const float*)y, (B*)out, a_n, a_n + y_n);
    return cudaGetLastError();
}
int w21_sinusoidal(const void* t, void* out, int dim, cudaStream_t s) {
    sinusoidal_f64<<<1, dim / 2, 0, s>>>((const int64_t*)t, (float*)out, dim / 2);
    return cudaGetLastError();
}
// residual_layer_norm: mode kModulate or kAffineBf16 on x (fp32, updated in place) + y * eg (gate) or + y.
int w21_residual_layer_norm(int mode, int gate, void* x, const void* y, const void* eg, const void* gamma, const void* beta,
                            const void* e0, const void* e1, void* out, int64_t rows, int cols, float eps, cudaStream_t s) {
    if (cols % 4 || rows < 1 || !vec8(8, {x, y, out}) || (gate && !vec8(8, {eg}))) return cudaErrorInvalidValue;
    const dim3 threads(32, 4);
    const auto g = (const float*)gamma, be = (const float*)beta, a0 = (const float*)e0, a1 = (const float*)e1;
    const auto xf = (float*)x;
    const auto yb = (const B*)y;
    const auto ge = (const float*)eg;
#define W21_RLN_I(M, I) \
    if (gate) residual_layer_norm<M, true, I><<<unsigned(rows), threads, 0, s>>>(xf, yb, ge, g, be, a0, a1, out, cols, eps); \
    else residual_layer_norm<M, false, I><<<unsigned(rows), threads, 0, s>>>(xf, yb, ge, g, be, a0, a1, out, cols, eps);
#define W21_RLN(M) if (cols == 5120) { W21_RLN_I(M, 10) } else { W21_RLN_I(M, 0) }
    if (mode == kModulate) { W21_RLN(kModulate) } else if (mode == kAffineBf16) { W21_RLN(kAffineBf16) } else return cudaErrorInvalidValue;
#undef W21_RLN
#undef W21_RLN_I
    return cudaGetLastError();
}
// in_dtype: 0 fp32, 1 bf16, 2 fp16.
int w21_layer_norm(int mode, int in_dtype, const void* x, const void* gamma, const void* beta, const void* e0, const void* e1,
                   void* out, int64_t rows, int cols, float eps, int round_out, cudaStream_t s) {
    if (cols % 4 || rows < 1) return cudaErrorInvalidValue;
    const dim3 threads(32, 4);
    const auto g = (const float*)gamma, be = (const float*)beta, a0 = (const float*)e0, a1 = (const float*)e1;
    // 4-element accesses need 16-byte-aligned x and out (rows are whole multiples of 4 elements).
    const bool v = (reinterpret_cast<uintptr_t>(x) | reinterpret_cast<uintptr_t>(out)) % 16 == 0;
#define W21_LN_I(T, V, I)                                                                                                 \
    switch (mode) {                                                                                                       \
    case kModulate: layer_norm<T, kModulate, V, I><<<unsigned(rows), threads, 0, s>>>((const T*)x, g, be, a0, a1, out, cols, eps, round_out); break; \
    case kAffineBf16: layer_norm<T, kAffineBf16, V, I><<<unsigned(rows), threads, 0, s>>>((const T*)x, g, be, a0, a1, out, cols, eps, round_out); break; \
    case kAffineF32: layer_norm<T, kAffineF32, V, I><<<unsigned(rows), threads, 0, s>>>((const T*)x, g, be, a0, a1, out, cols, eps, round_out); break; \
    case kModulateF32: layer_norm<T, kModulateF32, V, I><<<unsigned(rows), threads, 0, s>>>((const T*)x, g, be, a0, a1, out, cols, eps, round_out); break; \
    default: return cudaErrorInvalidValue;                                                                                \
    }
#define W21_LN_V(T, V) if (cols == 5120) { W21_LN_I(T, V, 10) } else { W21_LN_I(T, V, 0) }
#define W21_LN(T) if (v) { W21_LN_V(T, true) } else { W21_LN_V(T, false) }
    if (in_dtype == 0) { W21_LN(float) } else if (in_dtype == 1) { W21_LN(B) } else { W21_LN(__half) }
#undef W21_LN
#undef W21_LN_V
#undef W21_LN_I
    return cudaGetLastError();
}
int w21_rms_apply(const void* x, const void* mean, const void* w, void* out, int64_t rows, int64_t cols, float eps, cudaStream_t s) {
    if (vec8(cols, {x, w, out}))
        rms_apply8<<<row_grid(rows, cols), kVecThreads, 0, s>>>((const B*)x, (const float*)mean, (const B*)w, (B*)out, cols, eps, rows);
    else
        rms_apply<<<blocks(rows * cols), kThreads, 0, s>>>((const B*)x, (const float*)mean, (const B*)w, (B*)out, cols, eps, rows * cols);
    return cudaGetLastError();
}
// WanRMSNorm (with the mean from wan_square_mean_bf16) then rope, without the normalized tensor.
int w21_rms_rope(const void* x, const void* mean, const void* weight, const void* freqs, void* out, int heads, int f_, int h, int w,
                 float eps, cudaStream_t s) {
    const int64_t tokens = int64_t(f_) * h * w, cols = int64_t(heads) * 128;
    if (!vec8(cols, {x, weight, out})) return cudaErrorInvalidValue;
    if (heads == 40 && vec8(8, {freqs}))  // Wan 14B: five heads per thread
        rms_rope8_heads<5><<<unsigned(tokens < 8192 ? tokens : 8192), 128, 0, s>>>((const B*)x, (const float*)mean,
            (const B*)weight, (const double*)freqs, (B*)out, cols, eps, h, w, tokens);
    else
        rms_rope8<<<row_grid(tokens, cols), kVecThreads, 0, s>>>((const B*)x, (const float*)mean, (const B*)weight,
            (const double*)freqs, (B*)out, cols, eps, h, w, tokens);
    return cudaGetLastError();
}
// WanRMSNorm of bf16 [rows, 5120] (then rope when freqs is set, rows = f_ h w tokens) with the mean of
// squares reduced in the kernel; only for rows wan_square_mean_bf16 reduces with square_mean_rows
// (wan_square_mean_warp_rows), whose factor this takes.
int w21_rms_rows(const void* x, const void* weight, const void* freqs, void* out, int64_t rows, int f_, int h, int w, float eps,
                 float factor, cudaStream_t s) {
    if (rows < 1 || !vec8(8, {x, weight, out}) || (freqs && (!vec8(8, {freqs}) || rows != int64_t(f_) * h * w)))
        return cudaErrorInvalidValue;
    const unsigned grid = unsigned((rows + 3) / 4);
    if (freqs) rms_rows<true><<<grid, dim3(32, 4), 0, s>>>((const B*)x, (const B*)weight, (const double*)freqs, (B*)out, rows, eps, factor, h, w);
    else rms_rows<false><<<grid, dim3(32, 4), 0, s>>>((const B*)x, (const B*)weight, nullptr, (B*)out, rows, eps, factor, 0, 0);
    return cudaGetLastError();
}
int w21_rope(const void* x, const void* freqs, void* out, void* out_f32, int heads, int f_, int h, int w, cudaStream_t s) {
    const int64_t tokens = int64_t(f_) * h * w, n = tokens * heads * 64, cols = int64_t(heads) * 128;
    if (!out_f32 && vec8(cols, {x, out}))
        rope8<<<row_grid(tokens, cols), kVecThreads, 0, s>>>((const B*)x, (const double*)freqs, (B*)out, cols, h, w, tokens);
    else
        rope<<<blocks(n), kThreads, 0, s>>>((const B*)x, (const double*)freqs, (B*)out, (float*)out_f32, heads, h, w, tokens);
    return cudaGetLastError();
}
int w21_residual_gate(const void* x, int x_bf16, const void* y, const void* e, void* out, int64_t rows, int64_t cols, cudaStream_t s) {
    const int64_t n = rows * cols;
    if (vec8(cols, {x, y, e, out})) {
        const dim3 g = row_grid(rows, cols);
        if (x_bf16) residual_gate8<B><<<g, kVecThreads, 0, s>>>((const B*)x, (const B*)y, (const float*)e, (float*)out, cols, rows);
        else residual_gate8<float><<<g, kVecThreads, 0, s>>>((const float*)x, (const B*)y, (const float*)e, (float*)out, cols, rows);
    } else if (x_bf16) {
        residual_gate<B><<<blocks(n), kThreads, 0, s>>>((const B*)x, (const B*)y, (const float*)e, (float*)out, cols, n);
    } else {
        residual_gate<float><<<blocks(n), kThreads, 0, s>>>((const float*)x, (const B*)y, (const float*)e, (float*)out, cols, n);
    }
    return cudaGetLastError();
}
int w21_residual_add(const void* x, const void* y, void* out, int64_t n, cudaStream_t s) {
    const int64_t n8 = vec8(n, {x, y, out}) ? n / 8 : 0;
    if (n8 > 0) {
        const int64_t want = (n8 + kThreads - 1) / kThreads;
        residual_add8<<<unsigned(want < 65536 * 8 ? want : 65536 * 8), kThreads, 0, s>>>((const float*)x, (const B*)y, (float*)out, n8);
    }
    if (n8 * 8 < n)
        residual_add<<<blocks(n - n8 * 8), kThreads, 0, s>>>((const float*)x + n8 * 8, (const B*)y + n8 * 8, (float*)out + n8 * 8,
                                                             n - n8 * 8);
    return cudaGetLastError();
}
int w21_add_bf16_f32(const void* a, const void* e, void* out, int64_t n, int64_t e_period, cudaStream_t s) {
    add_bf16_f32<<<blocks(n), kThreads, 0, s>>>((const B*)a, (const float*)e, (float*)out, n, e_period);
    return cudaGetLastError();
}
// 0: fp32 -> bf16, 1: fp16 -> fp32, 2: bf16 -> fp32.
int w21_cast(int kind, const void* x, void* out, int64_t n, cudaStream_t s) {
    switch (kind) {
    case 0: cast<float, B><<<blocks(n), kThreads, 0, s>>>((const float*)x, (B*)out, n); break;
    case 1: cast<__half, float><<<blocks(n), kThreads, 0, s>>>((const __half*)x, (float*)out, n); break;
    case 2: cast<B, float><<<blocks(n), kThreads, 0, s>>>((const B*)x, (float*)out, n); break;
    default: return cudaErrorInvalidValue;
    }
    return cudaGetLastError();
}
int w21_unpatchify(const void* u, void* out, int f_, int h, int w, int c, cudaStream_t s) {
    const int64_t n = int64_t(c) * f_ * 4 * h * w;
    unpatchify<<<blocks(n), kThreads, 0, s>>>((const float*)u, (float*)out, f_, h, w, c, n);
    return cudaGetLastError();
}
}
