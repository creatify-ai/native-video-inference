// Generic native arithmetic; retain the original reduction and rounding boundaries.
// PyTorch BSD attribution remains in licenses/NOTICE.md.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cmath>

using B = __nv_bfloat16;
namespace infra_ops {
__device__ float f(B x) { return __bfloat162float(x); }
__device__ B b(float x) { return __float2bfloat16_rn(x); }
__device__ float rounded(float x) { return f(b(x)); }
constexpr int threads = 256;
unsigned grid(int64_t n) { return unsigned((n + threads - 1) / threads > 65535 ? 65535 : (n + threads - 1) / threads); }

__global__ void rms(const B *x, const B *weight, B *out, int cols, float eps) {
    __shared__ float sums[4];
    int lane = threadIdx.x, warp = threadIdx.y, tid = lane + 32 * warp;
    int64_t row = int64_t(blockIdx.x) * cols;
    float sum = 0.f;
    for (int v = tid; v < cols / 4; v += 128) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            float value = f(x[row + v * 4 + j]);
            sum = fmaf(value, value, sum);
        }
    }
    for (int offset = 16; offset; offset >>= 1)
        sum = __shfl_down_sync(0xffffffff, sum, offset) + sum;
    for (int offset = 2; offset; offset >>= 1) {
        if (!lane && warp >= offset && warp < offset * 2) sums[warp - offset] = sum;
        __syncthreads();
        if (!lane && warp < offset) sum = sums[warp] + sum;
        __syncthreads();
    }
    if (!lane && !warp) sums[0] = sum / float(cols);
    __syncthreads();
    float rstd = rsqrtf(sums[0] + eps);
    for (int v = tid; v < cols / 4; v += 128) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            int col = v * 4 + j;
            out[row + col] = b(f(weight[col]) * (rstd * f(x[row + col])));
        }
    }
}

__global__ void modulate(B *x, const B *scale, const B *shift,
                         const int64_t *index, int64_t n, int cols, int table_stride) {
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += int64_t(gridDim.x) * blockDim.x) {
        int64_t t = index[i / cols] * table_stride + i % cols;
        float factor = rounded(1.f + f(scale[t]));
        float product = rounded(f(x[i]) * factor);
        x[i] = b(product + f(shift[t]));
    }
}

__global__ void gated(const B *residual, const B *branch, const B *gate,
                      const int64_t *index, B *out, int64_t n, int cols, int table_stride) {
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += int64_t(gridDim.x) * blockDim.x) {
        int64_t t = index[i / cols] * table_stride + i % cols;
        out[i] = b(f(residual[i]) + rounded(f(gate[t]) * f(branch[i])));
    }
}

__global__ void swiglu(const B *x, B *out, int64_t n, int cols) {
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += int64_t(gridDim.x) * blockDim.x) {
        int64_t offset = (i / cols) * (2 * cols) + i % cols;
        float g = f(x[offset + cols]);
        float silu = rounded(g / (1.f + expf(-g)));
        out[i] = b(f(x[offset]) * silu);
    }
}

__global__ void add(const B *a, const B *c, B *out, int64_t n) {
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += int64_t(gridDim.x) * blockDim.x)
        out[i] = b(f(a[i]) + f(c[i]));
}

template<typename T, bool scatter>
__global__ void rows_copy(const T *input, const int64_t *index, T *out, int64_t n, int cols) {
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += int64_t(gridDim.x) * blockDim.x) {
        int64_t other = index[i / cols] * cols + i % cols;
        if constexpr (scatter) out[other] = input[i]; else out[i] = input[other];
    }
}

__global__ void to_bf16(const float *x, B *out, int64_t n) {
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += int64_t(gridDim.x) * blockDim.x) out[i] = b(x[i]);
}

__global__ void to_f32(const B *x, float *out, int64_t n) {
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += int64_t(gridDim.x) * blockDim.x) out[i] = f(x[i]);
}

template<typename Out>
__global__ void silu_f32(const float *x, Out *out, int64_t n) {
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += int64_t(gridDim.x) * blockDim.x)
        out[i] = Out(x[i] / (1.f + expf(-x[i])));
}

__global__ void euler(const float *sample, const float *velocity, float *out,
                      int64_t n, float sigma_from_timestep, float ratio) {
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += int64_t(gridDim.x) * blockDim.x) {
        float denoised = __fadd_rn(sample[i], __fmul_rn(sigma_from_timestep, velocity[i]));
        out[i] = __fadd_rn(__fmul_rn(ratio, sample[i]), __fmul_rn(__fsub_rn(1.f, ratio), denoised));
    }
}

__global__ void qkv_split(const B *input, B *q, B *k, B *v, int64_t n, int inner) {
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += int64_t(gridDim.x) * blockDim.x) {
        int64_t source = (i / inner) * (3 * inner) + i % inner;
        q[i] = input[source]; k[i] = input[source + inner]; v[i] = input[source + 2 * inner];
    }
}
}
extern "C" int infra_rms_bf16(const void *x, const void *weight, void *out, int64_t rows, int cols, float eps, cudaStream_t s) {
    if (rows < 1 || rows > 0x7fffffff || (cols < 4 || cols % 4) || !x || !weight || !out || !(eps > 0)) return cudaErrorInvalidValue;
    infra_ops::rms<<<unsigned(rows), dim3(32, 4), 0, s>>>((const B*)x, (const B*)weight, (B*)out, cols, eps);
    return cudaGetLastError();
}

extern "C" int infra_modulate_bf16(void *x, const void *scale, const void *shift, const int64_t *index, int64_t rows, int cols, int table_stride, cudaStream_t s) {
    infra_ops::modulate<<<infra_ops::grid(rows * cols), infra_ops::threads, 0, s>>>((B*)x, (const B*)scale, (const B*)shift, index, rows * cols, cols, table_stride); return cudaGetLastError();
}

extern "C" int infra_gated_bf16(const void *x, const void *branch, const void *gate, const int64_t *index, void *out, int64_t rows, int cols, int table_stride, cudaStream_t s) {
    infra_ops::gated<<<infra_ops::grid(rows * cols), infra_ops::threads, 0, s>>>((const B*)x, (const B*)branch, (const B*)gate, index, (B*)out, rows * cols, cols, table_stride); return cudaGetLastError();
}

extern "C" int infra_swiglu_bf16(const void *x, void *out, int64_t rows, int cols, cudaStream_t s) {
    infra_ops::swiglu<<<infra_ops::grid(rows * cols), infra_ops::threads, 0, s>>>((const B*)x, (B*)out, rows * cols, cols); return cudaGetLastError();
}

extern "C" int infra_add_bf16(const void *x, const void *y, void *out, int64_t n, cudaStream_t s) {
    infra_ops::add<<<infra_ops::grid(n), infra_ops::threads, 0, s>>>((const B*)x, (const B*)y, (B*)out, n); return cudaGetLastError();
}

extern "C" int infra_rows_copy(const void *x, const int64_t *index, void *out, int64_t rows, int cols, int dtype_bytes, int scatter, cudaStream_t s) {
    if (dtype_bytes == 2) {
        if (scatter) infra_ops::rows_copy<uint16_t, true><<<infra_ops::grid(rows * cols), infra_ops::threads, 0, s>>>((const uint16_t*)x, index, (uint16_t*)out, rows * cols, cols);
        else infra_ops::rows_copy<uint16_t, false><<<infra_ops::grid(rows * cols), infra_ops::threads, 0, s>>>((const uint16_t*)x, index, (uint16_t*)out, rows * cols, cols);
    } else if (dtype_bytes == 4) {
        if (scatter) infra_ops::rows_copy<uint32_t, true><<<infra_ops::grid(rows * cols), infra_ops::threads, 0, s>>>((const uint32_t*)x, index, (uint32_t*)out, rows * cols, cols);
        else infra_ops::rows_copy<uint32_t, false><<<infra_ops::grid(rows * cols), infra_ops::threads, 0, s>>>((const uint32_t*)x, index, (uint32_t*)out, rows * cols, cols);
    } else return cudaErrorInvalidValue;
    return cudaGetLastError();
}

extern "C" int infra_cast(const void *x, void *out, int64_t n, int to_float, cudaStream_t s) {
    if (to_float) infra_ops::to_f32<<<infra_ops::grid(n), infra_ops::threads, 0, s>>>((const B*)x, (float*)out, n);
    else infra_ops::to_bf16<<<infra_ops::grid(n), infra_ops::threads, 0, s>>>((const float*)x, (B*)out, n);
    return cudaGetLastError();
}

extern "C" int infra_silu_f32(const float *x, void *out, int64_t n, int to_bf16, cudaStream_t s) {
    if (!x || !out || n <= 0) return cudaErrorInvalidValue;
    if (to_bf16) infra_ops::silu_f32<<<infra_ops::grid(n), infra_ops::threads, 0, s>>>(x, (B*)out, n);
    else infra_ops::silu_f32<<<infra_ops::grid(n), infra_ops::threads, 0, s>>>(x, (float*)out, n);
    return cudaGetLastError();
}

extern "C" int infra_euler_f32(const float *sample, const float *velocity, float *out, int64_t n, float sigma_from_timestep, float ratio, cudaStream_t s) {
    infra_ops::euler<<<infra_ops::grid(n), infra_ops::threads, 0, s>>>(sample, velocity, out, n, sigma_from_timestep, ratio); return cudaGetLastError();
}

extern "C" int infra_qkv_split_bf16(const void *x, void *q, void *k, void *v, int64_t tokens, int inner, cudaStream_t s) {
    infra_ops::qkv_split<<<infra_ops::grid(tokens * inner), infra_ops::threads, 0, s>>>((const B*)x, (B*)q, (B*)k, (B*)v, tokens * inner, inner); return cudaGetLastError();
}
