// H3-only native arithmetic. CUDA/BF16 storage; no PyTorch host dependency.
// RMSNorm reduction/order adapted from PyTorch v2.13.0 layer_norm_kernel.cu
// and thread_constants.h, BSD-style license; see NOTICE.md.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cmath>
#include "../../../../core_infra/kernels/ops.cu"

using B = __nv_bfloat16;
namespace h3_ops {
__device__ float f(B x) { return __bfloat162float(x); }
__device__ B b(float x) { return __float2bfloat16_rn(x); }
__device__ float rounded(float x) { return f(b(x)); }
constexpr int threads = 256;
unsigned grid(int64_t n) { return unsigned((n + threads - 1) / threads > 65535 ? 65535 : (n + threads - 1) / threads); }

// Exactly the reference's 4 values/thread, 32 lanes x 4 warps, downward
// shuffle reduction followed by pairwise warp reduction. No fast-math build.

__global__ void rope(const B *x, const B *cos, const B *sin, B *out,
                     int64_t n, int heads, int rotary) {
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += int64_t(gridDim.x) * blockDim.x) {
        int col = i % 128;
        if (col >= rotary) { out[i] = x[i]; continue; }
        int half = rotary / 2;
        int64_t base = i - col;
        int64_t pos = (i / (heads * 128)) * rotary + col;
        float partner = f(x[base + (col < half ? col + half : col - half)]);
        if (col < half) partner = -partner;
        out[i] = b(rounded(f(x[i]) * f(cos[pos])) + rounded(partner * f(sin[pos])));
    }
}

// The timestep round trip and sigma ratio are supplied as dumped FP32 scalars.
// Explicit operations preserve the eager kernel materialization boundaries.

// MiniMaxH3RotaryPosEmbed: FP64 position -> FP32, FP32 multiply,
// then independent original CUDA cos/sin functions and duplicated axes.
__global__ void rotary_tables(const double *positions, const float *inv_freq,
                              float *cos_out, float *sin_out, int64_t tokens) {
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < tokens * 96; i += int64_t(gridDim.x) * blockDim.x) {
        int d = int(i % 48);
        float position = __double2float_rn(positions[(i / 96) * 3 + d / 16]);
        float angle = __fmul_rn(position, inv_freq[d % 16]);
        cos_out[i] = cosf(angle);
        sin_out[i] = sinf(angle);
    }
}
}

using namespace h3_ops;
extern "C" int h3_rms_bf16(const void *x, const void *weight, void *out, int64_t rows, int cols, float eps, cudaStream_t s) {if(cols!=128 && cols!=5376)return cudaErrorInvalidValue;return infra_rms_bf16(x,weight,out,rows,cols,eps,s);}
extern "C" int h3_modulate_bf16(void *x, const void *scale, const void *shift, const int64_t *index, int64_t rows, int cols, int table_stride, cudaStream_t s) {return infra_modulate_bf16(x,scale,shift,index,rows,cols,table_stride,s);}
extern "C" int h3_gated_bf16(const void *x, const void *branch, const void *gate, const int64_t *index, void *out, int64_t rows, int cols, int table_stride, cudaStream_t s) {return infra_gated_bf16(x,branch,gate,index,out,rows,cols,table_stride,s);}
extern "C" int h3_swiglu_bf16(const void *x, void *out, int64_t rows, int cols, cudaStream_t s) {return infra_swiglu_bf16(x,out,rows,cols,s);}
extern "C" int h3_rope_bf16(const void *x, const void *cos, const void *sin, void *out, int64_t tokens, int heads, int rotary, cudaStream_t s) {
    if (rotary < 0 || rotary > 128 || rotary % 2) return cudaErrorInvalidValue;
    int64_t n = tokens * heads * 128;
    rope<<<grid(n), threads, 0, s>>>((const B*)x, (const B*)cos, (const B*)sin, (B*)out, n, heads, rotary); return cudaGetLastError();
}
extern "C" int h3_add_bf16(const void *x, const void *y, void *out, int64_t n, cudaStream_t s) {return infra_add_bf16(x,y,out,n,s);}
extern "C" int h3_rows_copy(const void *x, const int64_t *index, void *out, int64_t rows, int cols, int dtype_bytes, int scatter, cudaStream_t s) {return infra_rows_copy(x,index,out,rows,cols,dtype_bytes,scatter,s);}
extern "C" int h3_cast(const void *x, void *out, int64_t n, int to_float, cudaStream_t s) {return infra_cast(x,out,n,to_float,s);}
extern "C" int h3_silu_f32(const float *x, void *out, int64_t n, int to_bf16, cudaStream_t s) {return infra_silu_f32(x,out,n,to_bf16,s);}
extern "C" int h3_euler_f32(const float *sample, const float *velocity, float *out, int64_t n, float sigma_from_timestep, float ratio, cudaStream_t s) {return infra_euler_f32(sample,velocity,out,n,sigma_from_timestep,ratio,s);}
extern "C" int h3_qkv_split_bf16(const void *x, void *q, void *k, void *v, int64_t tokens, int inner, cudaStream_t s) {return infra_qkv_split_bf16(x,q,k,v,tokens,inner,s);}

extern "C" int h3_rotary_tables(const double *positions, const float *inv_freq,
                                 float *cos_out, float *sin_out, int64_t tokens, cudaStream_t s) {
    if (!positions || !inv_freq || !cos_out || !sin_out || tokens < 1 || tokens > 1048576)
        return cudaErrorInvalidValue;
    rotary_tables<<<grid(tokens * 96), threads, 0, s>>>(positions, inv_freq, cos_out, sin_out, tokens);
    return cudaGetLastError();
}
