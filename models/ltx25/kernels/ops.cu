// Eager operation order of LTX 2.5 rectified-flow ancestral Euler. No ATen linkage.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>

namespace {
__global__ void ancestral(const float* x, const float* denoised, const void* noise,
                         float* out, int64_t n, const float* sigmas, int step,
                         float eta, float s_noise, int noise_bf16) {
    int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float sigma = sigmas[step], next = sigmas[step+1];
    if (next == 0.0f) { out[i] = denoised[i]; return; }
    const float ratio = __fadd_rn(1.0f, __fmul_rn(__fsub_rn(__fdiv_rn(next, sigma), 1.0f), eta));
    const float down = __fmul_rn(next, ratio);
    const float down_ratio = __fdiv_rn(down, sigma);
    float y = __fadd_rn(__fmul_rn(down_ratio, x[i]), __fmul_rn(__fsub_rn(1.0f, down_ratio), denoised[i]));
    if (eta > 0.0f) {
        const float a_next = __fsub_rn(1.0f, next), a_down = __fsub_rn(1.0f, down);
        const float coefficient = __fsqrt_rn(fmaxf(0.0f, __fsub_rn(__fmul_rn(next, next),
            __fdiv_rn(__fmul_rn(__fmul_rn(down, down), __fmul_rn(a_next, a_next)), __fmul_rn(a_down, a_down)))));
        const float z = noise_bf16 ? __bfloat162float(((const __nv_bfloat16*)noise)[i]) : ((const float*)noise)[i];
        y = __fadd_rn(__fmul_rn(__fdiv_rn(a_next, a_down), y), __fmul_rn(__fmul_rn(z, s_noise), coefficient));
    }
    out[i] = y;
}
__global__ void gelu(const __nv_bfloat16* x, __nv_bfloat16* y, int64_t n) {
    int64_t i = int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if (i >= n) return;
    const float v = __bfloat162float(x[i]);
    constexpr float beta = M_SQRT2 * M_2_SQRTPI * float(0.5);
    const float cube = v * v * v;
    const float inner = beta * (v + 0.044715f * cube);
    y[i] = __float2bfloat16_rn(0.5f * v * (1.0f + tanhf(inner)));
}
}
extern "C" int ltx25_ancestral(const void* x, const void* denoised, const void* noise,
    void* out, int64_t n, const float* sigmas, int step, float eta, float s_noise,
    int noise_bf16, cudaStream_t stream) {
    if (!n) return 0;
    ancestral<<<(n+255)/256,256,0,stream>>>((const float*)x,(const float*)denoised,noise,
        (float*)out,n,sigmas,step,eta,s_noise,noise_bf16);
    return cudaGetLastError();
}
extern "C" int ltx25_gelu_bf16(const void* x, void* y, int64_t n, cudaStream_t stream) {
    if (n) gelu<<<(n+255)/256,256,0,stream>>>((const __nv_bfloat16*)x,(__nv_bfloat16*)y,n);
    return cudaGetLastError();
}
