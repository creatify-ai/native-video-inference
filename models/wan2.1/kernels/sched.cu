// Classifier-free guidance and FlowUniPCMultistepScheduler.step (wan/utils/fm_solvers_unipc.py) as
// the reference computes them with offload_model=True: fp32 tensors on the CPU, one IEEE rounding per
// eager op (CPU sub is add with alpha -1, an exact fmadd), true division, Python scalars and 0-dim
// tensors as fp32 operands. The per-step scalars come from the host (sampler.rs).
#include <cuda_runtime.h>
#include <cstdint>

namespace {
constexpr int kThreads = 256;
inline unsigned blocks(int64_t n) { return unsigned((n + kThreads - 1) / kThreads); }

// noise_pred = uncond + guide * (cond - uncond)
__global__ void cfg(const float* cond, const float* uncond, float* out, float guide, int64_t n) {
    int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __fadd_rn(uncond[i], __fmul_rn(guide, __fsub_rn(cond[i], uncond[i])));
}
// x0_pred = sample - sigma_t * model_output
__global__ void convert(const float* sample, const float* mo, float* x0, float sigma, int64_t n) {
    int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) x0[i] = __fsub_rn(sample[i], __fmul_rn(sigma, mo[i]));
}

struct Coef {
    float s1, s2, s3;     // sigma_t / sigma_s0, alpha_t * h_phi_1, alpha_t * B_h
    float rk;             // order 2: (lambda_si - lambda_s0) / h
    float rho0, rho1;     // predictor: rho0 = rhos_p[0]; corrector: rhos_c[0], rhos_c[-1]
};
// x_t_ = s1 * x - s2 * m0, then
//   predictor order 1:  x_t_ - s3 * 0                          (pred_res is the int 0)
//   predictor order 2:  x_t_ - s3 * (rho0 * ((m1 - m0) / rk))
//   corrector order 1:  x_t_ - s3 * (0 + rho1 * (mt - m0))     (corr_res is the int 0)
//   corrector order 2:  x_t_ - s3 * (rho0 * ((m1 - m0) / rk) + rho1 * (mt - m0))
// m0 = the newest converted output, m1 the one before, mt = this step's converted output.
__global__ void update(const float* x, const float* m0, const float* m1, const float* mt, float* out, Coef c, int corrector,
                       int order, int64_t n) {
    int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float x_t_ = __fsub_rn(__fmul_rn(c.s1, x[i]), __fmul_rn(c.s2, m0[i]));
    float res;
    if (!corrector) {
        res = order == 1 ? __fmul_rn(c.s3, 0.0f) : __fmul_rn(c.s3, __fmul_rn(c.rho0, __fdiv_rn(__fsub_rn(m1[i], m0[i]), c.rk)));
    } else {
        const float tail = __fmul_rn(c.rho1, __fsub_rn(mt[i], m0[i]));
        const float inner = order == 1 ? __fadd_rn(tail, 0.0f)
                                       : __fadd_rn(__fmul_rn(c.rho0, __fdiv_rn(__fsub_rn(m1[i], m0[i]), c.rk)), tail);
        res = __fmul_rn(c.s3, inner);
    }
    out[i] = __fsub_rn(x_t_, res);
}
}  // namespace

extern "C" {
int w21_cfg(const void* cond, const void* uncond, void* out, float guide, int64_t n, cudaStream_t s) {
    cfg<<<blocks(n), kThreads, 0, s>>>((const float*)cond, (const float*)uncond, (float*)out, guide, n);
    return cudaGetLastError();
}
int w21_convert(const void* sample, const void* mo, void* x0, float sigma, int64_t n, cudaStream_t s) {
    convert<<<blocks(n), kThreads, 0, s>>>((const float*)sample, (const float*)mo, (float*)x0, sigma, n);
    return cudaGetLastError();
}
int w21_unipc_update(const void* x, const void* m0, const void* m1, const void* mt, void* out, float s1, float s2, float s3,
                     float rk, float rho0, float rho1, int corrector, int order, int64_t n, cudaStream_t s) {
    Coef c{s1, s2, s3, rk, rho0, rho1};
    update<<<blocks(n), kThreads, 0, s>>>((const float*)x, (const float*)m0, (const float*)m1, (const float*)mt, (float*)out, c,
                                          corrector, order, n);
    return cudaGetLastError();
}
}
