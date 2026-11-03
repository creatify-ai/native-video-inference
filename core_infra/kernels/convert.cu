// Weight dtype conversion for streaming loads (core_infra::model_load): each element converted
// once with round-to-nearest-even, as torch's Tensor.to(dtype) rounds (c10::BFloat16 / c10::Half).
// Dtype codes: 0 F32, 1 F16, 2 BF16.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace infra_convert_detail {
constexpr int threads = 256;
template <typename S, typename D> __device__ D cast(S x);
template <> __device__ __nv_bfloat16 cast(float x) { return __float2bfloat16_rn(x); }
template <> __device__ __half cast(float x) { return __float2half_rn(x); }
template <> __device__ float cast(__nv_bfloat16 x) { return __bfloat162float(x); }
template <> __device__ float cast(__half x) { return __half2float(x); }
template <typename S, typename D>
__global__ void convert(const S* src, D* dst, int64_t n) {
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += int64_t(gridDim.x) * blockDim.x)
        dst[i] = cast<S, D>(src[i]);
}
template <typename S, typename D>
int launch(const void* src, void* dst, int64_t n, cudaStream_t s) {
    int64_t blocks = (n + threads - 1) / threads;
    convert<S, D><<<unsigned(blocks > 65535 ? 65535 : blocks), threads, 0, s>>>((const S*)src, (D*)dst, n);
    return cudaGetLastError();
}
}  // namespace infra_convert_detail

extern "C" int infra_convert(const void* src, int src_dtype, void* dst, int dst_dtype, int64_t n, cudaStream_t s) {
    using namespace infra_convert_detail;
    if (n <= 0) return 0;
    if (src_dtype == 0 && dst_dtype == 2) return launch<float, __nv_bfloat16>(src, dst, n, s);
    if (src_dtype == 0 && dst_dtype == 1) return launch<float, __half>(src, dst, n, s);
    if (src_dtype == 2 && dst_dtype == 0) return launch<__nv_bfloat16, float>(src, dst, n, s);
    if (src_dtype == 1 && dst_dtype == 0) return launch<__half, float>(src, dst, n, s);
    return int(cudaErrorInvalidValue);
}
