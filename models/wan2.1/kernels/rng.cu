// torch.randn (normal_) on CUDA for contiguous bf16/fp32 tensors, as pinned PyTorch 2.13
// launches it (aten/src/ATen/native/cuda/DistributionTemplates.h): grid-stride Philox
// kernel, curand_init(seed, thread, offset), curand_normal4 unrolled 4x across the grid,
// value = rand * std + mean, converted to the output dtype.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cstdint>

namespace {
template <typename T>
__global__ void __launch_bounds__(256, 4) normal_grid_stride(int64_t numel, uint64_t seed, uint64_t offset,
                                                             float mean, float stdv, T* out) {
    constexpr int unroll_factor = 4;
    int64_t idx = ((int64_t)blockIdx.x) * blockDim.x + threadIdx.x;
    curandStatePhilox4_32_10_t state;
    curand_init(seed, idx, offset, &state);
    int64_t rounded_size = ((numel - 1) / (blockDim.x * gridDim.x * unroll_factor) + 1) * blockDim.x * gridDim.x * unroll_factor;
    for (int64_t linear_index = idx; linear_index < rounded_size; linear_index += blockDim.x * gridDim.x * unroll_factor) {
        float4 rand = curand_normal4(&state);
#pragma unroll
        for (int ii = 0; ii < unroll_factor; ii++) {
            int64_t li = linear_index + blockDim.x * gridDim.x * ii;
            if (li < numel) {
                float value = (&rand.x)[ii] * stdv + mean;
                if constexpr (sizeof(T) == 2) out[li] = __float2bfloat16_rn(value); else out[li] = value;
            }
        }
        __syncthreads();
    }
}
}  // namespace

extern "C" int wan_randn(void* out, int bf16, int64_t numel, uint64_t seed, uint64_t offset, unsigned grid,
                         cudaStream_t stream) {
    if (bf16) normal_grid_stride<__nv_bfloat16><<<grid, 256, 0, stream>>>(numel, seed, offset, 0.0f, 1.0f, (__nv_bfloat16*)out);
    else normal_grid_stride<float><<<grid, 256, 0, stream>>>(numel, seed, offset, 0.0f, 1.0f, (float*)out);
    return cudaGetLastError();
}
