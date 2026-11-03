// Layout-only candidate. All loads/stores preserve the original BF16 bits.
// No floating-point arithmetic or attention-kernel changes.
#pragma once
#include <cuda_runtime.h>
#include <cstdint>

namespace h3_bsa_pack_opt {
// One block owns a 64-token, 64-channel half of one head. The input, Q,
// and K accesses are uint32 pairs of adjacent channels. Shared memory
// changes only V's copy order so that global stores pair adjacent tokens.
__global__ void tiled(const uint16_t* q, const uint16_t* k, const uint16_t* v,
                      uint16_t* qp, uint16_t* kp, uint16_t* vp,
                      int tokens, int heads, int rows) {
    __shared__ uint32_t tile[64][33];
    const int group = int(blockIdx.x) >> 1;
    const int half = int(blockIdx.x) & 1;
    const int head = int(blockIdx.y);
    const int x = int(threadIdx.x), y = int(threadIdx.y);
    const size_t tile_base = ((size_t(head) * (rows / 64) + group) * 2 + half) * 4096;
    const size_t q_base = (size_t(head) * rows + size_t(group) * 64) * 128 + half * 64;
    #pragma unroll
    for (int row = y; row < 64; row += 8) {
        const int token = group * 64 + row;
        const size_t source = (size_t(token) * heads + head) * 128 + half * 64 + 2 * x;
        uint32_t q2 = 0, k2 = 0, v2 = 0;
        if (token < tokens) {
            q2 = *reinterpret_cast<const uint32_t*>(q + source);
            k2 = *reinterpret_cast<const uint32_t*>(k + source);
            v2 = *reinterpret_cast<const uint32_t*>(v + source);
        }
        *reinterpret_cast<uint32_t*>(qp + q_base + row * 128 + 2 * x) = q2;
        *reinterpret_cast<uint32_t*>(kp + tile_base + row * 64 + 2 * x) = k2;
        tile[row][x] = v2;
    }
    __syncthreads();
    #pragma unroll
    for (int channel = y; channel < 64; channel += 8) {
        const int pair = channel >> 1, shift = (channel & 1) * 16;
        const uint32_t a = (tile[2 * x][pair] >> shift) & 0xffffu;
        const uint32_t b = (tile[2 * x + 1][pair] >> shift) & 0xffffu;
        *reinterpret_cast<uint32_t*>(vp + tile_base + channel * 64 + 2 * x) = a | (b << 16);
    }
}
} // namespace h3_bsa_pack_opt
