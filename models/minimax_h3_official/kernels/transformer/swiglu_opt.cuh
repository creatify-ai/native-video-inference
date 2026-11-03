// Candidate indexing/loads only. Arithmetic is the accepted ops.cu expression.
#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <climits>
#include <cmath>

namespace h3_swiglu_opt {
using B = __nv_bfloat16;
constexpr unsigned cols = 14336;
constexpr unsigned vectors_per_row = cols / 8;

__device__ __forceinline__ unsigned pair(unsigned value, unsigned gate) {
    B v0 = __ushort_as_bfloat16(static_cast<unsigned short>(value));
    B v1 = __ushort_as_bfloat16(static_cast<unsigned short>(value >> 16));
    float g0 = __bfloat162float(__ushort_as_bfloat16(static_cast<unsigned short>(gate)));
    float g1 = __bfloat162float(__ushort_as_bfloat16(static_cast<unsigned short>(gate >> 16)));
    float s0 = __bfloat162float(__float2bfloat16_rn(g0 / (1.f + expf(-g0))));
    float s1 = __bfloat162float(__float2bfloat16_rn(g1 / (1.f + expf(-g1))));
    unsigned o0 = __bfloat16_as_ushort(__float2bfloat16_rn(__bfloat162float(v0) * s0));
    unsigned o1 = __bfloat16_as_ushort(__float2bfloat16_rn(__bfloat162float(v1) * s1));
    return o0 | (o1 << 16);
}

__device__ __forceinline__ uint4 activate(uint4 value, uint4 gate) {
    return make_uint4(pair(value.x, gate.x), pair(value.y, gate.y),
                      pair(value.z, gate.z), pair(value.w, gate.w));
}

template<int Threads>
__global__ void row_v8(const uint4* __restrict__ x, uint4* __restrict__ out) {
    unsigned row = blockIdx.x;
    uint64_t input = uint64_t(row) * (2 * vectors_per_row);
    uint64_t output = uint64_t(row) * vectors_per_row;
    #pragma unroll 1
    for (unsigned c = threadIdx.x; c < vectors_per_row; c += Threads) {
        uint4 v = x[input + c], g = x[input + c + vectors_per_row];
        out[output + c] = activate(v, g);
    }
}

template<int Threads>
__global__ void flat_v8(const uint4* __restrict__ x, uint4* __restrict__ out, unsigned count) {
    unsigned stride = gridDim.x * Threads;
    for (unsigned i = blockIdx.x * Threads + threadIdx.x; i < count; i += stride) {
        // Both are compile-time, 32-bit integer operations; no runtime int64 divide.
        unsigned row = i / vectors_per_row;
        unsigned c = i % vectors_per_row;
        uint64_t offset = uint64_t(row) * (2 * vectors_per_row) + c;
        out[i] = activate(x[offset], x[offset + vectors_per_row]);
    }
}

inline bool supported(const void* x, const void* out, int64_t rows, int columns) {
    return x && out && columns == int(cols) && rows > 0 && rows <= INT_MAX / cols
        && (reinterpret_cast<uintptr_t>(x) & 15) == 0
        && (reinterpret_cast<uintptr_t>(out) & 15) == 0;
}

template<int Variant>
inline int launch(const void* x, void* out, int64_t rows, cudaStream_t stream) {
    const auto* input = reinterpret_cast<const uint4*>(x);
    auto* output = reinterpret_cast<uint4*>(out);
    if constexpr (Variant == 1) row_v8<128><<<unsigned(rows), 128, 0, stream>>>(input, output);
    if constexpr (Variant == 2) row_v8<256><<<unsigned(rows), 256, 0, stream>>>(input, output);
    unsigned count = unsigned(rows) * vectors_per_row;
    if constexpr (Variant == 3) {
        unsigned grid = (count + 127) / 128;
        if (grid > 65535) grid = 65535;
        flat_v8<128><<<grid, 128, 0, stream>>>(input, output, count);
    }
    if constexpr (Variant == 4) {
        unsigned grid = (count + 255) / 256;
        if (grid > 65535) grid = 65535;
        flat_v8<256><<<grid, 256, 0, stream>>>(input, output, count);
    }
    return cudaGetLastError();
}
} // namespace h3_swiglu_opt
