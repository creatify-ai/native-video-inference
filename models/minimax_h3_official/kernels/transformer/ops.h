#pragma once
#include <stdint.h>
#include <cuda_runtime.h>
#ifdef __cplusplus
extern "C" {
#endif
// The caller owns buffers, validates extents/indices and retains resources until
// stream completion. All inputs are contiguous except modulation tables whose
// row stride is explicit in elements. No function allocates memory.
int h3_rms_bf16(const void*, const void*, void*, int64_t rows, int cols, float eps, cudaStream_t);
int h3_modulate_bf16(void*, const void *scale, const void *shift, const int64_t*, int64_t rows, int cols, int table_stride, cudaStream_t);
int h3_gated_bf16(const void*, const void *branch, const void *gate, const int64_t*, void*, int64_t rows, int cols, int table_stride, cudaStream_t);
int h3_swiglu_bf16(const void*, void*, int64_t rows, int cols, cudaStream_t);
// RoPE input and output must not alias; input must already have been normalized.
int h3_rope_bf16(const void*, const void *cos, const void *sin, void*, int64_t tokens, int heads, int rotary, cudaStream_t);
int h3_add_bf16(const void*, const void*, void*, int64_t elements, cudaStream_t);
int h3_rows_copy(const void*, const int64_t*, void*, int64_t rows, int cols, int dtype_bytes, int scatter, cudaStream_t);
int h3_cast(const void*, void*, int64_t elements, int to_float, cudaStream_t);
int h3_euler_f32(const float*, const float*, float*, int64_t elements, float sigma_from_timestep, float ratio, cudaStream_t);
int h3_qkv_split_bf16(const void*, void*, void*, void*, int64_t tokens, int inner, cudaStream_t);
int h3_rotary_tables(const double*, const float *inv_freq, float *cos_out, float *sin_out, int64_t tokens, cudaStream_t);
#ifdef __cplusplus
}
#endif
