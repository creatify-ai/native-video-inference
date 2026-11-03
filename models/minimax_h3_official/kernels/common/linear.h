#pragma once
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Caller owns stream and both device workspaces. They outlive this context and
// every queued operation. B200 defaults: 32 MiB cuBLAS, 1 MiB Lt.
int h3_linear_create(void* stream, void* blas_workspace, size_t blas_workspace_bytes,
    void* lt_workspace, size_t lt_workspace_bytes, int allow_tf32, int allow_reduced,
    void** out);
void h3_linear_destroy(void* handle);
int h3_linear_cublas_version(void* handle, int* version);
const char* h3_linear_error(void);
// Register representative Qwen backend nodes under stream capture. Never
// launch or instantiate a graph; model parameters and outputs are untouched.
int h3_linear_prepare_qwen(void* handle);

// Contiguous row-major tensors: input[M,K], weight[N,K], bias[N], output[M,N].
// dtype=0: BF16; dtype=1: FP32. bias=null selects the matching no-bias backend.
// Enqueues on the context stream. No implicit synchronization or allocation of
// GPU tensor memory. Return nonzero on unsupported inputs/library errors.
int h3_linear_forward(void* handle, const void* input, const void* weight,
    const void* bias, void* output, int rows, int out_features, int in_features,
    int dtype);
// JSON array of reference-pinned shape decisions (bitwise-identical candidates only).
// Returns -2 when the buffer is too small.
int h3_linear_pinned_report(void* handle, char* out, int size);
int h3_linear_tune_against(void* handle, const void* input, const void* weight, const void* bias,
    void* output, int rows, int out_features, int in_features, int dtype, int* exact_found);
int h3_linear_forward_pinned(void* handle, const void* input, const void* weight, const void* bias,
    void* output, int rows, int out_features, int in_features, int dtype);
// 1 when the two device buffers are bitwise equal; synchronizes the context stream.
int h3_linear_bitwise_equal(void* handle, const void* a, const void* b, size_t bytes, int* equal);

#ifdef __cplusplus
}
#endif
