#pragma once
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Caller owns stream and both device workspaces. They outlive this context and
// every queued operation. Workspace sizes and precision are supplied by the caller.
int infra_linear_create(void* stream, void* blas_workspace, size_t blas_workspace_bytes,
    void* lt_workspace, size_t lt_workspace_bytes, int allow_tf32, int allow_reduced,
    void** out, int (*fallback_policy)());
void infra_linear_destroy(void* handle);
int infra_linear_cublas_version(void* handle, int* version);
const char* infra_linear_error(void);

// Contiguous row-major tensors: input[M,K], weight[N,K], bias[N], output[M,N].
// dtype=0: BF16; dtype=1: FP32; dtype=2: FP16 (same contract as BF16). bias=null selects the matching no-bias backend.
// Enqueues on the context stream. No implicit synchronization or allocation of
// GPU tensor memory. Return nonzero on unsupported inputs/library errors.
int infra_linear_forward(void* handle, const void* input, const void* weight,
    const void* bias, void* output, int rows, int out_features, int in_features,
    int dtype);
// JSON array of reference-pinned shape decisions (bitwise-identical candidates only).
// Returns -2 when the buffer is too small.
int infra_linear_pinned_report(void* handle, char* out, int size);
int infra_linear_tune_against(void* handle, const void* input, const void* weight, const void* bias,
    void* output, int rows, int out_features, int in_features, int dtype, int* exact_found);
int infra_linear_forward_pinned(void* handle, const void* input, const void* weight, const void* bias,
    void* output, int rows, int out_features, int in_features, int dtype);
// 1 when the two device buffers are bitwise equal; synchronizes the context stream.
int infra_linear_bitwise_equal(void* handle, const void* a, const void* b, size_t bytes, int* equal);

#ifdef __cplusplus
}
#endif
