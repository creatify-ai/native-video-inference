#pragma once
#include <stdint.h>
#include <stddef.h>

// Thin cuDNN v8 interface, with no ATen dependency. All strides are in elements.
// Conv1d is represented as [N,C,1,L] / [O,I,1,K], as in pinned PyTorch.
// dtype: 0 = FP32, 1 = FP16, 2 = BF16. Bias is a separate original operation.
typedef struct InfraConvSpec {
    int32_t ndim, dtype, transposed, allow_tf32, deterministic, channels_last;
    int64_t input_dims[5], input_strides[5];
    int64_t weight_dims[5], weight_strides[5];
    int64_t output_dims[5], output_strides[5];
    int64_t padding[3], stride[3], dilation[3];
} InfraConvSpec;

#ifdef __cplusplus
extern "C" {
#endif
int infra_conv_create(void** context, size_t plan_limit, size_t workspace_limit);
int infra_conv_forward(void* context, const InfraConvSpec* spec,
                    const void* input, const void* weight, void* output, void* stream);
const char* infra_conv_error(void);
// Optional: called when a workspace cudaMalloc fails, to release cached device memory (the caller's
// allocator pool); the allocation is then retried once. With a handler set, a workspace that still
// cannot be allocated is an error instead of a fall-through to the next engine (which would change
// the algorithm, and the numerics, with available memory).
typedef int (*infra_conv_oom_handler)(void);
void infra_conv_set_oom_handler(infra_conv_oom_handler handler);
const char* infra_conv_plan_tag(void* context);
uint64_t infra_conv_workspace_bytes(void* context);
// Synchronize and free scratch: synchronize the handle stream and free scratch, preserving plans.
int infra_conv_cache_stats(void* context, uint64_t* out);
int infra_conv_release_workspace(void* context, void* stream);
int infra_conv_destroy(void* context);
#ifdef __cplusplus
}
#endif
