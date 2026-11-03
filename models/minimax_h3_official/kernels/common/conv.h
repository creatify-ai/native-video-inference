#pragma once
#include <stdint.h>

// Thin cuDNN v8 interface, with no ATen dependency. All strides are in elements.
// Conv1d is represented as [N,C,1,L] / [O,I,1,K], as in pinned PyTorch.
// dtype: 0 = FP32, 1 = FP16, 2 = BF16. Bias is a separate original operation.
#include "../../../../core_infra/kernels/conv.h"
typedef InfraConvSpec H3ConvSpec;

#ifdef __cplusplus
extern "C" {
#endif
int h3_conv_create(void** context);
int h3_conv_forward(void* context, const H3ConvSpec* spec,
                    const void* input, const void* weight, void* output, void* stream);
const char* h3_conv_error(void);
const char* h3_conv_plan_tag(void* context);
uint64_t h3_conv_workspace_bytes(void* context);
// Synchronize and free scratch: synchronize the handle stream and free scratch, preserving plans.
int h3_conv_release_workspace(void* context);
int h3_conv_destroy(void* context);
#ifdef __cplusplus
}
#endif
