// Compatibility ABI and H3 configuration; numerical operators live in core infra.
#include "../../../../core_infra/kernels/linear.cu"
#include "linear.h"
namespace {
int h3_fallback_policy() {
    const char* value=std::getenv("H3_LINEAR_FORCE_FALLBACK");
    if(!value || std::strcmp(value,"0")==0)return 0;
    if(std::strcmp(value,"1")==0)return 1;
    return -1;
}
}
extern "C" int h3_linear_create(void* stream, void* blas_workspace, size_t blas_workspace_bytes,
    void* lt_workspace, size_t lt_workspace_bytes, int allow_tf32, int allow_reduced,
    void** out) {return infra_linear_create(stream, blas_workspace, blas_workspace_bytes, lt_workspace, lt_workspace_bytes, allow_tf32, allow_reduced, out, h3_fallback_policy);}
extern "C" void h3_linear_destroy(void* handle) {infra_linear_destroy(handle);}
extern "C" int h3_linear_cublas_version(void* handle, int* version) {return infra_linear_cublas_version(handle, version);}
extern "C" const char* h3_linear_error(void) {return infra_linear_error();}
extern "C" int h3_linear_forward(void* handle, const void* input, const void* weight,
    const void* bias, void* output, int rows, int out_features, int in_features,
    int dtype) {return infra_linear_forward(handle, input, weight, bias, output, rows, out_features, in_features, dtype);}
extern "C" int h3_linear_pinned_report(void* handle, char* out, int size) {return infra_linear_pinned_report(handle, out, size);}
extern "C" int h3_linear_tune_against(void* handle, const void* input, const void* weight, const void* bias,
    void* output, int rows, int out_features, int in_features, int dtype, int* exact_found) {return infra_linear_tune_against(handle, input, weight, bias, output, rows, out_features, in_features, dtype, exact_found);}
extern "C" int h3_linear_forward_pinned(void* handle, const void* input, const void* weight, const void* bias,
    void* output, int rows, int out_features, int in_features, int dtype) {return infra_linear_forward_pinned(handle, input, weight, bias, output, rows, out_features, in_features, dtype);}
extern "C" int h3_linear_bitwise_equal(void* handle, const void* a, const void* b, size_t bytes, int* equal) {return infra_linear_bitwise_equal(handle, a, b, bytes, equal);}
extern "C" void h3_linear_fallback_reset(){infra_linear_fallback_reset();}
extern "C" int h3_linear_fallback_counts(uint64_t* out,int size){return infra_linear_fallback_counts(out,size);}
