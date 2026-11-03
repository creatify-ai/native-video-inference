// Wan ABI over the core cuBLAS/cuBLASLt port of pinned Torch 2.13 addmm/mm.
#include "../../../core_infra/kernels/linear.cu"
namespace {
int wan_fallback_policy() {
    const char* value = std::getenv("WAN_LINEAR_FORCE_FALLBACK");
    if (!value || std::strcmp(value, "0") == 0) return 0;
    if (std::strcmp(value, "1") == 0) return 1;
    return -1;
}
}
extern "C" int wan_linear_create(void* stream, void* blas_workspace, size_t blas_workspace_bytes,
    void* lt_workspace, size_t lt_workspace_bytes, int allow_tf32, int allow_reduced, void** out) {
    return infra_linear_create(stream, blas_workspace, blas_workspace_bytes, lt_workspace, lt_workspace_bytes,
        allow_tf32, allow_reduced, out, wan_fallback_policy);
}
