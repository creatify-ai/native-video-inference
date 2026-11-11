// torch's fp32 GEMM path (at::cuda::blas::gemm<float> -> cublasSgemm) on the reference libcublas,
// loaded at run time: the MTP draft's hc_head computes F.linear(x, hc_fn) in fp32, and cuBLAS picks
// its kernel (gemv / dot for one row, SIMT SGEMMs, split-K) from the handle's configuration, which
// this shim sets as torch's handle pool does: its own handle on the caller's stream, default math
// mode (TF32 off: torch.backends.cuda.matmul.allow_tf32 is False), torch's cuBLAS workspace.
#include <cublas_v2.h>
#include <dlfcn.h>
#include <cstdio>

namespace {
struct Cublas {
  void* lib = nullptr;
  cublasHandle_t handle = nullptr;
  decltype(&cublasCreate_v2) create;
  decltype(&cublasSetStream_v2) set_stream;
  decltype(&cublasSetWorkspace_v2) set_workspace;
  decltype(&cublasSetMathMode) set_math_mode;
  decltype(&cublasSgemm_v2) sgemm;
};
Cublas g;

template <typename F>
bool sym(F& f, const char* name) {
  f = reinterpret_cast<F>(dlsym(g.lib, name));
  return f != nullptr;
}
}  // namespace

extern "C" {

// Load `path` (libcublas.so.N) and create the handle. Returns 1 on success.
int dsv4_cublas_open(const char* path, char* err, size_t cap) {
  if (g.handle) return 1;
  g.lib = dlopen(path, RTLD_NOW | RTLD_LOCAL);
  if (!g.lib) { snprintf(err, cap, "dlopen %s: %s", path, dlerror()); return 0; }
  bool ok = sym(g.create, "cublasCreate_v2") && sym(g.set_stream, "cublasSetStream_v2") &&
            sym(g.set_workspace, "cublasSetWorkspace_v2") && sym(g.set_math_mode, "cublasSetMathMode") && sym(g.sgemm, "cublasSgemm_v2");
  if (!ok) { snprintf(err, cap, "libcublas lacks a symbol"); return 0; }
  if (g.create(&g.handle) != CUBLAS_STATUS_SUCCESS) { snprintf(err, cap, "cublasCreate failed"); return 0; }
  if (g.set_math_mode(g.handle, CUBLAS_DEFAULT_MATH) != CUBLAS_STATUS_SUCCESS) { snprintf(err, cap, "cublasSetMathMode failed"); return 0; }
  return 1;
}

// Column-major C[m, n] = op(A)[m, k] * op(B)[k, n] (alpha 1, beta 0), as cublasSgemm. Returns 1 on success.
int dsv4_sgemm(int trans_a, int trans_b, int m, int n, int k, const float* a, int lda, const float* b, int ldb, float* c, int ldc,
               void* workspace, size_t workspace_bytes, void* stream, char* err, size_t cap) {
  if (!g.handle) { snprintf(err, cap, "cuBLAS not opened"); return 0; }
  if (g.set_stream(g.handle, static_cast<cudaStream_t>(stream)) != CUBLAS_STATUS_SUCCESS) { snprintf(err, cap, "cublasSetStream failed"); return 0; }
  if (g.set_workspace(g.handle, workspace, workspace_bytes) != CUBLAS_STATUS_SUCCESS) { snprintf(err, cap, "cublasSetWorkspace failed"); return 0; }
  const float alpha = 1.0f, beta = 0.0f;
  cublasStatus_t st = g.sgemm(g.handle, trans_a ? CUBLAS_OP_T : CUBLAS_OP_N, trans_b ? CUBLAS_OP_T : CUBLAS_OP_N, m, n, k, &alpha, a, lda, b, ldb,
                              &beta, c, ldc);
  if (st != CUBLAS_STATUS_SUCCESS) { snprintf(err, cap, "cublasSgemm failed: %d", static_cast<int>(st)); return 0; }
  return 1;
}

}  // extern "C"
