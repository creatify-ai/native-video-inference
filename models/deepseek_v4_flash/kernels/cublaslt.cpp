// torch's cuBLASLt GEMM path (at::cuda::blas gemm via cublasLtMatmul), against the reference's own
// libcublasLt (nvidia-cublas 13.1.1.3, loaded from a pinned path): the same descriptors,
// preferences and first heuristic result, so the library picks the reference's algorithm and
// launches the same kernels. The calls mirror the reference's CUBLASLT_LOG_LEVEL=5 trace:
//
//   computeDesc=[computeType=COMPUTE_32F scaleType=R_32F transa=OP_T smCountTarget=148]
//   preference=[maxWorkspaceSizeinBytes=33554432 minBytesAlignmentA..D=<pointer alignment, <=16>
//               implMask=0xFFFFFFFEFFFFFFFF]
//   A = weight [rows=K cols=N ld=K] (bf16), B = input [K, M] (bf16), C = D = out [N, M] (bf16 / fp32)
//   cublasLtMatmul(alpha=1, beta=0, algo=heuristic[0], workspaceSize=heuristic[0].workspaceSize)
#include <cublasLt.h>
#include <dlfcn.h>

#include <cstdint>
#include <cstdio>
#include <mutex>

namespace {

struct Lt {
  void* lib = nullptr;
  decltype(&cublasLtCreate) create;
  decltype(&cublasLtMatmulDescCreate) desc_create;
  decltype(&cublasLtMatmulDescDestroy) desc_destroy;
  decltype(&cublasLtMatmulDescSetAttribute) desc_set;
  decltype(&cublasLtMatrixLayoutCreate) layout_create;
  decltype(&cublasLtMatrixLayoutDestroy) layout_destroy;
  decltype(&cublasLtMatmulPreferenceCreate) pref_create;
  decltype(&cublasLtMatmulPreferenceDestroy) pref_destroy;
  decltype(&cublasLtMatmulPreferenceSetAttribute) pref_set;
  decltype(&cublasLtMatmulAlgoGetHeuristic) heuristic;
  decltype(&cublasLtMatmul) matmul;
  decltype(&cublasLtGetVersion) version;
  cublasLtHandle_t handle = nullptr;
};

Lt g_lt;
std::mutex g_mutex;

template <typename F>
bool sym(F& f, const char* name) {
  f = reinterpret_cast<F>(dlsym(g_lt.lib, name));
  return f != nullptr;
}

// at::cuda::blas::_getAlignment: the largest of 16/8/4/2/1 bytes dividing the address.
uint32_t alignment(const void* p) {
  auto a = reinterpret_cast<uintptr_t>(p);
  for (uint32_t v : {16u, 8u, 4u, 2u}) if (a % v == 0) return v;
  return 1;
}

}  // namespace

extern "C" {

// Load libcublasLt from `path` (once). Returns its version, or 0 with err set.
size_t dsv4_lt_init(const char* path, char* err, size_t cap) {
  std::lock_guard<std::mutex> lock(g_mutex);
  if (g_lt.handle) return g_lt.version();
  g_lt.lib = dlopen(path, RTLD_NOW | RTLD_LOCAL);
  if (!g_lt.lib) { snprintf(err, cap, "dlopen %s: %s", path, dlerror()); return 0; }
  bool ok = sym(g_lt.create, "cublasLtCreate") && sym(g_lt.desc_create, "cublasLtMatmulDescCreate") &&
            sym(g_lt.desc_destroy, "cublasLtMatmulDescDestroy") && sym(g_lt.desc_set, "cublasLtMatmulDescSetAttribute") &&
            sym(g_lt.layout_create, "cublasLtMatrixLayoutCreate") && sym(g_lt.layout_destroy, "cublasLtMatrixLayoutDestroy") &&
            sym(g_lt.pref_create, "cublasLtMatmulPreferenceCreate") && sym(g_lt.pref_destroy, "cublasLtMatmulPreferenceDestroy") &&
            sym(g_lt.pref_set, "cublasLtMatmulPreferenceSetAttribute") && sym(g_lt.heuristic, "cublasLtMatmulAlgoGetHeuristic") &&
            sym(g_lt.matmul, "cublasLtMatmul") && sym(g_lt.version, "cublasLtGetVersion");
  if (!ok) { snprintf(err, cap, "%s lacks a cublasLt entry point", path); return 0; }
  if (g_lt.create(&g_lt.handle) != CUBLAS_STATUS_SUCCESS) { snprintf(err, cap, "cublasLtCreate failed"); return 0; }
  return g_lt.version();
}

// out [m, n] = x [m, k] . w [n, k]^T, x / w bf16, out bf16 (out_f32 = 0) or fp32 (out_f32 = 1).
// Writes the chosen algorithm's id / tile / splits into info[0..4] for diagnostics.
int dsv4_lt_linear(int m, int n, int k, const void* x, const void* w, void* out, int out_f32, void* workspace,
                   size_t workspace_cap, int sm_count, void* stream, int32_t* info, char* err, size_t cap) {
  if (!g_lt.handle) { snprintf(err, cap, "cuBLASLt not initialized"); return 1; }
  cublasLtMatmulDesc_t desc = nullptr;
  cublasLtMatrixLayout_t a = nullptr, b = nullptr, c = nullptr;
  cublasLtMatmulPreference_t pref = nullptr;
  int rc = 1;
  do {
    if (g_lt.desc_create(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F) != CUBLAS_STATUS_SUCCESS) { snprintf(err, cap, "desc"); break; }
    cublasOperation_t ta = CUBLAS_OP_T;
    g_lt.desc_set(desc, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof ta);
    int32_t sms = sm_count;
    g_lt.desc_set(desc, CUBLASLT_MATMUL_DESC_SM_COUNT_TARGET, &sms, sizeof sms);
    cudaDataType_t ct = out_f32 ? CUDA_R_32F : CUDA_R_16BF;
    if (g_lt.layout_create(&a, CUDA_R_16BF, k, n, k) || g_lt.layout_create(&b, CUDA_R_16BF, k, m, k) ||
        g_lt.layout_create(&c, ct, n, m, n)) { snprintf(err, cap, "layout"); break; }
    if (g_lt.pref_create(&pref) != CUBLAS_STATUS_SUCCESS) { snprintf(err, cap, "preference"); break; }
    uint64_t ws = workspace_cap;
    g_lt.pref_set(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof ws);
    uint32_t al[4] = {alignment(w), alignment(x), alignment(out), alignment(out)};
    g_lt.pref_set(pref, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES, &al[0], sizeof al[0]);
    g_lt.pref_set(pref, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES, &al[1], sizeof al[1]);
    g_lt.pref_set(pref, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES, &al[2], sizeof al[2]);
    g_lt.pref_set(pref, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES, &al[3], sizeof al[3]);
    uint64_t mask = 0xFFFFFFFEFFFFFFFFull;
    g_lt.pref_set(pref, CUBLASLT_MATMUL_PREF_IMPL_MASK, &mask, sizeof mask);
    cublasLtMatmulHeuristicResult_t result = {};
    int returned = 0;
    auto st = g_lt.heuristic(g_lt.handle, desc, a, b, c, c, pref, 1, &result, &returned);
    if (st != CUBLAS_STATUS_SUCCESS || returned == 0) { snprintf(err, cap, "no cuBLASLt heuristic (status %d)", (int)st); break; }
    const float alpha = 1.0f, beta = 0.0f;
    st = g_lt.matmul(g_lt.handle, desc, &alpha, w, a, x, b, &beta, out, c, out, c, &result.algo, workspace, result.workspaceSize,
                     static_cast<cudaStream_t>(stream));
    if (st != CUBLAS_STATUS_SUCCESS) { snprintf(err, cap, "cublasLtMatmul status %d", (int)st); break; }
    if (info) {
      const int32_t* raw = reinterpret_cast<const int32_t*>(&result.algo);
      for (int i = 0; i < 4; ++i) info[i] = raw[i];
      info[4] = static_cast<int32_t>(result.workspaceSize);
    }
    rc = 0;
  } while (false);
  if (pref) g_lt.pref_destroy(pref);
  if (c) g_lt.layout_destroy(c);
  if (b) g_lt.layout_destroy(b);
  if (a) g_lt.layout_destroy(a);
  if (desc) g_lt.desc_destroy(desc);
  return rc;
}

}  // extern "C"
