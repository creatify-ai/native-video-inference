#ifndef H3_DENSE_NATIVE_H
#define H3_DENSE_NATIVE_H
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
/* All device tensors are BF16 contiguous [S,H,128], batch 1; H=56.
 * Frozen case-01 variants: S=63510 main and S=2456 refiner.
 * No mask, causal mode, dropout, ragged tensors or backward stats.
 * Caller owns a current B200 CUDA context and keeps all buffers alive until
 * the caller's CUstream/cudaStream_t (void*) has completed. */
int h3_dense_workspace_bytes(int tokens, int heads, size_t *bytes);
int h3_dense_forward(const void *q, const void *k, const void *v, void *out,
                     void *workspace, size_t bytes, int tokens, int heads,
                     float scale, void *stream);
const char *h3_dense_error(void);
size_t h3_dense_cudnn_version(void);
const char *h3_dense_plan_name(int tokens, int heads);
int h3_dense_dump_plan(int tokens, int heads, const char *path);
int h3_dense_cleanup(void);
/* v2 explicitly identifies the original PyTorch view, independent of S.
 * layout=0: main SHD.transpose(0,1).unsqueeze(0), batch stride H*D.
 * layout=1: refiner BSHD.transpose(1,2), batch stride S*H*D.
 * Admission is structural; numerical validation is reported per real shape. */
int h3_dense_workspace_bytes_v2(int tokens, int heads, int layout, size_t *bytes);
int h3_dense_forward_v2(const void *q, const void *k, const void *v, void *out,
                       void *workspace, size_t bytes, int tokens, int heads,
                       int layout, float scale, void *stream);
const char *h3_dense_plan_name_v2(int tokens, int heads, int layout);
int h3_dense_dump_plan_v2(int tokens, int heads, int layout, const char *path);
#ifdef __cplusplus
}
#endif
#endif
