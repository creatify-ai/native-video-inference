#ifndef H3_COMM_H
#define H3_COMM_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct h3_comm h3_comm;
enum h3_comm_dtype { H3_COMM_U8 = 0, H3_COMM_F16 = 1, H3_COMM_BF16 = 2,
                     H3_COMM_F32 = 3, H3_COMM_F64 = 4, H3_COMM_I32 = 5 };
/* Status: 0 success, 1 argument/contract, 2 CUDA, 3 NCCL, 4 timeout.
 * Error text is thread-local and valid until the next call on this thread.
 * A communicator is single-host-thread only. Any failed operation poisons it;
 * every rank must stop the request, abort/destroy, and let the supervisor restart.
 * All ranks must enter the same operations in the same order. validate=1 adds
 * synchronous metadata agreement checks; validate=0 relies on the caller.
 * Operations enqueue onto stream, not necessarily complete on return. Device
 * buffers must remain alive and unmodified until stream completion. Buffer
 * extents and offsets/counts are in elements, not bytes. */
const char *h3_comm_error(void);
size_t h3_comm_unique_id_size(void);
/* The process calling get_unique_id must remain alive until all ranks finish
 * communicator initialization: NCCL starts a bootstrap listener in that process. */
int h3_comm_get_unique_id(void *id, size_t id_bytes);
/* borrowed_stream=NULL creates an owned nonblocking stream. Non-NULL borrows
 * the caller's cudaStream_t and never destroys it. The default stream cannot
 * be borrowed. Supports world=1,2,4,8. world=1 accepts id=NULL/id_bytes=0
 * and bypasses NCCL. All ranks must use the same native build. */
int h3_comm_create(int device, int rank, int world, const void *id, size_t id_bytes,
                   void *borrowed_stream, uint64_t timeout_ms, int validate,
                   h3_comm **out);
void *h3_comm_stream(h3_comm *comm);
/* Checks previous stream completion before rebinding. NULL selects/creates the
 * communicator's owned stream. Unbind a borrowed request stream before destroying
 * it. This lets a resident communicator outlive per-request execution arenas. */
int h3_comm_set_stream(h3_comm *comm, void *borrowed_stream);
int h3_comm_synchronize(h3_comm *comm);
/* Collective duplicate with an independent owned stream. Requires NCCL >=2.26,
 * CUDA >=12.3 and NCCL_LAUNCH_ORDER_IMPLICIT=1. Single host thread, identical
 * cross-communicator enqueue order on every rank; no outstanding parent work. */
int h3_comm_duplicate(h3_comm *comm, h3_comm **out);
/* Explicit close reports flush/finalize errors and frees the handle on all paths.
 * destroy is abortive best-effort cleanup; both require an external process
 * watchdog because a failed CUDA driver / native teardown cannot be bounded. */
int h3_comm_close(h3_comm *comm);
void h3_comm_destroy(h3_comm *comm);
/* Separate, non-overlapping send/recv storage is required for alltoallv and
 * allgatherv. Zero-length peers are valid. recv intervals cannot overlap. */
int h3_comm_alltoallv(h3_comm *comm, const void *send, size_t send_elements,
                      const size_t *send_counts, const size_t *send_offsets,
                      void *recv, size_t recv_elements,
                      const size_t *recv_counts, const size_t *recv_offsets,
                      int dtype);
/* rows[rank] * row_width input elements; output concatenates ranks in order. */
int h3_comm_allgatherv(h3_comm *comm, const void *send, size_t send_elements,
                       void *recv, size_t recv_elements, const size_t *rows,
                       size_t row_width, int dtype);
/* Equal counts, native NCCL AllGather, disjoint send/receive storage. */
int h3_comm_allgather(h3_comm *comm, const void *send, size_t send_elements,
                     void *recv, size_t recv_elements, int dtype);
/* In-place broadcast on every rank. */
int h3_comm_broadcast(h3_comm *comm, void *buffer, size_t elements, int dtype, int root);
/* Exact in-place send==recv allowed; partially overlapping buffers rejected.
 * Floating-point SUM is not a bitwise-equivalence guarantee to a single GPU. */
int h3_comm_allreduce_sum(h3_comm *comm, const void *send, void *recv,
                          size_t elements, int dtype);
/* Direct peer views over CUDA IPC. Every rank passes a pointer inside one of its
 * own cudaMalloc allocations; views[p] becomes a device pointer usable in this
 * process for rank p's pointer (views[rank] is the local pointer itself) and
 * bases[p] the mapping to close. Collective and synchronous on the bound stream;
 * all ranks call together with count == world. Peer access is enabled lazily. */
int h3_comm_ipc_open(h3_comm *comm, const void *local, void **views, void **bases, int count);
/* Barrier, unmap every peer base (own entry ignored), barrier again so no rank
 * frees storage that another rank still maps. Collective and synchronous. */
int h3_comm_ipc_close(h3_comm *comm, void **bases, int count);
/* Unmap peer bases without any collective: only for process teardown, after
 * every rank has stopped using the mappings. */
int h3_comm_ipc_unmap(h3_comm *comm, void **bases, int count);
/* Stream-ordered group barrier (one-element allreduce); no host synchronization.
 * Work enqueued after it on every rank's stream starts only once all ranks'
 * previously enqueued work has completed. */
int h3_comm_stream_barrier(h3_comm *comm);
/* Barrier includes GPU completion, unlike the enqueue APIs above. */
int h3_comm_barrier(h3_comm *comm);
/* Synchronous request-boundary consensus; result is 0 or 1 on every rank. */
int h3_comm_any(h3_comm *comm, int value, int *result);
#ifdef __cplusplus
}
#endif
#endif
