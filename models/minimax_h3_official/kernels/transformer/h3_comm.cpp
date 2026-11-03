#include "h3_comm.h"
#include "contract.hpp"
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <nccl.h>
#include <array>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <memory>
#include <string>
#include <thread>

#if NCCL_VERSION_CODE < NCCL_VERSION(2, 14, 0)
#error "h3_comm requires NCCL >= 2.14 for nonblocking communicator initialization"
#endif
namespace {
using namespace h3_comm_contract;
using Clock = std::chrono::steady_clock;
thread_local std::string last_error;
struct Failure : std::runtime_error {
    int code;
    Failure(int code_, const std::string& text) : std::runtime_error(text), code(code_) {}
};
void cu(cudaError_t e) { if (e != cudaSuccess) throw Failure(2, cudaGetErrorString(e)); }
void nc(ncclResult_t e) { if (e != ncclSuccess && e != ncclInProgress) throw Failure(3, ncclGetErrorString(e)); }
ncclDataType_t nc_type(int dtype) {
    dtype_bytes(dtype);
    switch (dtype) {
        case H3_COMM_U8: return ncclUint8;
        case H3_COMM_F16: return ncclFloat16;
        case H3_COMM_BF16: return ncclBfloat16;
        case H3_COMM_F32: return ncclFloat32;
        case H3_COMM_F64: return ncclFloat64;
        default: return ncclInt32;
    }
}
int save_error() noexcept {
    try { throw; }
    catch (const Failure& e) { last_error = e.what(); return e.code; }
    catch (const std::exception& e) { last_error = e.what(); return 1; }
    catch (...) { last_error = "unknown native exception"; return 1; }
}
void *at(void *p, size_t elements, size_t bytes) {
    return reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(p) + mul(elements, bytes));
}
const void *at(const void *p, size_t elements, size_t bytes) {
    return at(const_cast<void *>(p), elements, bytes);
}
}

struct h3_comm {
    int device = 0, rank = 0, world = 1;
    bool validate = false, failed = false;
    uint64_t timeout_ms = 0, sequence = 0;
    ncclComm_t nccl = nullptr;
    cudaStream_t stream = nullptr;
    cudaStream_t owned_stream = nullptr;
    void *metadata_send = nullptr, *metadata_recv = nullptr, *barrier_value = nullptr;
    Metadata host_metadata_send, host_metadata_recv;
    ~h3_comm() {
        (void)cudaSetDevice(device);
        // Abort is deliberately used for best-effort drop, never an implicit
        // blocking destroy collective. The supervisor remains the final watchdog.
        if (nccl) (void)ncclCommAbort(nccl);
        if (metadata_send) (void)cudaFree(metadata_send);
        if (metadata_recv) (void)cudaFree(metadata_recv);
        if (barrier_value) (void)cudaFree(barrier_value);
        if (owned_stream) (void)cudaStreamDestroy(owned_stream);
    }
    Clock::time_point deadline() const { return Clock::now() + std::chrono::milliseconds(timeout_ms); }
    void check_deadline(Clock::time_point end) const {
        if (Clock::now() >= end) throw Failure(4, "NCCL/CUDA progress deadline exceeded; abort all ranks");
    }
    void poll(ncclResult_t state, Clock::time_point end) {
        nc(state);
        while (state == ncclInProgress) {
            check_deadline(end);
            nc(ncclCommGetAsyncError(nccl, &state)); nc(state);
            if (state == ncclInProgress) std::this_thread::sleep_for(std::chrono::microseconds(50));
        }
    }
    void posted(ncclResult_t state) { poll(state, deadline()); }
    void sync() {
        const auto end = deadline();
        for (;;) {
            if (nccl) {
                ncclResult_t state;
                nc(ncclCommGetAsyncError(nccl, &state));
                poll(state, end);
            }
            const auto state = cudaStreamQuery(stream);
            if (state == cudaSuccess) return;
            if (state != cudaErrorNotReady) cu(state);
            check_deadline(end);
            std::this_thread::sleep_for(std::chrono::microseconds(50));
        }
    }
    void preflight(int op, int dtype, size_t count = 0, int root = 0,
                   const size_t *sc = nullptr, const size_t *rc = nullptr,
                   const size_t *rows = nullptr) {
        ++sequence;
        if (!validate || world == 1) return;
        auto& local = host_metadata_send;
        auto& all = host_metadata_recv;
        std::fill(local.begin(), local.end(), 0);
        local[0] = sequence; local[1] = op; local[2] = dtype;
        local[3] = count; local[4] = root; local[5] = world;
        for (int p = 0; p < world; ++p) {
            if (sc) local[meta_send_offset + p] = sc[p];
            if (rc) local[meta_recv_offset + p] = rc[p];
            if (rows) local[meta_rows_offset + p] = rows[p];
        }
        cu(cudaMemcpyAsync(metadata_send, local.data(), local.size() * sizeof(uint64_t), cudaMemcpyHostToDevice, stream));
        posted(ncclAllGather(metadata_send, metadata_recv, meta_words, ncclUint64, nccl, stream));
        cu(cudaMemcpyAsync(all.data(), metadata_recv, all.size() * sizeof(uint64_t), cudaMemcpyDeviceToHost, stream));
        sync();
        agreement(all, world);
    }
    template<class F> void group(F fn) {
        nc(ncclGroupStart());
        try { fn(); }
        catch (...) { (void)ncclGroupEnd(); throw; }
        posted(ncclGroupEnd());
    }
};

namespace {
template<class F> int call(h3_comm *c, F fn) noexcept {
    last_error.clear();
    try {
        require(c != nullptr, "null communicator");
        require(!c->failed, "communicator is poisoned; destroy and recreate all ranks");
        cu(cudaSetDevice(c->device));
        fn(); return 0;
    } catch (...) { if (c) c->failed = true; return save_error(); }
}
void local_copy(h3_comm *c, void *dst, const void *src, size_t bytes) {
    if (bytes && dst != src) cu(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, c->stream));
}
}

extern "C" const char *h3_comm_error(void) { return last_error.c_str(); }
extern "C" size_t h3_comm_unique_id_size(void) { return sizeof(ncclUniqueId); }
extern "C" int h3_comm_get_unique_id(void *id, size_t bytes) {
    last_error.clear();
    try {
        require(id && bytes == sizeof(ncclUniqueId), "unique ID extent mismatch");
        ncclUniqueId native; nc(ncclGetUniqueId(&native)); std::memcpy(id, &native, bytes); return 0;
    } catch (...) { return save_error(); }
}
extern "C" int h3_comm_create(int device, int rank, int world, const void *id, size_t id_bytes,
                               void *stream, uint64_t timeout_ms, int validate, h3_comm **out) {
    last_error.clear();
    if (out) *out = nullptr;
    try {
        require(out != nullptr, "null communicator output");
        rank_world(rank, world);
        require(timeout_ms > 0 && timeout_ms <= 3600000, "timeout must be 1..3600000 ms");
        require(validate == 0 || validate == 1, "validate must be 0 or 1");
        require(world == 1 || (id && id_bytes == sizeof(ncclUniqueId)), "unique ID extent mismatch");
        auto c = std::make_unique<h3_comm>();
        c->device = device; c->rank = rank; c->world = world;
        c->timeout_ms = timeout_ms; c->validate = validate != 0;
        cu(cudaSetDevice(device));
        if (stream) c->stream = static_cast<cudaStream_t>(stream);
        else { cu(cudaStreamCreateWithFlags(&c->owned_stream, cudaStreamNonBlocking)); c->stream = c->owned_stream; }
        if (world > 1) {
            ncclUniqueId native; std::memcpy(&native, id, sizeof(native));
            ncclConfig_t config = NCCL_CONFIG_INITIALIZER; config.blocking = 0;
            c->posted(ncclCommInitRankConfig(&c->nccl, world, native, rank, &config));
        }
        cu(cudaMalloc(&c->barrier_value, sizeof(int32_t)));
        cu(cudaMemsetAsync(c->barrier_value, 0, sizeof(int32_t), c->stream));
        if (c->validate && world > 1) {
            c->host_metadata_send.resize(meta_words);
            c->host_metadata_recv.resize(world * meta_words);
            cu(cudaMalloc(&c->metadata_send, meta_words * sizeof(uint64_t)));
            cu(cudaMalloc(&c->metadata_recv, world * meta_words * sizeof(uint64_t)));
        }
        *out = c.release(); return 0;
    } catch (...) { return save_error(); }
}
extern "C" void *h3_comm_stream(h3_comm *c) { return c ? c->stream : nullptr; }
extern "C" int h3_comm_set_stream(h3_comm *c, void *borrowed_stream) {
    return call(c, [&] {
        c->sync();
        if (borrowed_stream) c->stream = static_cast<cudaStream_t>(borrowed_stream);
        else {
            if (!c->owned_stream) cu(cudaStreamCreateWithFlags(&c->owned_stream, cudaStreamNonBlocking));
            c->stream = c->owned_stream;
        }
    });
}
extern "C" int h3_comm_synchronize(h3_comm *c) { return call(c, [&] { c->sync(); }); }
extern "C" int h3_comm_duplicate(h3_comm *c, h3_comm **out) {
    if (out) *out = nullptr;
    return call(c, [&] {
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 26, 0)
        require(out != nullptr && c->world > 1, "duplicate requires communicator and output");
        int version = 0, runtime = 0; nc(ncclGetVersion(&version)); cu(cudaRuntimeGetVersion(&runtime));
        const char *ordering = std::getenv("NCCL_LAUNCH_ORDER_IMPLICIT");
        require(version >= 22600 && runtime >= 12030 && ordering && std::strcmp(ordering,"1") == 0,
                "overlap requires NCCL >=2.26, CUDA >=12.3, implicit launch ordering");
        c->sync();
        auto child = std::make_unique<h3_comm>();
        child->device=c->device; child->rank=c->rank; child->world=c->world;
        child->timeout_ms=c->timeout_ms; child->validate=false;
        cu(cudaStreamCreateWithFlags(&child->owned_stream,cudaStreamNonBlocking));
        child->stream=child->owned_stream;
        ncclConfig_t config=NCCL_CONFIG_INITIALIZER; config.blocking=0; config.splitShare=0;
        c->posted(ncclCommSplit(c->nccl,0,c->rank,&child->nccl,&config));
        *out=child.release();
#else
        throw Failure(1,"overlap requires NCCL >=2.26 headers");
#endif
    });
}
extern "C" void h3_comm_destroy(h3_comm *c) { delete c; }
extern "C" int h3_comm_close(h3_comm *c) {
    const auto result = call(c, [&] {
        c->sync();
        if (c->nccl) {
            c->posted(ncclCommFinalize(c->nccl));
            // NCCL can return success with asynchronous finalize still pending.
            ncclResult_t state; nc(ncclCommGetAsyncError(c->nccl, &state)); c->posted(state);
            nc(ncclCommDestroy(c->nccl)); c->nccl = nullptr;
        }
    });
    delete c; return result;
}
extern "C" int h3_comm_alltoallv(h3_comm *c, const void *send, size_t send_elements,
                                  const size_t *sc, const size_t *so, void *recv,
                                  size_t recv_elements, const size_t *rc, const size_t *ro, int dtype) {
    return call(c, [&] {
        const auto bytes = dtype_bytes(dtype); const auto type = nc_type(dtype);
        intervals(sc, so, c->world, send_elements, bytes, false);
        intervals(rc, ro, c->world, recv_elements, bytes, true);
        disjoint(send, mul(send_elements, bytes), recv, mul(recv_elements, bytes));
        require(sc[c->rank] == rc[c->rank], "self send/receive count mismatch");
        c->preflight(1, dtype, 0, 0, sc, rc);
        // Self copy outside NCCL group; no zero-count pointer dereference.
        if (sc[c->rank]) local_copy(c, at(recv, ro[c->rank], bytes), at(send, so[c->rank], bytes), mul(sc[c->rank], bytes));
        if (c->world == 1) return;
        c->group([&] {
            for (int p = 0; p < c->world; ++p) if (p != c->rank) {
                if (sc[p]) nc(ncclSend(at(send, so[p], bytes), sc[p], type, p, c->nccl, c->stream));
                if (rc[p]) nc(ncclRecv(at(recv, ro[p], bytes), rc[p], type, p, c->nccl, c->stream));
            }
        });
    });
}
extern "C" int h3_comm_allgather(h3_comm *c, const void *send, size_t send_elements,
                                  void *recv, size_t recv_elements, int dtype) {
    return call(c,[&] {
        const auto bytes=dtype_bytes(dtype); const auto type=nc_type(dtype);
        require(recv_elements==mul(send_elements,c->world),"allgather output extent mismatch");
        disjoint(send,mul(send_elements,bytes),recv,mul(recv_elements,bytes));
        c->preflight(6,dtype,send_elements);
        if(c->world==1) { local_copy(c,recv,send,mul(send_elements,bytes)); return; }
        c->posted(ncclAllGather(send,recv,send_elements,type,c->nccl,c->stream));
    });
}
extern "C" int h3_comm_allgatherv(h3_comm *c, const void *send, size_t send_elements, void *recv,
                                   size_t recv_elements, const size_t *rows, size_t width, int dtype) {
    return call(c, [&] {
        require(rows != nullptr, "null row counts");
        const auto bytes = dtype_bytes(dtype); const auto type = nc_type(dtype);
        std::vector<size_t> counts(c->world);
        for (int p = 0; p < c->world; ++p) counts[p] = mul(rows[p], width);
        auto off = offsets(counts);
        require(send_elements == counts[c->rank], "allgatherv input extent mismatch");
        require(recv_elements == add(off.back(), counts.back()), "allgatherv output extent mismatch");
        disjoint(send, mul(send_elements, bytes), recv, mul(recv_elements, bytes));
        c->preflight(2, dtype, width, 0, nullptr, nullptr, rows);
        if (send_elements) local_copy(c, at(recv, off[c->rank], bytes), send, mul(send_elements, bytes));
        if (c->world == 1) return;
        c->group([&] {
            for (int p = 0; p < c->world; ++p) if (p != c->rank) {
                if (send_elements) nc(ncclSend(send, send_elements, type, p, c->nccl, c->stream));
                if (counts[p]) nc(ncclRecv(at(recv, off[p], bytes), counts[p], type, p, c->nccl, c->stream));
            }
        });
    });
}
extern "C" int h3_comm_broadcast(h3_comm *c, void *buffer, size_t elements, int dtype, int root) {
    return call(c, [&] {
        const auto bytes = mul(elements, dtype_bytes(dtype)); const auto type = nc_type(dtype);
        pointer(buffer, bytes); require(root >= 0 && root < c->world, "invalid broadcast root");
        c->preflight(3, dtype, elements, root);
        if (c->world > 1 && elements) c->posted(ncclBroadcast(buffer, buffer, elements, type, root, c->nccl, c->stream));
    });
}
extern "C" int h3_comm_allreduce_sum(h3_comm *c, const void *send, void *recv, size_t elements, int dtype) {
    return call(c, [&] {
        const auto bytes = mul(elements, dtype_bytes(dtype)); const auto type = nc_type(dtype);
        disjoint(send, bytes, recv, bytes, true);
        c->preflight(4, dtype, elements);
        if (c->world == 1) local_copy(c, recv, send, bytes);
        else if (elements) c->posted(ncclAllReduce(send, recv, elements, type, ncclSum, c->nccl, c->stream));
    });
}
extern "C" int h3_comm_barrier(h3_comm *c) {
    return call(c, [&] {
        c->preflight(5, H3_COMM_I32, 1);
        if (c->world > 1) c->posted(ncclAllReduce(c->barrier_value, c->barrier_value, 1, ncclInt32, ncclSum, c->nccl, c->stream));
        c->sync();
    });
}
extern "C" int h3_comm_any(h3_comm *c, int value, int *result) {
    return call(c, [&] {
        require(result != nullptr && (value == 0 || value == 1), "invalid consensus value");
        c->preflight(7, H3_COMM_I32, 1);
        int32_t flag = value;
        cu(cudaMemcpyAsync(c->barrier_value, &flag, sizeof(flag), cudaMemcpyHostToDevice, c->stream));
        if (c->world > 1) c->posted(ncclAllReduce(c->barrier_value, c->barrier_value, 1, ncclInt32, ncclMax, c->nccl, c->stream));
        cu(cudaMemcpyAsync(&flag, c->barrier_value, sizeof(flag), cudaMemcpyDeviceToHost, c->stream));
        c->sync();
        *result = flag;
    });
}

namespace {
struct IpcRecord { cudaIpcMemHandle_t handle; uint64_t offset; uint64_t size; uint64_t valid; };
void group_barrier(h3_comm *c, int op) {
    c->preflight(op, H3_COMM_I32, 1);
    if (c->world > 1) c->posted(ncclAllReduce(c->barrier_value, c->barrier_value, 1, ncclInt32, ncclSum, c->nccl, c->stream));
}
}
extern "C" int h3_comm_ipc_open(h3_comm *c, const void *local, void **views, void **bases, int count) {
    return call(c, [&] {
        require(local && views && bases && count == c->world && c->world > 1, "ipc open arguments");
        CUdeviceptr base = 0; size_t size = 0;
        require(cuMemGetAddressRange(&base, &size, reinterpret_cast<CUdeviceptr>(local)) == CUDA_SUCCESS,
                "cuMemGetAddressRange failed for local buffer");
        IpcRecord mine{};
        mine.offset = reinterpret_cast<uintptr_t>(local) - static_cast<uintptr_t>(base); mine.size = size; mine.valid = 1;
        cu(cudaIpcGetMemHandle(&mine.handle, reinterpret_cast<void *>(static_cast<uintptr_t>(base))));
        std::vector<IpcRecord> all(c->world);
        void *dsend = nullptr, *drecv = nullptr;
        cu(cudaMalloc(&dsend, sizeof(IpcRecord)));
        if (cudaMalloc(&drecv, sizeof(IpcRecord) * c->world) != cudaSuccess) { (void)cudaFree(dsend); throw Failure(2, "ipc staging allocation failed"); }
        try {
            cu(cudaMemcpyAsync(dsend, &mine, sizeof(mine), cudaMemcpyHostToDevice, c->stream));
            c->preflight(9, H3_COMM_U8, sizeof(IpcRecord));
            c->posted(ncclAllGather(dsend, drecv, sizeof(IpcRecord), ncclUint8, c->nccl, c->stream));
            cu(cudaMemcpyAsync(all.data(), drecv, sizeof(IpcRecord) * c->world, cudaMemcpyDeviceToHost, c->stream));
            c->sync();
        } catch (...) { (void)cudaFree(dsend); (void)cudaFree(drecv); throw; }
        (void)cudaFree(dsend); (void)cudaFree(drecv);
        for (int p = 0; p < c->world; ++p) { views[p] = nullptr; bases[p] = nullptr; }
        for (int p = 0; p < c->world; ++p) {
            if (p == c->rank) { views[p] = const_cast<void *>(local); continue; }
            require(all[p].valid == 1 && all[p].offset < all[p].size, "peer ipc record invalid");
            void *peer = nullptr;
            cu(cudaIpcOpenMemHandle(&peer, all[p].handle, cudaIpcMemLazyEnablePeerAccess));
            bases[p] = peer;
            views[p] = reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(peer) + all[p].offset);
        }
    });
}
extern "C" int h3_comm_ipc_close(h3_comm *c, void **bases, int count) {
    return call(c, [&] {
        require(bases && count == c->world, "ipc close arguments");
        group_barrier(c, 5); c->sync();
        for (int p = 0; p < c->world; ++p) if (p != c->rank && bases[p]) { cu(cudaIpcCloseMemHandle(bases[p])); bases[p] = nullptr; }
        group_barrier(c, 5); c->sync();
    });
}
extern "C" int h3_comm_ipc_unmap(h3_comm *c, void **bases, int count) {
    return call(c, [&] {
        require(bases && count == c->world, "ipc unmap arguments");
        for (int p = 0; p < c->world; ++p) if (p != c->rank && bases[p]) { (void)cudaIpcCloseMemHandle(bases[p]); bases[p] = nullptr; }
    });
}
extern "C" int h3_comm_stream_barrier(h3_comm *c) {
    return call(c, [&] { group_barrier(c, 8); });
}
