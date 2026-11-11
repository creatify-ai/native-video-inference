// Launch tape: a CUPTI injection library that records every kernel launch of a process.
//
//   CUDA_INJECTION64_PATH=/path/libcupti_tape.so TAPE_DIR=/out python ...
//
// Always recorded (from process start):
//   modules/<moduleId>.cubin    every module image the driver loads (written when first launched)
//   modules.jsonl               moduleId, sha-free size, and the CUmodule/CUlibrary handles bound to it
//   tensormaps.jsonl            every cuTensorMapEncodeTiled/Im2col: the 128-byte map and its arguments
//   allocations.jsonl           device allocations (cuMemAlloc*, cuMemCreate+cuMemMap, cuMemAllocAsync)
// Recorded between tape_start(label) and tape_stop() (exported C symbols; call through ctypes):
//   launches.jsonl              seq, label, kernel name, CUmodule, grid, block, dynamic shared memory,
//                               cluster dims and other launch attributes, stream, capture state and
//                               the raw parameter bytes split by cuFuncGetParamInfo.
// The tape is a porting aid: the native code must reproduce each launch (same binary, configuration
// and parameter bytes up to device addresses), and a diff of two tapes locates the first divergence.
#include <cuda.h>
#include <cupti.h>
#include <cuda_runtime_api.h>
#include <dlfcn.h>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <unordered_map>
#include <vector>

namespace {

std::mutex g_mutex;
std::string g_dir;
FILE *g_launches = nullptr, *g_modules = nullptr, *g_maps = nullptr, *g_allocs = nullptr;
std::atomic<bool> g_recording{false};
// TAPE_CENSUS=N: record at most N launches per (function, grid, block, dynamic smem) and no memory
// operations: a compact census of every kernel variant a long run uses (for the kernel bundle).
int g_census = 0;
std::mutex g_census_mutex;
std::unordered_map<std::string, unsigned long> g_census_counts;
std::string g_label;
uint64_t g_seq = 0;
CUpti_SubscriberHandle g_subscriber;

struct Module { std::vector<char> image; bool written = false; };
std::unordered_map<uint32_t, Module> g_images;                 // moduleId -> cubin
std::unordered_map<uintptr_t, std::vector<uint32_t>> g_handles; // CUmodule / CUlibrary -> moduleIds

// Snapshot mode: around each recorded launch, synchronize and save the bytes behind every pointer
// parameter that falls into a recorded device allocation of at most g_snap_max_segment bytes and
// outside the excluded ranges (weights), before and after, capped at g_snap_cap bytes per buffer.
// Payloads go to snapshots.bin (deduplicated by content), indexed by snapshots.jsonl.
std::atomic<bool> g_snapshot{false};
std::vector<std::pair<uint64_t, uint64_t>> g_alloc_ranges;  // every recorded allocation [begin, end)
std::vector<std::pair<uint64_t, uint64_t>> g_excluded;      // sorted [begin, end)
FILE *g_snaps = nullptr, *g_snap_pack = nullptr;
uint64_t g_snap_id = 0, g_pack_offset = 0;
size_t g_snap_cap = 256u << 10, g_snap_max_segment = 128u << 20;
std::unordered_map<std::string, uint64_t> g_pack_index;  // content key -> pack offset
bool g_all_modules = false;  // TAPE_ALL_MODULES=1: write every module image when it loads

struct Snap { uint64_t ptr, bytes; int index; };

struct Frame {
    uint32_t cbid; bool runtime; std::vector<uint32_t> loaded;
    long snap = -1; std::vector<Snap> snaps;  // snapshot mode: this launch's id and buffers
    bool launch = false, recorded = false;  // runtime launch frames: recorded by a nested driver launch?
    bool memop = false;  // runtime memcpy/memset frames: nested driver memory ops are not recorded again
};

// cudaGetFuncBySymbol of the runtime instance that registered the kernel (torch's libcudart.so.13).
typedef cudaError_t (*GetFuncBySymbol)(cudaFunction_t *, const void *);
GetFuncBySymbol runtime_get_func() {
    static GetFuncBySymbol fn = [] {
        const char *names[] = {"libcudart.so.13", "libcudart.so.12", "libcudart.so"};
        for (const char *n : names)
            if (void *h = dlopen(n, RTLD_NOLOAD | RTLD_LAZY))
                if (void *f = dlsym(h, "cudaGetFuncBySymbol")) return (GetFuncBySymbol)f;
        return (GetFuncBySymbol)dlsym(RTLD_DEFAULT, "cudaGetFuncBySymbol");
    }();
    return fn;
}
thread_local std::vector<Frame> t_stack;
thread_local int t_depth = 0;  // re-entrancy guard for driver calls made from the callback

void hex(std::string &out, const void *data, size_t n) {
    static const char *digits = "0123456789abcdef";
    const unsigned char *p = static_cast<const unsigned char *>(data);
    out.reserve(out.size() + 2 * n);
    for (size_t i = 0; i < n; ++i) { out.push_back(digits[p[i] >> 4]); out.push_back(digits[p[i] & 15]); }
}

std::string escape(const char *s) {
    std::string out;
    for (; s && *s; ++s) {
        if (*s == '"' || *s == '\\') { out.push_back('\\'); out.push_back(*s); }
        else if ((unsigned char)*s < 0x20) { char b[8]; snprintf(b, sizeof b, "\\u%04x", *s); out += b; }
        else out.push_back(*s);
    }
    return out;
}

void bind(uintptr_t handle, const std::vector<uint32_t> &ids, const char *kind) {
    if (!handle || ids.empty()) return;
    auto &v = g_handles[handle];
    for (uint32_t id : ids) {
        v.push_back(id);
        fprintf(g_modules, "{\"bind\":\"%s\",\"handle\":%lu,\"module_id\":%u}\n", kind, (unsigned long)handle, id);
    }
    fflush(g_modules);
}

void write_module(uint32_t id) {
    auto it = g_images.find(id);
    if (it == g_images.end() || it->second.written) return;
    std::string path = g_dir + "/modules/" + std::to_string(id) + ".cubin";
    if (FILE *f = fopen(path.c_str(), "wb")) {
        fwrite(it->second.image.data(), 1, it->second.image.size(), f);
        fclose(f);
        it->second.written = true;
    }
}

// Runtime launches pass CUkernel handles where CUfunction is expected; resolve them to the
// context's CUfunction so the CUfunction queries (name, parameters, module) work.
CUfunction as_function(CUfunction f) {
    const char *name = nullptr;
    ++t_depth;
    CUresult r = cuFuncGetName(&name, f);
    CUfunction out = f;
    if (r != CUDA_SUCCESS || !name || !*name) {
        CUfunction g = nullptr;
        if (cuKernelGetFunction(&g, (CUkernel)f) == CUDA_SUCCESS && g) out = g;
    }
    --t_depth;
    return out;
}

// Parameter bytes of one launch, split into the kernel's declared parameters.
std::string params_json(CUfunction f, void **params, void **extra) {
    std::string out = "[";
    if (params) {
        for (size_t i = 0;; ++i) {
            size_t offset = 0, size = 0;
            if (cuFuncGetParamInfo(f, i, &offset, &size) != CUDA_SUCCESS) break;
            if (i) out += ",";
            out += "{\"offset\":" + std::to_string(offset) + ",\"size\":" + std::to_string(size) + ",\"hex\":\"";
            hex(out, params[i], size);
            out += "\"}";
        }
    } else if (extra) {
        void *buffer = nullptr; size_t size = 0;
        for (int i = 0; extra[i] != CU_LAUNCH_PARAM_END; i += 2) {
            if (extra[i] == CU_LAUNCH_PARAM_BUFFER_POINTER) buffer = extra[i + 1];
            else if (extra[i] == CU_LAUNCH_PARAM_BUFFER_SIZE) size = *static_cast<size_t *>(extra[i + 1]);
        }
        out += "{\"offset\":0,\"size\":" + std::to_string(size) + ",\"hex\":\"";
        if (buffer) hex(out, buffer, size);
        out += "\",\"packed\":true}";
    }
    return out + "]";
}

void record_launch(CUfunction f, const unsigned grid[3], const unsigned block[3], unsigned smem, CUstream stream,
                   void **params, void **extra, const CUlaunchAttribute *attrs, unsigned num_attrs, const char *api,
                   long snap = -1) {
    CUfunction handle = f;
    f = as_function(f);
    if (g_census > 0) {
        char key[160];
        snprintf(key, sizeof key, "%p/%u,%u,%u/%u,%u,%u/%u", (void *)f, grid[0], grid[1], grid[2], block[0], block[1], block[2], smem);
        std::lock_guard<std::mutex> lock(g_census_mutex);
        if (++g_census_counts[key] > (unsigned long)g_census) return;
    }
    CUlibrary library = nullptr;
    if (f != handle) { ++t_depth; cuKernelGetLibrary(&library, (CUkernel)handle); --t_depth; }
    ++t_depth;
    const char *name = nullptr;
    cuFuncGetName(&name, f);
    CUmodule module = nullptr;
    cuFuncGetModule(&module, f);
    CUstreamCaptureStatus capture = CU_STREAM_CAPTURE_STATUS_NONE;
    cuStreamIsCapturing(stream, &capture);
    int regs = 0, static_smem = 0, local = 0;
    cuFuncGetAttribute(&regs, CU_FUNC_ATTRIBUTE_NUM_REGS, f);
    cuFuncGetAttribute(&static_smem, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, f);
    cuFuncGetAttribute(&local, CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, f);
    std::string p = params_json(f, params, extra);
    --t_depth;
    std::string a = "[";
    for (unsigned i = 0; i < num_attrs; ++i) {
        if (i) a += ",";
        a += "{\"id\":" + std::to_string((int)attrs[i].id) + ",\"hex\":\"";
        hex(a, &attrs[i].value, sizeof(attrs[i].value));
        a += "\"}";
    }
    a += "]";
    std::lock_guard<std::mutex> lock(g_mutex);
    std::string ids = "[";
    auto it = g_handles.find((uintptr_t)module);
    if (it == g_handles.end() && library) it = g_handles.find((uintptr_t)library);
    if (it != g_handles.end()) {
        for (size_t i = 0; i < it->second.size(); ++i) {
            if (i) ids += ",";
            ids += std::to_string(it->second[i]);
            write_module(it->second[i]);
        }
    }
    ids += "]";
    fprintf(g_launches,
            "{\"seq\":%lu,\"label\":\"%s\",\"api\":\"%s\",\"name\":\"%s\",\"module\":%lu,\"library\":%lu,\"module_ids\":%s,"
            "\"grid\":[%u,%u,%u],\"block\":[%u,%u,%u],\"smem\":%u,\"static_smem\":%d,\"regs\":%d,\"local\":%d,"
            "\"stream\":%lu,\"capturing\":%d,\"snap\":%ld,\"attrs\":%s,\"params\":%s}\n",
            (unsigned long)g_seq++, escape(g_label.c_str()).c_str(), api, escape(name).c_str(),
            (unsigned long)(uintptr_t)module, (unsigned long)(uintptr_t)library, ids.c_str(), grid[0], grid[1], grid[2], block[0], block[1], block[2],
            smem, static_smem, regs, local, (unsigned long)(uintptr_t)stream, (int)capture, snap, a.c_str(), p.c_str());
}

bool is_device_pointer(const void *p) {
    CUmemorytype type = (CUmemorytype)0;
    ++t_depth;
    CUresult r = cuPointerGetAttribute(&type, CU_POINTER_ATTRIBUTE_MEMORY_TYPE, (CUdeviceptr)p);
    --t_depth;
    return r == CUDA_SUCCESS && (type == CU_MEMORYTYPE_DEVICE || type == CU_MEMORYTYPE_UNIFIED);
}

// kind: 1 host->device, 2 device->host, 3 device->device, 4 default (inferred), 0 host->host.
void record_memcpy(void *dst, const void *src, size_t count, int kind, CUstream stream, const char *api) {
    if (g_census > 0) return;
    bool src_dev = is_device_pointer(src), dst_dev = is_device_pointer(dst);
    if (kind == 4) kind = src_dev ? (dst_dev ? 3 : 2) : (dst_dev ? 1 : 0);
    std::string data;
    if (kind == 1 && count <= (64u << 20)) hex(data, src, count);
    std::lock_guard<std::mutex> lock(g_mutex);
    fprintf(g_launches, "{\"seq\":%lu,\"label\":\"%s\",\"api\":\"%s\",\"op\":\"memcpy\",\"kind\":%d,\"dst\":%lu,"
            "\"src\":%lu,\"bytes\":%zu,\"stream\":%lu,\"data\":\"%s\"}\n",
            (unsigned long)g_seq++, escape(g_label.c_str()).c_str(), api, kind, (unsigned long)(uintptr_t)dst,
            (unsigned long)(uintptr_t)src, count, (unsigned long)(uintptr_t)stream, data.c_str());
}

void record_memset(void *dst, unsigned value, size_t count, int width, CUstream stream, const char *api) {
    if (g_census > 0) return;
    std::lock_guard<std::mutex> lock(g_mutex);
    fprintf(g_launches, "{\"seq\":%lu,\"label\":\"%s\",\"api\":\"%s\",\"op\":\"memset\",\"dst\":%lu,\"value\":%u,"
            "\"count\":%zu,\"width\":%d,\"stream\":%lu}\n", (unsigned long)g_seq++, escape(g_label.c_str()).c_str(), api,
            (unsigned long)(uintptr_t)dst, value, count, width, (unsigned long)(uintptr_t)stream);
}

bool in_sorted(const std::vector<std::pair<uint64_t, uint64_t>> &v, uint64_t x) {
    auto it = std::upper_bound(v.begin(), v.end(), std::pair<uint64_t, uint64_t>(x, UINT64_MAX));
    if (it == v.begin()) return false;
    --it;
    return x >= it->first && x < it->second;
}

// Device pointers a launch passes: every 8-byte-aligned word of every parameter (plain pointers,
// pointers inside by-value structs, the global address in word 0 of a CUtensorMap) that falls in an
// eligible allocation. Each distinct address is saved once; `index` packs param * 1024 + word.
std::vector<Snap> snapshot_targets(CUfunction f, void **params) {
    std::vector<Snap> out;
    if (!params || !f) return out;
    f = as_function(f);
    std::vector<uint64_t> seen;
    for (size_t i = 0;; ++i) {
        size_t offset = 0, size = 0;
        ++t_depth;
        bool ok = cuFuncGetParamInfo(f, i, &offset, &size) == CUDA_SUCCESS;
        --t_depth;
        if (!ok) break;
        for (size_t w = 0; w + 8 <= size; w += 8) {
            uint64_t v = 0;
            memcpy(&v, static_cast<const char *>(params[i]) + w, 8);
            if (v < 0x10000 || in_sorted(g_excluded, v)) continue;
            if (std::find(seen.begin(), seen.end(), v) != seen.end()) continue;
            for (auto it = g_alloc_ranges.rbegin(); it != g_alloc_ranges.rend(); ++it) {
                if (v >= it->first && v < it->second) {
                    if (it->second - it->first <= g_snap_max_segment) {
                        out.push_back({v, std::min<uint64_t>(it->second - v, g_snap_cap), (int)(i * 1024 + w / 8)});
                        seen.push_back(v);
                    }
                    break;
                }
            }
        }
    }
    return out;
}

void dump_snaps(long id, const std::vector<Snap> &snaps, const char *phase) {
    if (snaps.empty()) return;
    ++t_depth;
    cuCtxSynchronize();
    for (const Snap &sn : snaps) {
        std::vector<char> host(sn.bytes);
        if (cuMemcpyDtoH(host.data(), (CUdeviceptr)sn.ptr, sn.bytes) != CUDA_SUCCESS) continue;
        // Content key: FNV-1a 64 over the bytes plus the length (collisions only cost deduplication).
        uint64_t h = 1469598103934665603ull;
        for (char c : host) { h ^= (unsigned char)c; h *= 1099511628211ull; }
        std::string key = std::to_string(h) + ":" + std::to_string(host.size());
        std::lock_guard<std::mutex> lock(g_mutex);
        uint64_t offset;
        auto it = g_pack_index.find(key);
        if (it != g_pack_index.end()) {
            offset = it->second;
        } else {
            offset = g_pack_offset;
            fwrite(host.data(), 1, host.size(), g_snap_pack);
            g_pack_offset += host.size();
            g_pack_index[key] = offset;
        }
        fprintf(g_snaps, "{\"snap\":%ld,\"param\":%d,\"phase\":\"%s\",\"ptr\":%lu,\"bytes\":%lu,\"offset\":%lu,\"fnv\":\"%s\"}\n",
                id, sn.index, phase, (unsigned long)sn.ptr, (unsigned long)sn.bytes, (unsigned long)offset, key.c_str());
    }
    --t_depth;
}

// Snapshot mode, launch entry: choose the buffers and save their current bytes.
void snapshot_enter(Frame &frame, CUfunction f, void **params) {
    if (!g_snapshot.load() || !g_recording.load()) return;
    frame.snaps = snapshot_targets(f, params);
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        frame.snap = (long)g_snap_id++;
    }
    dump_snaps(frame.snap, frame.snaps, "in");
}

void record_unresolved(const char *symbol, const char *api) {
    std::lock_guard<std::mutex> lock(g_mutex);
    fprintf(g_launches, "{\"seq\":%lu,\"label\":\"%s\",\"api\":\"%s\",\"name\":\"%s\",\"unresolved\":true}\n",
            (unsigned long)g_seq++, escape(g_label.c_str()).c_str(), api, escape(symbol).c_str());
}

void CUPTIAPI callback(void *, CUpti_CallbackDomain domain, CUpti_CallbackId cbid, const void *cbdata) {
    if (t_depth > 0) return;
    if (domain == CUPTI_CB_DOMAIN_RESOURCE) {
        if (cbid != CUPTI_CBID_RESOURCE_MODULE_LOADED) return;
        auto *data = static_cast<const CUpti_ResourceData *>(cbdata);
        auto *mod = static_cast<const CUpti_ModuleResourceData *>(data->resourceDescriptor);
        std::lock_guard<std::mutex> lock(g_mutex);
        auto &m = g_images[mod->moduleId];
        m.image.assign(mod->pCubin, mod->pCubin + mod->cubinSize);
        m.written = false;
        bool orphan = t_stack.empty();
        fprintf(g_modules, "{\"loaded\":%u,\"bytes\":%zu,\"in\":%u,\"orphan\":%d}\n", mod->moduleId, mod->cubinSize,
                orphan ? 0u : t_stack.back().cbid, (int)orphan);
        if (g_all_modules) write_module(mod->moduleId);
        fflush(g_modules);
        if (!t_stack.empty()) t_stack.back().loaded.push_back(mod->moduleId);
        return;
    }
    auto *info = static_cast<const CUpti_CallbackData *>(cbdata);
    bool runtime = domain == CUPTI_CB_DOMAIN_RUNTIME_API;
    if (info->callbackSite == CUPTI_API_ENTER) {
        bool inside_runtime_launch = !t_stack.empty() && t_stack.back().runtime && t_stack.back().launch;
        bool inside_runtime_memop = !t_stack.empty() && t_stack.back().memop;
        t_stack.push_back({cbid, runtime, {}});
        if (runtime) {
            switch (cbid) {
            case CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy_v3020: case CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy_ptds_v7000: {
                t_stack.back().memop = true;
                if (!g_recording.load()) return;
                auto *p = static_cast<const cudaMemcpy_v3020_params *>(info->functionParams);
                record_memcpy(p->dst, p->src, p->count, (int)p->kind, nullptr, "cudaMemcpy");
                return;
            }
            case CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyAsync_v3020: case CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyAsync_ptsz_v7000: {
                t_stack.back().memop = true;
                if (!g_recording.load()) return;
                auto *p = static_cast<const cudaMemcpyAsync_v3020_params *>(info->functionParams);
                record_memcpy(p->dst, p->src, p->count, (int)p->kind, (CUstream)p->stream, "cudaMemcpyAsync");
                return;
            }
            case CUPTI_RUNTIME_TRACE_CBID_cudaMemsetAsync_v3020: case CUPTI_RUNTIME_TRACE_CBID_cudaMemsetAsync_ptsz_v7000: {
                t_stack.back().memop = true;
                if (!g_recording.load()) return;
                auto *p = static_cast<const cudaMemsetAsync_v3020_params *>(info->functionParams);
                record_memset(p->devPtr, (unsigned)p->value, p->count, 1, (CUstream)p->stream, "cudaMemsetAsync");
                return;
            }
            default:
                t_stack.back().launch = true;  // the remaining runtime callbacks we enable are launches
                return;
            }
        }
        if (!g_recording.load()) return;
        if (!inside_runtime_memop) {
            switch (cbid) {
            case CUPTI_DRIVER_TRACE_CBID_cuMemcpyHtoD_v2: {
                auto *p = static_cast<const cuMemcpyHtoD_v2_params *>(info->functionParams);
                record_memcpy((void *)p->dstDevice, p->srcHost, p->ByteCount, 1, nullptr, "cuMemcpyHtoD");
                return;
            }
            case CUPTI_DRIVER_TRACE_CBID_cuMemcpyHtoDAsync_v2: {
                auto *p = static_cast<const cuMemcpyHtoDAsync_v2_params *>(info->functionParams);
                record_memcpy((void *)p->dstDevice, p->srcHost, p->ByteCount, 1, p->hStream, "cuMemcpyHtoDAsync");
                return;
            }
            case CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoDAsync_v2: {
                auto *p = static_cast<const cuMemcpyDtoDAsync_v2_params *>(info->functionParams);
                record_memcpy((void *)p->dstDevice, (void *)p->srcDevice, p->ByteCount, 3, p->hStream, "cuMemcpyDtoDAsync");
                return;
            }
            case CUPTI_DRIVER_TRACE_CBID_cuMemcpyAsync: {
                auto *p = static_cast<const cuMemcpyAsync_params *>(info->functionParams);
                record_memcpy((void *)p->dst, (void *)p->src, p->ByteCount, 4, p->hStream, "cuMemcpyAsync");
                return;
            }
            case CUPTI_DRIVER_TRACE_CBID_cuMemsetD8Async: {
                auto *p = static_cast<const cuMemsetD8Async_params *>(info->functionParams);
                record_memset((void *)p->dstDevice, p->uc, p->N, 1, p->hStream, "cuMemsetD8Async");
                return;
            }
            case CUPTI_DRIVER_TRACE_CBID_cuMemsetD32Async: {
                auto *p = static_cast<const cuMemsetD32Async_params *>(info->functionParams);
                record_memset((void *)p->dstDevice, p->ui, p->N, 4, p->hStream, "cuMemsetD32Async");
                return;
            }
            default: break;
            }
        }
        bool recorded = true;
        long snap = -1;
        if (cbid == CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel || cbid == CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel_ptsz) {
            auto *p = static_cast<const cuLaunchKernel_params *>(info->functionParams);
            unsigned g[3] = {p->gridDimX, p->gridDimY, p->gridDimZ}, b[3] = {p->blockDimX, p->blockDimY, p->blockDimZ};
            snapshot_enter(t_stack.back(), p->f, p->kernelParams);
            snap = t_stack.back().snap;
            record_launch(p->f, g, b, p->sharedMemBytes, p->hStream, p->kernelParams, p->extra, nullptr, 0,
                          inside_runtime_launch ? "runtime>cuLaunchKernel" : info->functionName, snap);
        } else if (cbid == CUPTI_DRIVER_TRACE_CBID_cuLaunchCooperativeKernel ||
                   cbid == CUPTI_DRIVER_TRACE_CBID_cuLaunchCooperativeKernel_ptsz) {
            auto *p = static_cast<const cuLaunchCooperativeKernel_params *>(info->functionParams);
            unsigned g[3] = {p->gridDimX, p->gridDimY, p->gridDimZ}, b[3] = {p->blockDimX, p->blockDimY, p->blockDimZ};
            record_launch(p->f, g, b, p->sharedMemBytes, p->hStream, p->kernelParams, nullptr, nullptr, 0,
                          info->functionName);
        } else if (cbid == CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx || cbid == CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx_ptsz) {
            auto *p = static_cast<const cuLaunchKernelEx_params *>(info->functionParams);
            const CUlaunchConfig *c = p->config;
            unsigned g[3] = {c->gridDimX, c->gridDimY, c->gridDimZ}, b[3] = {c->blockDimX, c->blockDimY, c->blockDimZ};
            snapshot_enter(t_stack.back(), p->f, p->kernelParams);
            snap = t_stack.back().snap;
            record_launch(p->f, g, b, c->sharedMemBytes, c->hStream, p->kernelParams, p->extra, c->attrs, c->numAttrs,
                          inside_runtime_launch ? "runtime>cuLaunchKernelEx" : "cuLaunchKernelEx", snap);
        } else if (cbid == CUPTI_DRIVER_TRACE_CBID_cuGraphLaunch || cbid == CUPTI_DRIVER_TRACE_CBID_cuGraphLaunch_ptsz) {
            std::lock_guard<std::mutex> lock(g_mutex);
            fprintf(g_launches, "{\"seq\":%lu,\"label\":\"%s\",\"api\":\"cuGraphLaunch\"}\n", (unsigned long)g_seq++,
                    escape(g_label.c_str()).c_str());
        } else {
            recorded = false;
        }
        if (recorded && t_stack.size() >= 2 && t_stack[t_stack.size() - 2].launch) t_stack[t_stack.size() - 2].recorded = true;
        return;
    }
    // API exit: bind loaded modules to the handle the call produced, record encodes and allocations.
    if (t_stack.empty()) return;
    Frame frame = std::move(t_stack.back());
    t_stack.pop_back();
    if (frame.snap >= 0) dump_snaps(frame.snap, frame.snaps, "out");
    if (runtime) {
        if (!frame.launch || frame.recorded || !g_recording.load()) return;  // memops were recorded at entry
        GetFuncBySymbol get = runtime_get_func();
        cudaFunction_t f = nullptr;
        if (cbid == CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000 || cbid == CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_ptsz_v7000) {
            auto *p = static_cast<const cudaLaunchKernel_v7000_params *>(info->functionParams);
            ++t_depth; if (get) get(&f, p->func); --t_depth;
            unsigned g[3] = {p->gridDim.x, p->gridDim.y, p->gridDim.z}, b[3] = {p->blockDim.x, p->blockDim.y, p->blockDim.z};
            if (f) record_launch((CUfunction)f, g, b, (unsigned)p->sharedMem, (CUstream)p->stream, p->args, nullptr, nullptr, 0,
                                 "cudaLaunchKernel", frame.snap);
            else record_unresolved(info->symbolName, "cudaLaunchKernel");
        } else {
            auto *p = static_cast<const cudaLaunchKernelExC_v11060_params *>(info->functionParams);
            ++t_depth; if (get) get(&f, p->func); --t_depth;
            const cudaLaunchConfig_t *c = p->config;
            unsigned g[3] = {c->gridDim.x, c->gridDim.y, c->gridDim.z}, b[3] = {c->blockDim.x, c->blockDim.y, c->blockDim.z};
            if (f) record_launch((CUfunction)f, g, b, (unsigned)c->dynamicSmemBytes, (CUstream)c->stream, p->args, nullptr,
                                 reinterpret_cast<const CUlaunchAttribute *>(c->attrs), c->numAttrs, "cudaLaunchKernelExC",
                                 frame.snap);
            else record_unresolved(info->symbolName, "cudaLaunchKernelExC");
        }
        return;
    }
    const void *fp = info->functionParams;
    CUresult rc = info->functionReturnValue ? *static_cast<CUresult *>(info->functionReturnValue) : CUDA_SUCCESS;
    if (rc != CUDA_SUCCESS) return;
    std::lock_guard<std::mutex> lock(g_mutex);
    switch (cbid) {
    case CUPTI_DRIVER_TRACE_CBID_cuModuleLoad:
        bind((uintptr_t)*static_cast<const cuModuleLoad_params *>(fp)->module, frame.loaded, "module"); break;
    case CUPTI_DRIVER_TRACE_CBID_cuModuleLoadData:
        bind((uintptr_t)*static_cast<const cuModuleLoadData_params *>(fp)->module, frame.loaded, "module"); break;
    case CUPTI_DRIVER_TRACE_CBID_cuModuleLoadDataEx:
        bind((uintptr_t)*static_cast<const cuModuleLoadDataEx_params *>(fp)->module, frame.loaded, "module"); break;
    case CUPTI_DRIVER_TRACE_CBID_cuModuleLoadFatBinary:
        bind((uintptr_t)*static_cast<const cuModuleLoadFatBinary_params *>(fp)->module, frame.loaded, "module"); break;
    case CUPTI_DRIVER_TRACE_CBID_cuLibraryLoadData:
    case CUPTI_DRIVER_TRACE_CBID_cuLibraryLoadFromFile: {
        CUlibrary lib = cbid == CUPTI_DRIVER_TRACE_CBID_cuLibraryLoadData
            ? *static_cast<const cuLibraryLoadData_params *>(fp)->library
            : *static_cast<const cuLibraryLoadFromFile_params *>(fp)->library;
        bind((uintptr_t)lib, frame.loaded, "library");
        CUmodule m = nullptr;
        ++t_depth;
        bool ok = cuLibraryGetModule(&m, lib) == CUDA_SUCCESS;
        --t_depth;
        if (ok) bind((uintptr_t)m, frame.loaded, "module");
        break;
    }
    case CUPTI_DRIVER_TRACE_CBID_cuTensorMapEncodeTiled: {
        if (!g_recording.load()) break;
        auto *p = static_cast<const cuTensorMapEncodeTiled_params *>(fp);
        std::string s = "{\"kind\":\"tiled\",\"map\":\"";
        hex(s, p->tensorMap, sizeof(CUtensorMap));
        s += "\",\"dtype\":" + std::to_string((int)p->tensorDataType) + ",\"rank\":" + std::to_string(p->tensorRank) +
             ",\"address\":" + std::to_string((uintptr_t)p->globalAddress) + ",\"dims\":[";
        for (unsigned i = 0; i < p->tensorRank; ++i) s += (i ? "," : "") + std::to_string(p->globalDim[i]);
        s += "],\"strides\":[";
        for (unsigned i = 0; i + 1 < p->tensorRank; ++i) s += (i ? "," : "") + std::to_string(p->globalStrides[i]);
        s += "],\"box\":[";
        for (unsigned i = 0; i < p->tensorRank; ++i) s += (i ? "," : "") + std::to_string(p->boxDim[i]);
        s += "],\"element_strides\":[";
        for (unsigned i = 0; i < p->tensorRank; ++i) s += (i ? "," : "") + std::to_string(p->elementStrides[i]);
        s += "],\"interleave\":" + std::to_string((int)p->interleave) + ",\"swizzle\":" + std::to_string((int)p->swizzle) +
             ",\"l2\":" + std::to_string((int)p->l2Promotion) + ",\"oob\":" + std::to_string((int)p->oobFill) + "}";
        fprintf(g_maps, "%s\n", s.c_str());
        break;
    }
    case CUPTI_DRIVER_TRACE_CBID_cuMemAlloc_v2: {
        auto *p = static_cast<const cuMemAlloc_v2_params *>(fp);
        fprintf(g_allocs, "{\"api\":\"cuMemAlloc\",\"ptr\":%llu,\"bytes\":%zu}\n", (unsigned long long)*p->dptr, p->bytesize);
        g_alloc_ranges.push_back({(uint64_t)*p->dptr, (uint64_t)*p->dptr + p->bytesize});
        break;
    }
    case CUPTI_DRIVER_TRACE_CBID_cuMemAllocAsync: {
        auto *p = static_cast<const cuMemAllocAsync_params *>(fp);
        fprintf(g_allocs, "{\"api\":\"cuMemAllocAsync\",\"ptr\":%llu,\"bytes\":%zu}\n", (unsigned long long)*p->dptr, p->bytesize);
        g_alloc_ranges.push_back({(uint64_t)*p->dptr, (uint64_t)*p->dptr + p->bytesize});
        break;
    }
    case CUPTI_DRIVER_TRACE_CBID_cuMemMap: {
        auto *p = static_cast<const cuMemMap_params *>(fp);
        fprintf(g_allocs, "{\"api\":\"cuMemMap\",\"ptr\":%llu,\"bytes\":%zu}\n", (unsigned long long)p->ptr, p->size);
        g_alloc_ranges.push_back({(uint64_t)p->ptr, (uint64_t)p->ptr + p->size});
        break;
    }
    default:
        // A module loaded lazily during a launch belongs to the launched function's module.
        if (!frame.loaded.empty() && (cbid == CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel ||
                                      cbid == CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx ||
                                      cbid == CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel_ptsz ||
                                      cbid == CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx_ptsz)) {
            CUfunction f = cbid == CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx || cbid == CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx_ptsz
                ? static_cast<const cuLaunchKernelEx_params *>(fp)->f : static_cast<const cuLaunchKernel_params *>(fp)->f;
            CUmodule m = nullptr;
            CUfunction g = as_function(f);
            ++t_depth; cuFuncGetModule(&m, g); --t_depth;
            bind((uintptr_t)m, frame.loaded, "module(lazy)");
        }
    }
}

void check(CUptiResult r, const char *what) {
    if (r == CUPTI_SUCCESS) return;
    const char *s = nullptr;
    cuptiGetResultString(r, &s);
    fprintf(stderr, "[cupti_tape] %s: %s\n", what, s ? s : "?");
}

FILE *open_out(const char *name) {
    std::string path = g_dir + "/" + name;
    FILE *f = fopen(path.c_str(), "a");
    if (!f) { fprintf(stderr, "[cupti_tape] cannot open %s\n", path.c_str()); abort(); }
    return f;
}

}  // namespace

extern "C" {

__attribute__((visibility("default"))) int InitializeInjection(void) {
    const char *dir = getenv("TAPE_DIR");
    g_dir = dir ? dir : "/tmp/cupti_tape";
    g_all_modules = getenv("TAPE_ALL_MODULES") && getenv("TAPE_ALL_MODULES")[0] == '1';
    g_dir += "/pid" + std::to_string(getpid());
    mkdir(g_dir.c_str(), 0755);
    mkdir((g_dir + "/modules").c_str(), 0755);
    g_launches = open_out("launches.jsonl");
    g_modules = open_out("modules.jsonl");
    g_maps = open_out("tensormaps.jsonl");
    g_allocs = open_out("allocations.jsonl");
    mkdir((g_dir + "/snapshots").c_str(), 0755);
    g_snaps = open_out("snapshots.jsonl");
    g_snap_pack = open_out("snapshots.bin");
    if (const char *m = getenv("TAPE_SNAPSHOT_MAX_SEGMENT")) g_snap_max_segment = strtoull(m, nullptr, 10);
    if (const char *c = getenv("TAPE_CENSUS")) g_census = atoi(c);
    check(cuptiSubscribe(&g_subscriber, callback, nullptr), "cuptiSubscribe");
    check(cuptiEnableDomain(1, g_subscriber, CUPTI_CB_DOMAIN_RESOURCE), "enable resource");
    const CUpti_CallbackId driver[] = {
        CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel, CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel_ptsz,
        CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx, CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx_ptsz,
        CUPTI_DRIVER_TRACE_CBID_cuLaunchCooperativeKernel, CUPTI_DRIVER_TRACE_CBID_cuLaunchCooperativeKernel_ptsz,
        CUPTI_DRIVER_TRACE_CBID_cuGraphLaunch, CUPTI_DRIVER_TRACE_CBID_cuGraphLaunch_ptsz,
        CUPTI_DRIVER_TRACE_CBID_cuModuleLoad, CUPTI_DRIVER_TRACE_CBID_cuModuleLoadData,
        CUPTI_DRIVER_TRACE_CBID_cuModuleLoadDataEx, CUPTI_DRIVER_TRACE_CBID_cuModuleLoadFatBinary,
        CUPTI_DRIVER_TRACE_CBID_cuLibraryLoadData, CUPTI_DRIVER_TRACE_CBID_cuLibraryLoadFromFile,
        CUPTI_DRIVER_TRACE_CBID_cuTensorMapEncodeTiled, CUPTI_DRIVER_TRACE_CBID_cuMemAlloc_v2,
        CUPTI_DRIVER_TRACE_CBID_cuMemAllocAsync, CUPTI_DRIVER_TRACE_CBID_cuMemMap,
        CUPTI_DRIVER_TRACE_CBID_cuMemcpyHtoD_v2, CUPTI_DRIVER_TRACE_CBID_cuMemcpyHtoDAsync_v2,
        CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoDAsync_v2, CUPTI_DRIVER_TRACE_CBID_cuMemcpyAsync,
        CUPTI_DRIVER_TRACE_CBID_cuMemsetD8Async, CUPTI_DRIVER_TRACE_CBID_cuMemsetD32Async,
    };
    for (CUpti_CallbackId id : driver) check(cuptiEnableCallback(1, g_subscriber, CUPTI_CB_DOMAIN_DRIVER_API, id), "enable driver");
    const CUpti_CallbackId runtime[] = {
        CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000, CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_ptsz_v7000,
        CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernelExC_v11060, CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernelExC_ptsz_v11060,
        CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy_v3020, CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy_ptds_v7000,
        CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyAsync_v3020, CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyAsync_ptsz_v7000,
        CUPTI_RUNTIME_TRACE_CBID_cudaMemsetAsync_v3020, CUPTI_RUNTIME_TRACE_CBID_cudaMemsetAsync_ptsz_v7000,
    };
    for (CUpti_CallbackId id : runtime) check(cuptiEnableCallback(1, g_subscriber, CUPTI_CB_DOMAIN_RUNTIME_API, id), "enable runtime");
    fprintf(stderr, "[cupti_tape] recording into %s\n", g_dir.c_str());
    return 1;
}

__attribute__((visibility("default"))) void tape_start(const char *label) {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_label = label ? label : "";
    g_recording.store(true);
}

__attribute__((visibility("default"))) void tape_stop(void) {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_recording.store(false);
    fflush(g_launches); fflush(g_maps); fflush(g_allocs); fflush(g_modules); fflush(g_snaps); fflush(g_snap_pack);
}

__attribute__((visibility("default"))) const char *tape_dir(void) { return g_dir.c_str(); }

// Snapshot mode: `excluded` holds (begin, bytes) pairs never saved (weights); cap bounds the bytes
// saved per buffer (0 keeps the current cap); on enables or disables saving.
__attribute__((visibility("default"))) void tape_set_snapshot(const uint64_t *excluded, size_t n, uint64_t cap, int on) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (excluded) {
        g_excluded.clear();
        for (size_t i = 0; i < n; ++i) g_excluded.push_back({excluded[2 * i], excluded[2 * i] + excluded[2 * i + 1]});
        std::sort(g_excluded.begin(), g_excluded.end());
    }
    if (cap) g_snap_cap = cap;
    g_snapshot.store(on != 0);
    fflush(g_snaps); fflush(g_snap_pack);
}

}  // extern "C"
