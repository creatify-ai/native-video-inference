// Block-sparse attention through the exact CuTe DSL kernel.
//
// The kernel is an optional ahead-of-time CuTe DSL export (FlashInfer 0.7.0's sm100 blk64 CuTe
// kernel structure with FlashInfer 0.6.15's blk64 arithmetic): its output is bit-identical to the
// served kernel behind h3_bsa_forward. An export provides one object and header per scheduler
// variant, libcute_dsl_runtime.so and export.json. The public build always compiles this file
// with H3_BSA_EXACT_UNAVAILABLE, so it is an "unavailable" stub and the FlashInfer SubBlock
// kernel stays the only sparse path.
//
// Contract (device pointers, one stream): q, k, v and out are bf16 [tokens, heads, 128]; k and v
// need no rows past `tokens` (the kernel's TMA reads stop at the true extent and zero-fill the last
// block). index is [heads, ceil(tokens / 64), kept] int32 block ids, block_sizes [ceil(tokens / 64)]
// int32. Same routing semantics as h3_bsa_forward; the audio rows' dense correction stays outside.
// h3_bsa_exact_accepts says whether a call is inside the exported kernel's validated domain; the
// bridge keeps the served kernel for every other call.
#include <cuda_runtime.h>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>

#ifndef H3_BSA_EXACT_UNAVAILABLE
#include "h3_bsa_exact_clc.h"
#include "h3_bsa_exact_tile.h"
#ifdef H3_BSA_EXACT_P23
// Export variant 'exact-p23': the same arithmetic with the softmax/correction release decoupled
// and each K/V stage completing in two halves. It measured faster up to 150k rows and slower
// beyond, hence the row gate.
#include "h3_bsa_exact_clc_p23.h"
#include "h3_bsa_exact_tile_p23.h"
#endif

namespace {
h3_bsa_exact_clc_Kernel_Module_t clc_module;
h3_bsa_exact_tile_Kernel_Module_t tile_module;
#ifdef H3_BSA_EXACT_P23
h3_bsa_exact_clc_p23_Kernel_Module_t clc_p23_module;
h3_bsa_exact_tile_p23_Kernel_Module_t tile_p23_module;
constexpr int p23_max_tokens = 150000;
// H3_BSA_EXACT_P23=0 keeps every call on the export-003 kernels (same-binary A/B).
bool p23_enabled() {
    static const bool enabled = [] { const char *v = std::getenv("H3_BSA_EXACT_P23"); return !(v && std::strcmp(v, "0") == 0); }();
    return enabled;
}
#endif
std::once_flag init_once;
cudaError_t init_status = cudaSuccess;
std::mutex load_mutex;
std::atomic<uint64_t> loaded_devices{0};

template <typename Init>
cudaError_t init_library(Init init, cudaLibrary_t *library) {
    cudaError_t ret = cudaSuccess;
    struct { cudaLibrary_t **library; cudaError_t *ret; } args = {&library, &ret};
    init(reinterpret_cast<void **>(&args));
    return ret;
}

template <typename Load>
cudaError_t load_library(Load load, cudaLibrary_t *library, int32_t device) {
    cudaError_t ret = cudaSuccess;
    struct { cudaLibrary_t **library; int32_t *device; cudaError_t *ret; } args = {&library, &device, &ret};
    load(reinterpret_cast<void **>(&args));
    return ret;
}

// The generated Module_Load helpers load onto every visible device; a rank must touch only its own.
cudaError_t ensure_loaded() {
    std::call_once(init_once, [] {
        init_status = init_library(_mlir_h3_bsa_exact_clc_cuda_init, &clc_module.module);
        if (init_status == cudaSuccess)
            init_status = init_library(_mlir_h3_bsa_exact_tile_cuda_init, &tile_module.module);
#ifdef H3_BSA_EXACT_P23
        if (init_status == cudaSuccess)
            init_status = init_library(_mlir_h3_bsa_exact_clc_p23_cuda_init, &clc_p23_module.module);
        if (init_status == cudaSuccess)
            init_status = init_library(_mlir_h3_bsa_exact_tile_p23_cuda_init, &tile_p23_module.module);
#endif
    });
    if (init_status != cudaSuccess) return init_status;
    int device = 0;
    cudaError_t status = cudaGetDevice(&device);
    if (status != cudaSuccess) return status;
    if (device < 0 || device >= 64) return cudaErrorInvalidDevice;
    const uint64_t bit = uint64_t(1) << device;
    if (loaded_devices.load(std::memory_order_acquire) & bit) return cudaSuccess;
    std::lock_guard<std::mutex> lock(load_mutex);
    if (loaded_devices.load(std::memory_order_relaxed) & bit) return cudaSuccess;
    status = load_library(_mlir_h3_bsa_exact_clc_cuda_load_to_device, &clc_module.module, device);
    if (status == cudaSuccess)
        status = load_library(_mlir_h3_bsa_exact_tile_cuda_load_to_device, &tile_module.module, device);
#ifdef H3_BSA_EXACT_P23
    if (status == cudaSuccess)
        status = load_library(_mlir_h3_bsa_exact_clc_p23_cuda_load_to_device, &clc_p23_module.module, device);
    if (status == cudaSuccess)
        status = load_library(_mlir_h3_bsa_exact_tile_p23_cuda_load_to_device, &tile_p23_module.module, device);
#endif
    if (status == cudaSuccess) loaded_devices.fetch_or(bit, std::memory_order_release);
    return status;
}

// The wrapper's scheduler policy (dispatch_helpers.choose_sm100_blk64_use_clc, uniform lists,
// one batch); the choice changes tile order only, never arithmetic.
bool use_clc(int tokens, int heads, int kept) {
    const int64_t row_tiles = (tokens + 63) / 64, tiles = row_tiles * heads;
    if (tiles >= 8192 && kept >= 256) return true;
    if (heads == 1) return false;
    return row_tiles >= 128 && tiles >= 512 && kept <= (heads == 2 ? 64 : 128);
}

template <typename T>
void natural(T &tensor, const void *data, int tokens, int heads) {
    const int64_t row = int64_t(heads) * 128;
    tensor = T{const_cast<void *>(data), {1, heads, tokens, 128}, {int64_t(tokens) * row, 128, row}};
}

template <typename M, typename Q, typename K, typename V, typename O, typename I, typename S, typename F>
int launch(M *module, const void *q, const void *k, const void *v, const int *index, const int *block_sizes,
           void *out, int tokens, int heads, int kept, float scale, cudaStream_t stream, F wrapper) {
    Q mQ; K mK; V mV; O mO;
    natural(mQ, q, tokens, heads);
    natural(mK, k, tokens, heads);
    natural(mV, v, tokens, heads);
    natural(mO, out, tokens, heads);
    const int32_t groups = (tokens + 63) / 64;
    I mIndex{const_cast<int *>(index), {1, heads, groups, kept},
             {int64_t(heads) * groups * kept, int64_t(groups) * kept, kept}};
    S mSizes{const_cast<int *>(block_sizes), {groups}};
    return wrapper(module, &mQ, &mK, &mV, &mO, scale, &mIndex, &mSizes, kept, stream);
}
}  // namespace

extern "C" int h3_bsa_exact_available() { return 1; }

// Loads both kernels onto the current device ahead of the first call.
extern "C" int h3_bsa_exact_prepare() { return int(ensure_loaded()); }

// The validated domain: the router's lengths and list size (dynamic_router.rs Plan), tensors below
// 2^31 elements (every runtime head mode), H3's scale, and the 16-byte alignment the export assumes
// for every tensor. The served kernel rounds
// softmax_scale * log2(e) once from double while this one multiplies in float; they agree for
// 128^-0.5, and any scale where they differ is refused.
extern "C" int h3_bsa_exact_accepts(const void *q, const void *k, const void *v, const int *index,
                                    const int *block_sizes, const void *out, int tokens, int heads, int kept,
                                    float scale) {
    const int64_t groups = (int64_t(tokens) + 63) / 64;
    const int64_t router_kept = std::min<int64_t>(groups, ((groups + 3) / 4 + 7) / 8 * 8);
    const auto aligned = [](const void *p) { return p && reinterpret_cast<uintptr_t>(p) % 16 == 0; };
    return tokens >= 4096 && tokens <= 524288 && heads >= 1 && int64_t(tokens) * heads * 128 < (int64_t(1) << 31) &&
           kept == router_kept &&
           aligned(q) && aligned(k) && aligned(v) && aligned(index) && aligned(block_sizes) && aligned(out) &&
           std::isfinite(scale) && scale > 0.0f &&
           float(double(scale) * 1.4426950408889634) == scale * float(1.4426950408889634);
}

extern "C" int h3_bsa_exact_forward(const void *q, const void *k, const void *v, const int *index,
                                    const int *block_sizes, void *out, int tokens, int heads, int kept,
                                    float scale, cudaStream_t stream) {
    if (!h3_bsa_exact_accepts(q, k, v, index, block_sizes, out, tokens, heads, kept, scale))
        return cudaErrorInvalidValue;
    cudaError_t status = ensure_loaded();
    if (status != cudaSuccess) return int(status);
#ifdef H3_BSA_EXACT_P23
    if (tokens <= p23_max_tokens && p23_enabled()) {
        const int ret = use_clc(tokens, heads, kept)
            ? launch<h3_bsa_exact_clc_p23_Kernel_Module_t, h3_bsa_exact_clc_p23_Tensor_mQ_t,
                     h3_bsa_exact_clc_p23_Tensor_mK_t, h3_bsa_exact_clc_p23_Tensor_mV_t,
                     h3_bsa_exact_clc_p23_Tensor_mO_t, h3_bsa_exact_clc_p23_Tensor_mBlockIndex_t,
                     h3_bsa_exact_clc_p23_Tensor_mBlockSizes_t>(
                  &clc_p23_module, q, k, v, index, block_sizes, out, tokens, heads, kept, scale, stream,
                  cute_dsl_h3_bsa_exact_clc_p23_wrapper)
            : launch<h3_bsa_exact_tile_p23_Kernel_Module_t, h3_bsa_exact_tile_p23_Tensor_mQ_t,
                     h3_bsa_exact_tile_p23_Tensor_mK_t, h3_bsa_exact_tile_p23_Tensor_mV_t,
                     h3_bsa_exact_tile_p23_Tensor_mO_t, h3_bsa_exact_tile_p23_Tensor_mBlockIndex_t,
                     h3_bsa_exact_tile_p23_Tensor_mBlockSizes_t>(
                  &tile_p23_module, q, k, v, index, block_sizes, out, tokens, heads, kept, scale, stream,
                  cute_dsl_h3_bsa_exact_tile_p23_wrapper);
        return ret ? ret : int(cudaGetLastError());
    }
#endif
    const int ret = use_clc(tokens, heads, kept)
        ? launch<h3_bsa_exact_clc_Kernel_Module_t, h3_bsa_exact_clc_Tensor_mQ_t, h3_bsa_exact_clc_Tensor_mK_t,
                 h3_bsa_exact_clc_Tensor_mV_t, h3_bsa_exact_clc_Tensor_mO_t,
                 h3_bsa_exact_clc_Tensor_mBlockIndex_t, h3_bsa_exact_clc_Tensor_mBlockSizes_t>(
              &clc_module, q, k, v, index, block_sizes, out, tokens, heads, kept, scale, stream,
              cute_dsl_h3_bsa_exact_clc_wrapper)
        : launch<h3_bsa_exact_tile_Kernel_Module_t, h3_bsa_exact_tile_Tensor_mQ_t, h3_bsa_exact_tile_Tensor_mK_t,
                 h3_bsa_exact_tile_Tensor_mV_t, h3_bsa_exact_tile_Tensor_mO_t,
                 h3_bsa_exact_tile_Tensor_mBlockIndex_t, h3_bsa_exact_tile_Tensor_mBlockSizes_t>(
              &tile_module, q, k, v, index, block_sizes, out, tokens, heads, kept, scale, stream,
              cute_dsl_h3_bsa_exact_tile_wrapper);
    return ret ? ret : int(cudaGetLastError());
}
#else
extern "C" int h3_bsa_exact_available() { return 0; }
extern "C" int h3_bsa_exact_prepare() { return cudaErrorNotSupported; }
extern "C" int h3_bsa_exact_accepts(const void *, const void *, const void *, const int *, const int *, const void *,
                                    int, int, int, float) {
    return 0;
}
extern "C" int h3_bsa_exact_forward(const void *, const void *, const void *, const int *, const int *, void *,
                                    int, int, int, float, cudaStream_t) {
    return cudaErrorNotSupported;
}
#endif
