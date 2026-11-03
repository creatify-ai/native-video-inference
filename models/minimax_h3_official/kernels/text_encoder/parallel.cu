// Sequence-parallel Qwen text decoder helpers. Each rank runs the decoder's
// projections on its own contiguous token rows and exchanges q/k/v so that
// attention runs on whole sequences for a head group. Only two kinds of work:
//  * GEMMs restricted to cuBLASLt algorithms without split-K. Their output
//    elements keep the accepted single-GPU reduction order; the accepted call
//    selects no split-K either above 768 rows (checked bitwise on B200). The
//    heuristic's top-1 is used, except that a tiny-tile top-1 on a large shape
//    yields to a faster candidate whose output is bitwise identical to it.
//  * Exact BF16 relayouts around attention: pack3 into the staging peers read,
//    and direct pulls from peers' memory.
#include <cublasLt.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <map>
#include <memory>
#include <new>
#include <string>
#include <tuple>

namespace {
thread_local std::string last_error;
int fail(const std::string& message) { last_error = message; return -1; }
#define LT(call) do { auto status_ = (call); if (status_ != CUBLAS_STATUS_SUCCESS) return fail(std::string(#call) + ": status " + std::to_string(int(status_))); } while (0)
#define CU(call) do { auto status_ = (call); if (status_ != cudaSuccess) return fail(std::string(#call) + ": " + cudaGetErrorString(status_)); } while (0)

unsigned blocks(int64_t words);
uint32_t alignment(const void* pointer) {
    uint32_t result = 256;
    while (reinterpret_cast<uintptr_t>(pointer) % result) result /= 2;
    return result;
}

struct Plan {
    cublasLtMatmulDesc_t op = nullptr;
    cublasLtMatrixLayout_t a = nullptr, b = nullptr, c = nullptr;
    cublasLtMatmulAlgo_t algo{};
    cublasLtMatmulHeuristicResult_t candidates[16]{};
    int count = 0;
    bool replaced = false;
    ~Plan() {
        if (c) cublasLtMatrixLayoutDestroy(c);
        if (b) cublasLtMatrixLayoutDestroy(b);
        if (a) cublasLtMatrixLayoutDestroy(a);
        if (op) cublasLtMatmulDescDestroy(op);
    }
};
struct Gemm {
    cublasLtHandle_t lt = nullptr;
    void* workspace = nullptr;
    size_t workspace_bytes = 0;
    void* candidate = nullptr;         // one candidate output, grown on demand
    size_t candidate_bytes = 0;
    unsigned long long* mismatches = nullptr;
    cudaEvent_t start = nullptr, stop = nullptr;
    size_t replaced = 0;               // plans that left a tiny-tile top-1
    // rows, n, k, bias and the joint pointer alignment select one fixed algorithm.
    std::map<std::tuple<int, int, int, bool, uint32_t>, std::unique_ptr<Plan>> plans;
    ~Gemm() {
        plans.clear();
        if (workspace) cudaFree(workspace);
        if (candidate) cudaFree(candidate);
        if (mismatches) cudaFree(mismatches);
        if (start) cudaEventDestroy(start);
        if (stop) cudaEventDestroy(stop);
        if (lt) cublasLtDestroy(lt);
    }
};
__global__ void count_mismatch(const uint4* a, const uint4* b, size_t words, unsigned long long* out) {
    unsigned long long local = 0;
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < words; i += size_t(gridDim.x) * blockDim.x)
        local += a[i].x != b[i].x || a[i].y != b[i].y || a[i].z != b[i].z || a[i].w != b[i].w;
    if (local) atomicAdd(out, local);
}

int build(Gemm* g, Plan* plan, int rows, int n, int k, const void* bias, uint32_t align) {
    LT(cublasLtMatmulDescCreate(&plan->op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    const cublasOperation_t transa = CUBLAS_OP_T, transb = CUBLAS_OP_N;
    LT(cublasLtMatmulDescSetAttribute(plan->op, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
    LT(cublasLtMatmulDescSetAttribute(plan->op, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));
    if (bias) {
        // As the accepted Linear-with-bias call: bias added in the epilogue before rounding.
        const cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;
        LT(cublasLtMatmulDescSetAttribute(plan->op, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));
        LT(cublasLtMatmulDescSetAttribute(plan->op, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias)));
    }
    // Row-major X[rows,k] times W[n,k]^T as column-major W^T(k,n) x X(k,rows).
    LT(cublasLtMatrixLayoutCreate(&plan->a, CUDA_R_16BF, k, n, k));
    LT(cublasLtMatrixLayoutCreate(&plan->b, CUDA_R_16BF, k, rows, k));
    LT(cublasLtMatrixLayoutCreate(&plan->c, CUDA_R_16BF, n, rows, n));
    cublasLtMatmulPreference_t preference = nullptr;
    LT(cublasLtMatmulPreferenceCreate(&preference));
    std::unique_ptr<std::remove_pointer_t<cublasLtMatmulPreference_t>, decltype(&cublasLtMatmulPreferenceDestroy)>
        owned(preference, &cublasLtMatmulPreferenceDestroy);
    LT(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                            &g->workspace_bytes, sizeof(g->workspace_bytes)));
    // An empty mask admits only CUBLASLT_REDUCTION_SCHEME_NONE: no split-K.
    const uint32_t mask = 0;
    LT(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK, &mask, sizeof(mask)));
    for (auto attribute : {CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES,
                           CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES})
        LT(cublasLtMatmulPreferenceSetAttribute(preference, attribute, &align, sizeof(align)));
    cublasLtMatmulHeuristicResult_t results[16];
    int count = 0;
    LT(cublasLtMatmulAlgoGetHeuristic(g->lt, plan->op, plan->a, plan->b, plan->c, plan->c, preference, 16, results, &count));
    if (count < 1 || results[0].state != CUBLAS_STATUS_SUCCESS) return fail("no split-K-free cuBLASLt algorithm for this shape");
    plan->algo = results[0].algo;
    plan->count = count;
    std::copy(results, results + count, plan->candidates);
    return 0;
}

uint32_t tile_of(const cublasLtMatmulAlgo_t& algo) {
    uint32_t tile = 0; size_t written = 0;
    cublasLtMatmulAlgoConfigGetAttribute(&algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &tile, sizeof(tile), &written);
    return tile;
}
// On a few large shapes (for example down_proj at 128-303 rows) the heuristic's
// top-1 without split-K is a 16x16/32x32 tile, up to 17x slower than the
// alternatives. Once per such shape, on the real operands, accept the fastest
// other candidate whose output is bitwise identical to top-1; top-1 itself is
// the algorithm verified against the accepted single-GPU result.
int prefer_identical(Gemm* g, Plan& p, const void* x, const void* w, void* y, int rows, int n, cudaStream_t stream) {
    const size_t bytes = size_t(rows) * n * 2;
    if (g->candidate_bytes < bytes) {
        if (g->candidate) cudaFree(g->candidate);
        g->candidate = nullptr; g->candidate_bytes = 0;
        CU(cudaMalloc(&g->candidate, bytes)); g->candidate_bytes = bytes;
    }
    if (!g->mismatches) CU(cudaMalloc(&g->mismatches, sizeof(unsigned long long)));
    if (!g->start) { CU(cudaEventCreate(&g->start)); CU(cudaEventCreate(&g->stop)); }
    const float alpha = 1.f, beta = 0.f;
    auto run = [&](const cublasLtMatmulAlgo_t& algo, void* out) {
        return cublasLtMatmul(g->lt, p.op, &alpha, w, p.a, x, p.b, &beta, out, p.c, out, p.c, &algo,
                              g->workspace, g->workspace_bytes, stream) == CUBLAS_STATUS_SUCCESS;
    };
    auto timed = [&](const cublasLtMatmulAlgo_t& algo, void* out) -> float {
        float best = 1e30f;
        for (int repeat = 0; repeat < 2; ++repeat) {
            if (cudaEventRecord(g->start, stream) != cudaSuccess || !run(algo, out) ||
                cudaEventRecord(g->stop, stream) != cudaSuccess || cudaEventSynchronize(g->stop) != cudaSuccess) return -1.f;
            float ms = 0; cudaEventElapsedTime(&ms, g->start, g->stop); best = std::min(best, ms);
        }
        return best;
    };
    float chosen_ms = timed(p.algo, y);
    if (chosen_ms < 0) return fail("timing the top-1 GEMM algorithm");
    const size_t words = bytes / 16;
    for (int i = 1; i < p.count; ++i) {
        if (p.candidates[i].state != CUBLAS_STATUS_SUCCESS || tile_of(p.candidates[i].algo) < CUBLASLT_MATMUL_TILE_64x64) continue;
        if (!run(p.candidates[i].algo, g->candidate)) continue;
        CU(cudaMemsetAsync(g->mismatches, 0, sizeof(unsigned long long), stream));
        count_mismatch<<<blocks(int64_t(words)), 256, 0, stream>>>(static_cast<const uint4*>(y), static_cast<const uint4*>(g->candidate), words, g->mismatches);
        unsigned long long host = 1;
        CU(cudaMemcpyAsync(&host, g->mismatches, sizeof(host), cudaMemcpyDeviceToHost, stream));
        CU(cudaStreamSynchronize(stream));
        if (host != 0) continue;
        const float ms = timed(p.candidates[i].algo, g->candidate);
        if (ms > 0 && ms < chosen_ms) { chosen_ms = ms; p.algo = p.candidates[i].algo; p.replaced = true; }
    }
    g->replaced += p.replaced;
    return 0;
}

// out[p][r] = a's, b's and c's p-th column groups of row r (widths/strides in 16-byte words).
__global__ void pack3(const uint4* a, int64_t sa, const uint4* b, int64_t sb, const uint4* c, int64_t sc,
                      int wa, int wb, int wc, uint4* out, int64_t rows, int world) {
    const int width = wa + wb + wc;
    const int64_t total = int64_t(world) * rows * width;
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < total; i += int64_t(gridDim.x) * blockDim.x) {
        const int64_t peer = i / (rows * width), within = i % (rows * width), row = within / width;
        const int col = int(within % width);
        out[i] = col < wa ? a[row * sa + peer * wa + col]
               : col < wa + wb ? b[row * sb + peer * wb + col - wa]
               : c[row * sc + peer * wc + col - wa - wb];
    }
}
// Direct exchange: after a stream-ordered barrier, read peers' device memory
// (CUDA IPC views) straight into the layouts attention and the next projection
// consume, as the DiT's direct exchange. Every BF16 bit is copied unchanged.
struct Peers { const uint4* p[8]; };
struct Starts { int64_t s[9]; };
// Peer p's staging is pack3's [world][rows_p][wa+wb+wc]; its block for `rank`
// holds global rows starts[p]..starts[p+1], whose columns split into a, b and c.
// Consecutive rows alternate peers so every link streams at once.
__global__ void pull3(Peers peers, Starts st, int rank, int world, int wa, int wb, int wc,
                      uint4* a, int64_t pa, uint4* b, int64_t pb, uint4* c, int64_t pc, int64_t max_rows) {
    const int width = wa + wb + wc;
    const int64_t total = max_rows * world * width;
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < total; i += int64_t(gridDim.x) * blockDim.x) {
        const int64_t j = i / width, local = j / world;
        const int col = int(i % width), p = int(j % world);
        const int64_t count = st.s[p + 1] - st.s[p];
        if (local >= count) continue;
        const int64_t row = st.s[p] + local;
        const uint4 value = peers.p[p][(int64_t(rank) * count + local) * width + col];
        if (col < wa) a[row * pa + col] = value;
        else if (col < wa + wb) b[row * pb + col - wa] = value;
        else c[row * pc + col - wa - wb] = value;
    }
}
// out[r][p * w + j] = peer p's out[(start + r) * pitch + p * step + j]: this rank's
// rows of every head group, in the merged column order.
__global__ void pull_rows(Peers peers, int world, int64_t start, int64_t rows, int w, int64_t pitch, int64_t step, uint4* out) {
    const int64_t width = int64_t(world) * w, total = rows * width;
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < total; i += int64_t(gridDim.x) * blockDim.x) {
        const int64_t r = i / width;
        const int col = int(i % width), p = col / w;
        out[i] = peers.p[p][(start + r) * pitch + p * step + col % w];
    }
}
unsigned blocks(int64_t words) { int64_t b = (words + 255) / 256; return unsigned(b < 1 ? 1 : b > 16384 ? 16384 : b); }
bool valid_world(int world) { return world == 1 || world == 2 || world == 4 || world == 8; }
bool aligned16(std::initializer_list<const void*> pointers) {
    for (const void* p : pointers) if (!p || reinterpret_cast<uintptr_t>(p) % 16) return false;
    return true;
}
}  // namespace

extern "C" const char* q3_parallel_error() { return last_error.c_str(); }

extern "C" int q3_gemm_create(size_t workspace_bytes, void** out) {
    if (!out) return fail("null GEMM output");
    auto g = new (std::nothrow) Gemm;
    if (!g) return fail("GEMM context allocation");
    if (cublasLtCreate(&g->lt) != CUBLAS_STATUS_SUCCESS) { delete g; return fail("cublasLtCreate"); }
    if (cudaMalloc(&g->workspace, workspace_bytes) != cudaSuccess) { delete g; return fail("GEMM workspace allocation"); }
    g->workspace_bytes = workspace_bytes;
    *out = g;
    return 0;
}
extern "C" void q3_gemm_destroy(void* handle) { delete static_cast<Gemm*>(handle); }
extern "C" size_t q3_gemm_plan_count(void* handle) { return handle ? static_cast<Gemm*>(handle)->plans.size() : 0; }
extern "C" size_t q3_gemm_replaced_count(void* handle) { return handle ? static_cast<Gemm*>(handle)->replaced : 0; }
extern "C" int q3_gemm_clear(void* handle) {
    if (!handle) return fail("null GEMM context");
    static_cast<Gemm*>(handle)->plans.clear();
    return 0;
}
// Y[rows,n] = X[rows,k] W[n,k]^T (+ bias[n]), BF16 in/out, FP32 accumulation.
extern "C" int q3_gemm_forward_bias(void* handle, const void* x, const void* w, const void* bias, void* y,
                                    int rows, int n, int k, void* stream) {
    auto g = static_cast<Gemm*>(handle);
    if (!g || !x || !w || !y || rows < 1 || n < 1 || k < 1) return fail("invalid GEMM arguments");
    uint32_t align = std::min(std::min(alignment(x), alignment(w)), alignment(y));
    if (bias) align = std::min(align, alignment(bias));
    auto key = std::make_tuple(rows, n, k, bias != nullptr, align);
    auto found = g->plans.find(key);
    if (found == g->plans.end()) {
        auto plan = std::make_unique<Plan>();
        if (build(g, plan.get(), rows, n, k, bias, align)) return -1;
        found = g->plans.emplace(key, std::move(plan)).first;
        Plan& p = *found->second;
        if (tile_of(p.algo) < CUBLASLT_MATMUL_TILE_64x64 && double(n) * k >= double(1 << 24) && p.count > 1 && (size_t(rows) * n * 2) % 16 == 0 &&
            prefer_identical(g, p, x, w, y, rows, n, static_cast<cudaStream_t>(stream))) return -1;
    }
    const float alpha = 1.f, beta = 0.f;
    Plan& p = *found->second;
    if (bias) LT(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias)));
    LT(cublasLtMatmul(g->lt, p.op, &alpha, w, p.a, x, p.b, &beta, y, p.c, y, p.c, &p.algo,
                      g->workspace, g->workspace_bytes, static_cast<cudaStream_t>(stream)));
    return 0;
}

// Element widths/strides; every width, stride and pointer must be 16-byte aligned.
extern "C" int q3_cp_pack3(const void* a, int64_t sa, const void* b, int64_t sb, const void* c, int64_t sc,
                           int wa, int wb, int wc, void* out, int64_t rows, int world, void* stream) {
    if (!valid_world(world) || rows < 1 || !aligned16({a, b, c, out}) || (sa | sb | sc | wa | wb | wc) % 8)
        return fail("invalid grouped pack");
    pack3<<<blocks(int64_t(world) * rows * (wa + wb + wc) / 8), 256, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const uint4*>(a), sa / 8, static_cast<const uint4*>(b), sb / 8, static_cast<const uint4*>(c), sc / 8,
        wa / 8, wb / 8, wc / 8, static_cast<uint4*>(out), rows, world);
    return cudaGetLastError() == cudaSuccess ? 0 : fail("grouped pack launch");
}

// Element widths, pitches and steps; each must be a multiple of 8 (16 bytes).
namespace {
bool load_peers(const void* const* views, int world, Peers& peers) {
    if (!views || world < 2 || !valid_world(world)) return false;
    for (int p = 0; p < world; ++p) {
        if (!views[p] || reinterpret_cast<uintptr_t>(views[p]) % 16) return false;
        peers.p[p] = static_cast<const uint4*>(views[p]);
    }
    return true;
}
}  // namespace
extern "C" int q3_cp_pull3(const void* const* views, const int64_t* starts, int world, int rank, int wa, int wb, int wc,
                           void* a, int64_t pa, void* b, int64_t pb, void* c, int64_t pc, void* stream) {
    Peers peers{}; Starts st{};
    if (!load_peers(views, world, peers) || !starts || starts[0] != 0 || rank < 0 || rank >= world || wa < 8 || wb < 8 || wc < 8 ||
        (wa | wb | wc) % 8 || (pa | pb | pc) % 8 || !aligned16({a, b, c})) return fail("invalid direct pull");
    int64_t max_rows = 0;
    for (int p = 0; p <= world; ++p) st.s[p] = starts[p];
    for (int p = 0; p < world; ++p) {
        if (st.s[p + 1] < st.s[p]) return fail("invalid direct pull row starts");
        max_rows = std::max(max_rows, st.s[p + 1] - st.s[p]);
    }
    if (!max_rows) return 0;
    pull3<<<blocks(max_rows * world * (wa + wb + wc) / 8), 256, 0, static_cast<cudaStream_t>(stream)>>>(peers, st, rank, world,
        wa / 8, wb / 8, wc / 8, static_cast<uint4*>(a), pa / 8, static_cast<uint4*>(b), pb / 8, static_cast<uint4*>(c), pc / 8, max_rows);
    return cudaGetLastError() == cudaSuccess ? 0 : fail("direct pull launch");
}
extern "C" int q3_cp_pull_rows(const void* const* views, int world, int64_t start, int64_t rows, int width, int64_t pitch,
                               int64_t step, void* out, void* stream) {
    Peers peers{};
    if (!load_peers(views, world, peers) || start < 0 || rows < 1 || width < 8 || (width | pitch | step) % 8 || !aligned16({out}))
        return fail("invalid direct row pull");
    pull_rows<<<blocks(rows * world * width / 8), 256, 0, static_cast<cudaStream_t>(stream)>>>(peers, world, start, rows,
        width / 8, pitch / 8, step / 8, static_cast<uint4*>(out));
    return cudaGetLastError() == cudaSuccess ? 0 : fail("direct row pull launch");
}
