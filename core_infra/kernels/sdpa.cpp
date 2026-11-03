// Native form of PyTorch's inference-only cuDNN SDPA (dense, no mask, no dropout,
// no softmax stats) for arbitrary [B,H,S,D] shapes and strides.
// Source: pytorch cf30153c4c131c8164ee7798e5022d810682e2cb,
// aten/src/ATen/native/cudnn/MHA.cpp::build_graph / run_cudnn_SDP_fprop and
// aten/src/ATen/native/transformers/sdp_utils.cpp::alloc_with_matching_layout.
// Derived portions retain PyTorch's BSD-style license.
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cudnn.h>
#include <cudnn_frontend.h>
#include "plan_cache.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace fe = cudnn_frontend;
namespace {
thread_local std::string last_error;
void check(cudnnStatus_t code) { if (code != CUDNN_STATUS_SUCCESS) throw std::runtime_error(cudnnGetErrorString(code)); }
void check(fe::error_t code) { if (!code.is_good()) throw std::runtime_error(code.get_message()); }
using Dims = std::array<int64_t, 4>;
// MHA.cpp UIDs: Q, K, V, O, (LSE), SCALE = 5.
constexpr int64_t UID_Q = 0, UID_K = 1, UID_V = 2, UID_O = 3, UID_SCALE = 5;

std::vector<int64_t> fix_size_one_strides(const Dims& sizes, const Dims& strides_in) {
    std::vector<int64_t> strides(strides_in.begin(), strides_in.end());
    for (int d = 0; d < 4; d++) {
        if (sizes[d] == 1 && !strides[d]) {
            int64_t s = 1;
            for (int d2 = d + 1; d2 < 4; d2++) s *= strides[d2];
            strides[d] = s;
        }
    }
    return strides;
}

bool dense_non_overlapping(const Dims& sizes, const Dims& strides) {
    std::array<int, 4> order{0, 1, 2, 3};
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return strides[a] < strides[b]; });
    int64_t expected = 1;
    for (int d : order) {
        if (sizes[d] == 1) continue;
        if (strides[d] != expected) return false;
        expected *= sizes[d];
    }
    return true;
}

// at::empty_like(q) (preserve format) when the output shape equals q's, else
// alloc_with_matching_layout: both order dimensions by q's strides.
Dims output_strides(const Dims& q_sizes, const Dims& q_strides, const Dims& o_sizes) {
    if (q_sizes == o_sizes && dense_non_overlapping(q_sizes, q_strides)) return q_strides;
    std::array<int, 4> fill{0, 1, 2, 3};
    std::stable_sort(fill.begin(), fill.end(), [&](int a, int b) {
        int64_t s1 = q_strides[a] ? q_strides[a] : INT64_MAX, s2 = q_strides[b] ? q_strides[b] : INT64_MAX;
        return s1 < s2;
    });
    Dims out{};
    int64_t current = 1;
    for (int d : fill) { out[d] = current; current *= o_sizes[d]; }
    return out;
}

struct Key {
    // Driver IDs are unique for the program's lifetime in this CUDA instance,
    // unlike CUcontext addresses. A recreated/private context needs its own plan.
    unsigned long long context_id;
    int dtype; Dims q, qs, k, ks, v, vs;
    bool operator<(const Key& o) const {
        return std::tie(context_id, dtype, q, qs, k, ks, v, vs) <
            std::tie(o.context_id, o.dtype, o.q, o.qs, o.k, o.ks, o.v, o.vs);
    }
};

std::string cache_shape(const Key& key) {
    std::ostringstream out;
    // Version the exact operator semantics, UIDs, output layout and heuristic.
    out << "dense-sdpa-exact-v1:" << key.dtype;
    for (const auto& dims : {key.q, key.qs, key.k, key.ks, key.v, key.vs})
        for (int64_t n : dims) out << ':' << n;
    return out.str();
}

struct Plan {
    // Mutable handle/graph calls on one plan are serialized; independent
    // contexts and shapes can construct and execute their own plans in parallel.
    std::mutex execution_mutex;
    cudnnHandle_t handle = nullptr;
    std::unique_ptr<fe::graph::Graph> graph;
    size_t workspace = 0;
    std::string plan_name;
    Dims o_sizes{}, o_strides{};
    explicit Plan(const Key& key) {
        int device = 0, major = 0, minor = 0;
        if (cudaGetDevice(&device) != cudaSuccess ||
            cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device) != cudaSuccess ||
            cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device) != cudaSuccess || major != 10)
            throw std::runtime_error("native exact cuDNN SDPA requires sm100");
        const char* ragged = std::getenv("TORCH_CUDNN_SDPA_AVOID_RECOMPILE");
        if (ragged && std::string(ragged) == "1") throw std::runtime_error("ragged cuDNN dense mode is not part of the gold");
        if (key.q[0] != key.k[0] || key.q[1] != key.k[1] || key.k != key.v || key.q[3] != key.k[3])
            throw std::runtime_error("SDPA requires matching batch/heads and equal K/V shapes");
        o_sizes = {key.q[0], key.q[1], key.q[2], key.v[3]};
        o_strides = output_strides(key.q, key.qs, o_sizes);
        check(cudnnCreate(&handle));
        const auto cache = infra_plan_cache::configuration();
        const auto shape = cache_shape(key);
        if (infra_plan_cache::load(cache, graph, handle, shape, &plan_name)) {
            workspace = graph->get_workspace_size();
            return;
        }
        graph = std::make_unique<fe::graph::Graph>();
        const auto io = key.dtype == 0 ? fe::DataType_t::BFLOAT16 : fe::DataType_t::HALF;
        graph->set_io_data_type(io).set_intermediate_data_type(fe::DataType_t::FLOAT)
             .set_compute_data_type(fe::DataType_t::FLOAT);
        auto scale = graph->tensor(fe::graph::Tensor_attributes().set_uid(UID_SCALE).set_name("Attn_scale")
            .set_dim({1, 1, 1, 1}).set_stride({1, 1, 1, 1}).set_is_pass_by_value(true)
            .set_data_type(fe::DataType_t::FLOAT));
        auto options = fe::graph::SDPA_attributes().set_name("CUDNN_SDPA").set_generate_stats(false)
            .set_causal_mask(false).set_attn_scale(scale);
        auto q = graph->tensor(fe::graph::Tensor_attributes().set_uid(UID_Q).set_name("Q"));
        auto k = graph->tensor(fe::graph::Tensor_attributes().set_uid(UID_K).set_name("K"));
        auto v = graph->tensor(fe::graph::Tensor_attributes().set_uid(UID_V).set_name("V"));
        auto [o, stats] = graph->sdpa(q, k, v, options);
        if (stats) throw std::runtime_error("unexpected softmax stats in an inference graph");
        o->set_uid(UID_O).set_output(true);
        auto vec = [](const Dims& d) { return std::vector<int64_t>(d.begin(), d.end()); };
        q->set_dim(vec(key.q)).set_stride(fix_size_one_strides(key.q, key.qs));
        k->set_dim(vec(key.k)).set_stride(fix_size_one_strides(key.k, key.ks));
        v->set_dim(vec(key.v)).set_stride(fix_size_one_strides(key.v, key.vs));
        o->set_dim(vec(o_sizes)).set_stride(fix_size_one_strides(o_sizes, o_strides));
        check(graph->validate());
        check(graph->build_operation_graph(handle));
        check(graph->create_execution_plans({fe::HeurMode_t::A}));
        check(graph->check_support(handle));
        check(graph->build_plans(handle));
        workspace = graph->get_workspace_size();
        check(graph->get_plan_name(plan_name));
        infra_plan_cache::save(cache, *graph, shape);
    }
    ~Plan() { graph.reset(); if (handle) cudnnDestroy(handle); }
};

// Process-wide, so a background thread can build a request's plans while weights stream in;
// map nodes are stable, so a returned plan outlives later insertions.
struct Slot { std::once_flag initialized; std::unique_ptr<Plan> value; };
std::map<Key, std::unique_ptr<Slot>>& plans() { static auto* cache = new std::map<Key, std::unique_ptr<Slot>>(); return *cache; }
std::mutex& plans_mutex() { static std::mutex m; return m; }

Plan& plan(int dtype, const int64_t* q, const int64_t* qs, const int64_t* k, const int64_t* ks,
           const int64_t* v, const int64_t* vs) {
    if (dtype != 0 && dtype != 1) throw std::runtime_error("SDPA dtype must be 0 (bf16) or 1 (fp16)");
    CUcontext context = nullptr;
    unsigned long long context_id = 0;
    if (cuCtxGetCurrent(&context) != CUDA_SUCCESS || !context ||
        cuCtxGetId(context, &context_id) != CUDA_SUCCESS)
        throw std::runtime_error("SDPA needs a live current CUDA context");
    Key key{context_id, dtype, {}, {}, {}, {}, {}, {}};
    std::copy(q, q + 4, key.q.begin()); std::copy(qs, qs + 4, key.qs.begin());
    std::copy(k, k + 4, key.k.begin()); std::copy(ks, ks + 4, key.ks.begin());
    std::copy(v, v + 4, key.v.begin()); std::copy(vs, vs + 4, key.vs.begin());
    Slot* slot;
    {
        std::lock_guard<std::mutex> lock(plans_mutex());
        auto& cache = plans();
        auto it = cache.find(key);
        if (it == cache.end()) it = cache.emplace(key, std::make_unique<Slot>()).first;
        slot = it->second.get();
    }
    // A failed build leaves call_once retryable. The retained Slot remains
    // stable without blocking another rank/context's compilation or lookup.
    std::call_once(slot->initialized, [&] { slot->value = std::make_unique<Plan>(key); });
    return *slot->value;
}

template <typename F> int guarded(F&& f) {
    try { f(); last_error.clear(); return 0; }
    catch (const std::exception& e) { last_error = e.what(); return -1; }
    catch (...) { last_error = "unknown SDPA error"; return -1; }
}
}  // namespace

extern "C" const char* infra_sdpa_error() { return last_error.c_str(); }

// Output strides PyTorch allocates for this query layout, workspace bytes and plan name.
extern "C" int infra_sdpa_prepare(int dtype, const int64_t* q, const int64_t* qs, const int64_t* k, const int64_t* ks,
                                const int64_t* v, const int64_t* vs, int64_t* o_strides, size_t* workspace,
                                char* plan_name, size_t plan_name_bytes) {
    return guarded([&] {
        auto& p = plan(dtype, q, qs, k, ks, v, vs);
        std::copy(p.o_strides.begin(), p.o_strides.end(), o_strides);
        *workspace = p.workspace;
        if (plan_name && plan_name_bytes) {
            auto n = std::min(plan_name_bytes - 1, p.plan_name.size());
            std::copy(p.plan_name.begin(), p.plan_name.begin() + n, plan_name);
            plan_name[n] = 0;
        }
    });
}

extern "C" int infra_sdpa_forward(int dtype, const int64_t* q_dims, const int64_t* q_strides, const int64_t* k_dims,
                                const int64_t* k_strides, const int64_t* v_dims, const int64_t* v_strides,
                                const void* q, const void* k, const void* v, void* o, void* workspace,
                                size_t workspace_bytes, float scale, void* stream) {
    return guarded([&] {
        auto& p = plan(dtype, q_dims, q_strides, k_dims, k_strides, v_dims, v_strides);
        std::lock_guard<std::mutex> execution_lock(p.execution_mutex);
        if (!std::isfinite(scale) || scale <= 0) throw std::runtime_error("invalid attention scale");
        for (const void* pointer : {q, k, v, static_cast<const void*>(o)})
            if (!pointer || reinterpret_cast<uintptr_t>(pointer) % 16) throw std::runtime_error("SDPA pointers must be 16-byte aligned");
        if (workspace_bytes < p.workspace || (p.workspace && !workspace)) throw std::runtime_error("SDPA workspace too small");
        check(cudnnSetStream(p.handle, static_cast<cudaStream_t>(stream)));
        std::unordered_map<int64_t, void*> pack{{UID_Q, const_cast<void*>(q)}, {UID_K, const_cast<void*>(k)},
            {UID_V, const_cast<void*>(v)}, {UID_O, o}, {UID_SCALE, &scale}};
        check(p.graph->execute(p.handle, pack, workspace));
    });
}
