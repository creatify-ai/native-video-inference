// cuDNN SDPA for one rank's head group after the sequence all-to-all.
// Text: the accepted causal GQA plan settings (attention.cpp kind 1) on
// 64/world query heads sharing 8/world KV heads. Vision: the accepted
// non-causal plan (kind 0) on 16/world heads of one image. Head-partitioned
// outputs match the full-head plan bit for bit (checked on B200).
#include <cuda_runtime_api.h>
#include <cudnn.h>
#include <cudnn_frontend.h>
#include <cmath>
#include <cstdint>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
namespace fe = cudnn_frontend;
namespace {
thread_local std::string last_error;
void check(cudnnStatus_t code) { if (code != CUDNN_STATUS_SUCCESS) throw std::runtime_error(cudnnGetErrorString(code)); }
void check(fe::error_t code) { if (!code.is_good()) throw std::runtime_error(code.get_message()); }
cudnnHandle_t shared_handle() {
    thread_local cudnnHandle_t handle = [] { cudnnHandle_t h = nullptr; check(cudnnCreate(&h)); return h; }();
    return handle;
}
// kind 1: text, packed q[S, 8192/world], k/v[S, 1024/world], out[S, 8192/world].
// kind 0: vision, the accepted full-head row strides (q/k/out 1152, v 3456 inside
// qkv) with the caller's head-group column offset, as verified by the probe.
struct State {
    std::unique_ptr<fe::graph::Graph> graph;
    size_t workspace = 0;
    std::string plan_name;
    State(int kind, int world, int64_t s) {
        if ((kind != 0 && kind != 1) || s <= 0 || (world != 1 && world != 2 && world != 4 && world != 8))
            throw std::runtime_error("invalid parallel Qwen attention shape");
        graph = std::make_unique<fe::graph::Graph>();
        graph->set_io_data_type(fe::DataType_t::BFLOAT16).set_intermediate_data_type(fe::DataType_t::FLOAT).set_compute_data_type(fe::DataType_t::FLOAT);
        auto scale = graph->tensor(fe::graph::Tensor_attributes().set_uid(5).set_name("Attn_scale")
            .set_dim({1,1,1,1}).set_stride({1,1,1,1}).set_is_pass_by_value(true).set_data_type(fe::DataType_t::FLOAT));
        auto options = fe::graph::SDPA_attributes().set_name("CUDNN_SDPA").set_generate_stats(false)
            .set_causal_mask(kind == 1).set_attn_scale(scale);
        auto q = graph->tensor(fe::graph::Tensor_attributes().set_uid(0).set_name("Q"));
        auto k = graph->tensor(fe::graph::Tensor_attributes().set_uid(1).set_name("K"));
        auto v = graph->tensor(fe::graph::Tensor_attributes().set_uid(2).set_name("V"));
        auto [o, stats] = graph->sdpa(q, k, v, options);
        o->set_uid(3).set_output(true);
        if (stats) throw std::runtime_error("unexpected statistics output");
        if (kind == 0) {
            for (auto& tensor : {q, k, v, o}) tensor->set_dim({1, 16 / world, s, 72});
            for (auto& tensor : {q, k, o}) tensor->set_stride({1152, 72, 1152, 1});
            v->set_stride({1152, 72, 3456, 1});
        } else {
            const int64_t heads = 64 / world, kv = 8 / world;
            for (auto& tensor : {q, o}) tensor->set_dim({1, heads, s, 128}).set_stride({s * heads * 128, 128, heads * 128, 1});
            for (auto& tensor : {k, v}) tensor->set_dim({1, kv, s, 128}).set_stride({s * kv * 128, 128, kv * 128, 1});
        }
        auto handle = shared_handle();
        check(graph->validate());
        check(graph->build_operation_graph(handle));
        check(graph->create_execution_plans({fe::HeurMode_t::A}));
        check(graph->check_support(handle));
        check(graph->build_plans(handle));
        workspace = graph->get_workspace_size();
        check(graph->get_plan_name(plan_name));
    }
};
using Key = std::tuple<int, int, int64_t>;
std::map<Key, std::unique_ptr<State>>& states() { thread_local auto* cache = new std::map<Key, std::unique_ptr<State>>(); return *cache; }
State& state(int kind, int world, int64_t s) {
    auto& cache = states(); Key key{kind, world, s};
    auto found = cache.find(key);
    if (found == cache.end()) found = cache.emplace(key, std::make_unique<State>(kind, world, s)).first;
    return *found->second;
}
template <typename F> int checked(F&& f) {
    try { f(); last_error.clear(); return 0; } catch (const std::exception& e) { last_error = e.what(); return -1; }
}
}  // namespace

extern "C" const char* q3_cp_attention_error() { return last_error.c_str(); }
extern "C" int q3_cp_attention_workspace(int kind, int world, int64_t sequence, size_t* bytes) {
    return checked([&] { if (!bytes) throw std::runtime_error("null workspace result"); *bytes = state(kind, world, sequence).workspace; });
}
extern "C" int q3_cp_attention_forward(int kind, int world, int64_t sequence, const void* q, const void* k, const void* v,
                                       void* out, void* workspace, size_t bytes, float scale, void* stream) {
    return checked([&] {
        auto& s = state(kind, world, sequence);
        if (bytes < s.workspace || (s.workspace && !workspace)) throw std::runtime_error("workspace too small");
        if (!std::isfinite(scale) || scale <= 0) throw std::runtime_error("invalid scale");
        for (const void* p : {q, k, v, static_cast<const void*>(out)})
            if (!p || reinterpret_cast<uintptr_t>(p) % 16) throw std::runtime_error("unaligned/null tensor");
        auto handle = shared_handle();
        check(cudnnSetStream(handle, static_cast<cudaStream_t>(stream)));
        std::unordered_map<int64_t, void*> pack{{0, const_cast<void*>(q)}, {1, const_cast<void*>(k)},
                                                 {2, const_cast<void*>(v)}, {3, out}, {5, &scale}};
        check(s.graph->execute(handle, pack, workspace));
    });
}
extern "C" const char* q3_cp_attention_plan_name(int kind, int world, int64_t sequence) {
    try { return state(kind, world, sequence).plan_name.c_str(); } catch (const std::exception& e) { last_error = e.what(); return nullptr; }
}
extern "C" int q3_cp_attention_cleanup() {
    return checked([&] { if (cudaDeviceSynchronize() != cudaSuccess) throw std::runtime_error("cleanup sync"); states().clear(); });
}
extern "C" size_t q3_cp_attention_cache_size() { return states().size(); }
