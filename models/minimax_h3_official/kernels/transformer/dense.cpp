// Native form of PyTorch's inference-only cuDNN SDPA host graph.
// Source: pytorch cf30153c4c131c8164ee7798e5022d810682e2cb,
// aten/src/ATen/native/cudnn/MHA.cpp::build_graph / run_cudnn_SDP_fprop.
// Derived portions retain the BSD-style license in vendor_reference/PYTORCH-LICENSE.
#include <cuda_runtime_api.h>
#include <cudnn.h>
#include <cudnn_frontend.h>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "../common/plan_cache.h"
namespace fe = cudnn_frontend;
namespace {
thread_local std::string last_error;
void check(cudnnStatus_t code) {
    if (code != CUDNN_STATUS_SUCCESS) throw std::runtime_error(cudnnGetErrorString(code));
}
void check(fe::error_t code) {
    if (!code.is_good()) throw std::runtime_error(code.get_message());
}
struct State {
    cudnnHandle_t handle = nullptr;
    std::unique_ptr<fe::graph::Graph> graph;
    size_t workspace = 0;
    std::string plan_name;
    State(int tokens, int heads, int layout, int kv_tokens) {
        if (tokens <= 0 || kv_tokens <= 0 || (heads != 7 && heads != 14 && heads != 28 && heads != 56) || (layout != 0 && layout != 1 && layout != 2))
            throw std::runtime_error("dense requires positive S, H in {7,14,28,56}, D=128 and explicit main/refiner layout");
        const char* ragged = std::getenv("TORCH_CUDNN_SDPA_AVOID_RECOMPILE");
        if (ragged && std::string(ragged) == "1") throw std::runtime_error("ragged cuDNN dense mode is not supported");
        int device = 0, major = 0, minor = 0;
        if (cudaGetDevice(&device) != cudaSuccess ||
            cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device) != cudaSuccess ||
            cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device) != cudaSuccess ||
            major != 10 || (minor != 0 && minor != 3))
            throw std::runtime_error("cuDNN dense requires B200/sm100 or B300/sm103");
        check(cudnnCreate(&handle));
        graph = std::make_unique<fe::graph::Graph>();
        const std::string cache_key="dit:00002512113123d0b5ec05d8441ec98b363c24a43075142567823db04306af1a:"+std::to_string(tokens)+":"+std::to_string(heads)+":"+std::to_string(layout)+":"+std::to_string(kv_tokens);
        if(h3_plan_cache::load(graph,handle,cache_key,&plan_name)){workspace=graph->get_workspace_size();return;}
        graph->set_io_data_type(fe::DataType_t::BFLOAT16)
             .set_intermediate_data_type(fe::DataType_t::FLOAT)
             .set_compute_data_type(fe::DataType_t::FLOAT);
        // Exact UIDs and pass-by-value scale boundary used by pinned PyTorch.
        auto scale = graph->tensor(fe::graph::Tensor_attributes()
            .set_uid(5).set_name("Attn_scale").set_dim({1, 1, 1, 1})
            .set_stride({1, 1, 1, 1}).set_is_pass_by_value(true)
            .set_data_type(fe::DataType_t::FLOAT));
        auto options = fe::graph::SDPA_attributes().set_name("CUDNN_SDPA")
            .set_generate_stats(false).set_causal_mask(false).set_attn_scale(scale);
        auto q = graph->tensor(fe::graph::Tensor_attributes().set_uid(0).set_name("Q"));
        auto k = graph->tensor(fe::graph::Tensor_attributes().set_uid(1).set_name("K"));
        auto v = graph->tensor(fe::graph::Tensor_attributes().set_uid(2).set_name("V"));
        auto [o, stats] = graph->sdpa(q, k, v, options);
        o->set_uid(3).set_output(true);
        if (stats) throw std::runtime_error("unexpected softmax stats in inference-only dense graph");
        const std::vector<int64_t> dimensions{1, heads, tokens, 128};
        // Main `_dense` makes BHSD via SHD.transpose(0,1).unsqueeze(0),
        // whose singleton batch stride is H*D. Refiner transposes BSHD.
        // PyTorch's empty_like output preserves those exact view strides.
        const int64_t batch_stride = layout == 0 ? int64_t(heads)*128 : int64_t(tokens)*heads*128;
        const std::vector<int64_t> strides{batch_stride, 128, int64_t(heads)*128, 1};
        for (auto& tensor : {q, o}) tensor->set_dim(dimensions).set_stride(strides);
        const std::vector<int64_t> kv_dimensions{1, heads, kv_tokens, 128};
        const int64_t kv_batch_stride = layout == 0 ? int64_t(heads)*128 : int64_t(kv_tokens)*heads*128;
        const std::vector<int64_t> kv_strides{kv_batch_stride,128,int64_t(heads)*128,1};
        for (auto& tensor : {k, v}) tensor->set_dim(kv_dimensions).set_stride(kv_strides);
        if(layout==2) {
            // Match SGLang packed-QKV SHD views after unsqueeze(0).transpose(1,2).
            const std::vector<int64_t> packed_stride{int64_t(tokens)*3*heads*128,128,int64_t(3)*heads*128,1};
            for(auto& tensor:{q,k,v})tensor->set_stride(packed_stride);
            o->set_stride({int64_t(tokens)*heads*128,128,int64_t(heads)*128,1});
        }
        check(graph->validate());
        check(graph->build_operation_graph(handle));
        check(graph->create_execution_plans({fe::HeurMode_t::A}));
        check(graph->check_support(handle));
        check(graph->build_plans(handle));
        h3_plan_cache::save(*graph,cache_key);
        workspace = graph->get_workspace_size();
        check(graph->get_plan_name(plan_name));
    }
    ~State() {
        // Shared plans stay alive until explicit cleanup, which synchronizes CUDA.
        graph.reset();
        if (handle) cudnnDestroy(handle);
    }
};
using States = std::map<std::tuple<int, int, int, int>, std::unique_ptr<State>>;
// Match PyTorch's thread-local graph cache lifetime, avoiding CUDA teardown races.
States& states() { thread_local States* cache = new States(); return *cache; }
struct CacheStats {size_t builds=0,hits=0,evictions=0,tick=0;std::map<std::tuple<int,int,int,int>,size_t> used;};
CacheStats& counters(){thread_local CacheStats* stats=new CacheStats();return *stats;}
constexpr size_t MAX_PLANS=32;
State& state(int tokens, int heads, int layout, int kv_tokens=0) {
    if(kv_tokens==0)kv_tokens=tokens;
    const auto key=std::make_tuple(tokens,heads,layout,kv_tokens);
    auto& cache=states();auto& stats=counters();
    if(cache.count(key)){stats.hits++;stats.used[key]=++stats.tick;return *cache.at(key);}
    if(cache.size()>=MAX_PLANS){
        auto oldest=stats.used.begin();
        for(auto it=stats.used.begin();it!=stats.used.end();++it)if(it->second<oldest->second)oldest=it;
        if(cudaDeviceSynchronize()!=cudaSuccess)throw std::runtime_error("dense cache eviction sync failed");
        cache.erase(oldest->first);stats.used.erase(oldest);stats.evictions++;
    }
    auto plan=std::make_unique<State>(tokens,heads,layout,kv_tokens);
    cache.emplace(key,std::move(plan));stats.used[key]=++stats.tick;stats.builds++;
    return *cache.at(key);
}
int legacy_layout(int tokens) {
    if (tokens == 63510) return 0;
    if (tokens == 2456) return 1;
    throw std::runtime_error("legacy dense ABI only admits validated case-01 shapes; use explicit layout v2 ABI");
}
template<typename F> int checked(F&& function) {
    try { function(); last_error.clear(); return 0; }
    catch (const std::exception& error) { last_error = error.what(); return -1; }
    catch (...) { last_error = "unknown dense C++ error"; return -1; }
}
}
extern "C" const char* h3_dense_error() { return last_error.c_str(); }
extern "C" size_t h3_dense_cudnn_version() { return cudnnGetVersion(); }
extern "C" int h3_dense_workspace_bytes(int tokens, int heads, size_t* bytes) {
    return checked([&] {
        if (!bytes) throw std::runtime_error("null workspace byte result");
        *bytes = state(tokens, heads, legacy_layout(tokens)).workspace;
    });
}
extern "C" int h3_dense_forward(const void* q, const void* k, const void* v, void* out,
    void* workspace, size_t bytes, int tokens, int heads, float scale, void* stream) {
    return checked([&] {
        auto& s = state(tokens, heads, legacy_layout(tokens));
        if (!std::isfinite(scale) || scale <= 0) throw std::runtime_error("invalid attention scale");
        for (const void* pointer : {q, k, v, static_cast<const void*>(out)})
            if (!pointer || reinterpret_cast<uintptr_t>(pointer) % 16)
                throw std::runtime_error("dense pointers must be nonnull and 16-byte aligned");
        if (bytes < s.workspace || (s.workspace && !workspace)) throw std::runtime_error("dense workspace too small");
        check(cudnnSetStream(s.handle, static_cast<cudaStream_t>(stream)));
        std::unordered_map<int64_t, void*> pack{{0, const_cast<void*>(q)},
            {1, const_cast<void*>(k)}, {2, const_cast<void*>(v)}, {3, out}, {5, &scale}};
        check(s.graph->execute(s.handle, pack, workspace));
    });
}
extern "C" const char* h3_dense_plan_name(int tokens, int heads) {
    try { return state(tokens, heads, legacy_layout(tokens)).plan_name.c_str(); }
    catch (const std::exception& e) { last_error = e.what(); return nullptr; }
}
extern "C" int h3_dense_dump_plan(int tokens, int heads, const char* path) {
    return checked([&] {
        auto& s = state(tokens, heads, legacy_layout(tokens));
        std::ofstream graph(std::string(path) + ".graph.json");
        graph << *s.graph;
        if (!graph) throw std::runtime_error("could not write graph provenance");
        std::vector<uint8_t> serialized;
        check(s.graph->serialize(serialized));
        std::ofstream plan(std::string(path) + ".plan.ubjson", std::ios::binary);
        plan.write(reinterpret_cast<const char*>(serialized.data()), serialized.size());
        if (!plan) throw std::runtime_error("could not write serialized plan provenance");
    });
}
extern "C" int h3_dense_cleanup() {
    return checked([&] {
        if (cudaDeviceSynchronize() != cudaSuccess) throw std::runtime_error("dense cleanup sync failed");
        states().clear();counters().used.clear();
    });
}
extern "C" int h3_dense_workspace_bytes_v2(int tokens, int heads, int layout, size_t* bytes) {
    return checked([&] {
        if (!bytes) throw std::runtime_error("null workspace byte result");
        *bytes = state(tokens, heads, layout).workspace;
    });
}
extern "C" int h3_dense_forward_v2(const void* q, const void* k, const void* v, void* out,
    void* workspace, size_t bytes, int tokens, int heads, int layout, float scale, void* stream) {
    return checked([&] {
        auto& s = state(tokens, heads, layout);
        if (!std::isfinite(scale) || scale <= 0) throw std::runtime_error("invalid attention scale");
        for (const void* pointer : {q, k, v, static_cast<const void*>(out)})
            if (!pointer || reinterpret_cast<uintptr_t>(pointer) % 16)
                throw std::runtime_error("dense pointers must be nonnull and 16-byte aligned");
        if (bytes < s.workspace || (s.workspace && !workspace)) throw std::runtime_error("dense workspace too small");
        check(cudnnSetStream(s.handle, static_cast<cudaStream_t>(stream)));
        std::unordered_map<int64_t, void*> pack{{0,const_cast<void*>(q)}, {1,const_cast<void*>(k)},
            {2,const_cast<void*>(v)}, {3,out}, {5,&scale}};
        check(s.graph->execute(s.handle, pack, workspace));
    });
}
extern "C" const char* h3_dense_plan_name_v2(int tokens, int heads, int layout) {
    try { return state(tokens, heads, layout).plan_name.c_str(); }
    catch (const std::exception& e) { last_error = e.what(); return nullptr; }
}
extern "C" int h3_dense_dump_plan_v2(int tokens, int heads, int layout, const char* path) {
    return checked([&] {
        auto& s=state(tokens,heads,layout);
        std::ofstream graph(std::string(path)+".graph.json");graph<<*s.graph;
        if (!graph) throw std::runtime_error("could not write graph provenance");
        std::vector<uint8_t> serialized;check(s.graph->serialize(serialized));
        std::ofstream plan(std::string(path)+".plan.ubjson",std::ios::binary);
        plan.write(reinterpret_cast<const char*>(serialized.data()),serialized.size());
        if (!plan) throw std::runtime_error("could not write serialized plan provenance");
    });
}

extern "C" const char* h3_dense_cache_report(){
    thread_local std::string report;
    auto& stats=counters();std::ostringstream out;
    out<<"{\"builds\":"<<stats.builds<<",\"hits\":"<<stats.hits
       <<",\"evictions\":"<<stats.evictions<<",\"size\":"<<states().size()
       <<",\"capacity\":"<<MAX_PLANS<<",\"plans\":[";
    bool first=true;
    for(auto& item:states()){
        if(!first)out<<",";first=false;
        auto [tokens,heads,layout,kv_tokens]=item.first;
        out<<"{\"tokens\":"<<tokens<<",\"heads\":"<<heads<<",\"layout\":"<<layout
           <<",\"workspace_bytes\":"<<item.second->workspace<<"}";
    }
    out<<"]}";report=out.str();return report.c_str();
}

// Ref2VA's sparse soundtrack correction: Q is a contiguous row slice,
// K/V span the complete sequence. Preserve the original SHD view strides.
extern "C" int h3_dense_rect_workspace_bytes(int q_tokens,int kv_tokens,int heads,size_t* bytes){
    return checked([&]{
        if(!bytes)throw std::runtime_error("null rectangular workspace result");
        *bytes=state(q_tokens,heads,0,kv_tokens).workspace;
    });
}
extern "C" int h3_dense_rect_forward(const void* q,const void* k,const void* v,void* out,
    void* workspace,size_t bytes,int q_tokens,int kv_tokens,int heads,float scale,void* stream){
    return checked([&]{
        auto& s=state(q_tokens,heads,0,kv_tokens);
        if(!std::isfinite(scale)||scale<=0)throw std::runtime_error("invalid rectangular scale");
        for(const void* pointer:{q,k,v,static_cast<const void*>(out)})
            if(!pointer||reinterpret_cast<uintptr_t>(pointer)%16)throw std::runtime_error("unaligned rectangular pointer");
        if(bytes<s.workspace||(s.workspace&&!workspace))throw std::runtime_error("rectangular workspace too small");
        check(cudnnSetStream(s.handle,static_cast<cudaStream_t>(stream)));
        std::unordered_map<int64_t,void*> pack{{0,const_cast<void*>(q)},{1,const_cast<void*>(k)},
            {2,const_cast<void*>(v)},{3,out},{5,&scale}};
        check(s.graph->execute(s.handle,pack,workspace));
    });
}
