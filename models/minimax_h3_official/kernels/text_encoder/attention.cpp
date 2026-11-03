// Qwen3-VL inference cuDNN SDPA. Derived from the validated H3 dense port
// and pinned PyTorch MHA.cpp; reference_sources/LICENSE retains its license.
#include <cuda_runtime_api.h>
#include <cudnn.h>
#include <cudnn_frontend.h>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
#include "../common/plan_cache.h"
namespace fe = cudnn_frontend;
namespace {
thread_local std::string last_error;
void check(cudnnStatus_t code) { if (code != CUDNN_STATUS_SUCCESS) throw std::runtime_error(cudnnGetErrorString(code)); }
void check(fe::error_t code) { if (!code.is_good()) throw std::runtime_error(code.get_message()); }
// One cuDNN handle for every plan: a new text length builds a new plan, and
// creating a handle per plan cost tens of milliseconds per request. Plans are
// used serially on the conditioner's one stream (set before each execute).
struct Handle {
    cudnnHandle_t raw = nullptr;
    Handle() { check(cudnnCreate(&raw)); }
    ~Handle() { if (raw) cudnnDestroy(raw); }
};
std::shared_ptr<Handle> shared_handle() {
    thread_local auto handle = std::make_shared<Handle>();
    return handle;
}
struct State {
    std::shared_ptr<Handle> owner;
    cudnnHandle_t handle = nullptr;
    std::unique_ptr<fe::graph::Graph> graph;
    size_t workspace = 0;
    std::string plan_name;
    State(int kind, int64_t sequence) {
        if ((kind != 0 && kind != 1) || sequence<=0) throw std::runtime_error("invalid Qwen attention shape");
        owner = shared_handle(); handle = owner->raw;
        graph = std::make_unique<fe::graph::Graph>();
        const std::string cache_key="qwen:5fffcc58a39ab00d061d8f9c398fe2652e1165c3557a2e3d3cd6b69223952596:"+std::to_string(kind)+":"+std::to_string(sequence);
        if(h3_plan_cache::load(graph,handle,cache_key,&plan_name)){workspace=graph->get_workspace_size();return;}
        graph->set_io_data_type(fe::DataType_t::BFLOAT16).set_intermediate_data_type(fe::DataType_t::FLOAT).set_compute_data_type(fe::DataType_t::FLOAT);
        auto scale = graph->tensor(fe::graph::Tensor_attributes().set_uid(5).set_name("Attn_scale")
            .set_dim({1,1,1,1}).set_stride({1,1,1,1}).set_is_pass_by_value(true).set_data_type(fe::DataType_t::FLOAT));
        auto options = fe::graph::SDPA_attributes().set_name("CUDNN_SDPA").set_generate_stats(false)
            .set_causal_mask(kind == 1).set_attn_scale(scale);
        auto q = graph->tensor(fe::graph::Tensor_attributes().set_uid(0).set_name("Q"));
        auto k = graph->tensor(fe::graph::Tensor_attributes().set_uid(1).set_name("K"));
        auto v = graph->tensor(fe::graph::Tensor_attributes().set_uid(2).set_name("V"));
        auto [o,stats] = graph->sdpa(q,k,v,options);
        o->set_uid(3).set_output(true);
        if (stats) throw std::runtime_error("unexpected statistics output");
        if (kind == 0) {
            for (auto& tensor : {q,k,v,o}) tensor->set_dim({1,16,sequence,72});
            for (auto& tensor : {q,k,o}) tensor->set_stride({1152,72,1152,1});
            v->set_stride({1152,72,3456,1});
        } else {
            for (auto& tensor : {q,o}) tensor->set_dim({1,64,sequence,128}).set_stride({sequence*8192,128,8192,1});
            for (auto& tensor : {k,v}) tensor->set_dim({1,8,sequence,128}).set_stride({sequence*1024,128,1024,1});
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
    ~State() { graph.reset(); }
};
using States = std::map<std::pair<int,int64_t>,std::unique_ptr<State>>;
States& states() { thread_local States* cache = new States(); return *cache; }
State& state(int kind,int64_t sequence) { auto& cache=states();auto key=std::make_pair(kind,sequence);if (!cache.count(key)) cache.emplace(key,std::make_unique<State>(kind,sequence));return *cache.at(key); }
template<typename F> int checked(F&& f) {
    try { f(); last_error.clear(); return 0; } catch (const std::exception& e) { last_error=e.what();return -1; }
}
}
extern "C" const char* q3_attention_error() { return last_error.c_str(); }
extern "C" int q3_attention_workspace(int kind, int64_t sequence, size_t* bytes) {
    return checked([&] { if (!bytes) throw std::runtime_error("null workspace result"); *bytes=state(kind,sequence).workspace; });
}
extern "C" int q3_attention_forward(int kind,int64_t sequence,const void* q,const void* k,const void* v,void* out,
    void* workspace,size_t bytes,float scale,void* stream) {
    return checked([&] {
        auto& s=state(kind,sequence);
        if (bytes<s.workspace || (s.workspace&&!workspace)) throw std::runtime_error("workspace too small");
        if (!std::isfinite(scale) || scale<=0) throw std::runtime_error("invalid scale");
        for (const void* p:{q,k,v,static_cast<const void*>(out)}) if (!p || reinterpret_cast<uintptr_t>(p)%16) throw std::runtime_error("unaligned/null tensor");
        check(cudnnSetStream(s.handle,static_cast<cudaStream_t>(stream)));
        std::unordered_map<int64_t,void*> pack{{0,const_cast<void*>(q)},{1,const_cast<void*>(k)},{2,const_cast<void*>(v)},{3,out},{5,&scale}};
        check(s.graph->execute(s.handle,pack,workspace));
    });
}
extern "C" const char* q3_attention_plan_name(int kind,int64_t sequence) {
    try { return state(kind,sequence).plan_name.c_str(); } catch(const std::exception& e) { last_error=e.what();return nullptr; }
}
extern "C" int q3_attention_cleanup() {
    return checked([&]{ if (cudaDeviceSynchronize()!=cudaSuccess) throw std::runtime_error("cleanup sync"); states().clear(); });
}

extern "C" size_t q3_attention_cache_size(){return states().size();}

// A metadata worker has its own handle and cache. It executes no attention.
// Only the serial owner adopts completed plans; the original device/shape key
// and vendor builder are unchanged. State retains the worker handle lifetime.
extern "C" int q3_attention_prepare_isolated(int device, int64_t vision, int64_t text, void** out) {
    return checked([&] {
        if (!out || vision < 0 || vision > 262144 || text <= 0 || text > 262144)
            throw std::runtime_error("invalid isolated attention shapes");
        if (cudaSetDevice(device) != cudaSuccess) throw std::runtime_error("prepare attention device");
        auto prepared = std::make_unique<States>();
        if (vision) prepared->emplace(std::make_pair(0,vision),std::make_unique<State>(0,vision));
        prepared->emplace(std::make_pair(1,text),std::make_unique<State>(1,text));
        *out = prepared.release();
    });
}
extern "C" int q3_attention_adopt_isolated(void* pointer) {
    return checked([&] {
        if (!pointer) throw std::runtime_error("missing prepared attention");
        std::unique_ptr<States> prepared(static_cast<States*>(pointer));
        for (auto& entry : *prepared)
            if (!states().count(entry.first)) states().emplace(entry.first,std::move(entry.second));
    });
}
extern "C" void q3_attention_drop_isolated(void* pointer) { delete static_cast<States*>(pointer); }
