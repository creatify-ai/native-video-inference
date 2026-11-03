// Native host form of pinned PyTorch MHA.cpp build_graph/run_cudnn_SDP_fprop.
// Adapted from an earlier, bitwise-validated native port of the same SDPA graph, with
// runtime geometry/strides and FP16 for the actual compiled video-VAE graph.
// Derived portions retain the PyTorch license (licenses/PYTORCH-LICENSE).
#include <cuda_runtime_api.h>
#include <cudnn.h>
#include <cudnn_frontend.h>
#include <array>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace fe = cudnn_frontend;
namespace {
thread_local std::string error;
void check(cudnnStatus_t c) { if (c != CUDNN_STATUS_SUCCESS) throw std::runtime_error(cudnnGetErrorString(c)); }
void check(fe::error_t c) { if (!c.is_good()) throw std::runtime_error(c.get_message()); }
struct State {
    cudnnHandle_t handle = nullptr;
    std::unique_ptr<fe::graph::Graph> graph;
    void* workspace = nullptr;
    std::string tag;
    State(const int64_t* dims, const int64_t* qs, const int64_t* ks,
          const int64_t* vs, const int64_t* os) {
        check(cudnnCreate(&handle));
        graph = std::make_unique<fe::graph::Graph>();
        graph->set_io_data_type(fe::DataType_t::HALF)
             .set_intermediate_data_type(fe::DataType_t::FLOAT)
             .set_compute_data_type(fe::DataType_t::FLOAT);
        auto scale = graph->tensor(fe::graph::Tensor_attributes().set_uid(5).set_name("Attn_scale")
            .set_dim({1,1,1,1}).set_stride({1,1,1,1}).set_is_pass_by_value(true).set_data_type(fe::DataType_t::FLOAT));
        auto options = fe::graph::SDPA_attributes().set_name("CUDNN_SDPA")
            .set_generate_stats(false).set_causal_mask(false).set_attn_scale(scale);
        auto q = graph->tensor(fe::graph::Tensor_attributes().set_uid(0).set_name("Q"));
        auto k = graph->tensor(fe::graph::Tensor_attributes().set_uid(1).set_name("K"));
        auto v = graph->tensor(fe::graph::Tensor_attributes().set_uid(2).set_name("V"));
        auto [o, stats] = graph->sdpa(q,k,v,options);
        if (stats) throw std::runtime_error("unexpected SDPA statistics output");
        o->set_uid(3).set_output(true);
        const std::vector<int64_t> dimensions(dims,dims+4);
        q->set_dim(dimensions).set_stride(std::vector<int64_t>(qs,qs+4));
        k->set_dim(dimensions).set_stride(std::vector<int64_t>(ks,ks+4));
        v->set_dim(dimensions).set_stride(std::vector<int64_t>(vs,vs+4));
        o->set_dim(dimensions).set_stride(std::vector<int64_t>(os,os+4));
        check(graph->validate()); check(graph->build_operation_graph(handle));
        check(graph->create_execution_plans({fe::HeurMode_t::A}));
        check(graph->check_support(handle)); check(graph->build_plans(handle));
        check(graph->get_plan_name(tag));
        reserve_workspace();
    }
    void reserve_workspace() {
        size_t bytes = graph->get_workspace_size();
        if (bytes && !workspace && cudaMalloc(&workspace,bytes) != cudaSuccess)
            throw std::runtime_error("SDPA workspace allocation failed");
    }
    void release_workspace() {
        if (!workspace) return;
        cudaStream_t stream; check(cudnnGetStream(handle,&stream));
        if (cudaStreamSynchronize(stream)!=cudaSuccess) throw std::runtime_error("SDPA workspace release sync failed");
        if (cudaFree(workspace)!=cudaSuccess) throw std::runtime_error("SDPA workspace release failed");
        workspace=nullptr;
    }
    ~State() { if (workspace) cudaFree(workspace); graph.reset(); if(handle)cudnnDestroy(handle); }
};
using Key = std::array<int64_t,20>;
std::map<Key,std::unique_ptr<State>>& states() { thread_local auto* cache = new std::map<Key,std::unique_ptr<State>>; return *cache; }
struct Stats {std::map<Key,uint64_t> ages;uint64_t clock=0,hits=0,builds=0,evictions=0;};
Stats& stats(){thread_local Stats value;return value;}

}
extern "C" const char* h3v_dense_error() { return error.c_str(); }
extern "C" int h3v_dense(const void*q,const void*k,const void*v,void*out,const int64_t*dims,
                         const int64_t*qs,const int64_t*ks,const int64_t*vs,const int64_t*os,
                         float scale,void*stream) {
    try {
        Key key; const int64_t* arrays[] = {dims,qs,ks,vs,os};
        for(int i=0;i<5;++i)for(int j=0;j<4;++j)key[i*4+j]=arrays[i][j];
        auto& cache=states();
        auto& counts=stats();
        if(!cache.count(key)){
            auto next=std::make_unique<State>(dims,qs,ks,vs,os);
            if(cache.size()>=16){
                if(cudaStreamSynchronize(static_cast<cudaStream_t>(stream))!=cudaSuccess)throw std::runtime_error("video SDPA cache eviction sync failed");
                auto oldest=counts.ages.begin();for(auto i=counts.ages.begin();i!=counts.ages.end();++i)if(i->second<oldest->second)oldest=i;
                cache.erase(oldest->first);counts.ages.erase(oldest);++counts.evictions;
            }
            cache.emplace(key,std::move(next));++counts.builds;
        }else ++counts.hits;
        counts.ages[key]=++counts.clock;
        auto& s=*cache.at(key); check(cudnnSetStream(s.handle,static_cast<cudaStream_t>(stream)));
        s.reserve_workspace();
        std::unordered_map<int64_t,void*>pack{{0,const_cast<void*>(q)},{1,const_cast<void*>(k)},
            {2,const_cast<void*>(v)},{3,out},{5,&scale}};
        check(s.graph->execute(s.handle,pack,s.workspace));error.clear();return 0;
    }catch(const std::exception&e){error=e.what();return -1;}
}
extern "C" int h3v_dense_prepare(const int64_t*dims,const int64_t*qs,const int64_t*ks,const int64_t*vs,const int64_t*os){
    try {
        Key key;const int64_t* arrays[]={dims,qs,ks,vs,os};
        for(int i=0;i<5;++i)for(int j=0;j<4;++j)key[i*4+j]=arrays[i][j];
        auto& cache=states();auto& counts=stats();
        if(!cache.count(key)){
            if(cache.size()>=16)return -1;
            cache.emplace(key,std::make_unique<State>(dims,qs,ks,vs,os));++counts.builds;
        }
        cache.at(key)->reserve_workspace();
        counts.ages[key]=++counts.clock;error.clear();return 0;
    }catch(const std::exception&e){error=e.what();return -1;}
}
extern "C" int h3v_dense_cleanup(){
    if(cudaDeviceSynchronize()!=cudaSuccess)return -1;
    states().clear();stats().ages.clear();return 0;
}

extern "C" int h3v_dense_cache_stats(uint64_t* out){
 const auto& c=stats();uint64_t values[]={states().size(),16,c.hits,c.builds,c.evictions};
 for(int i=0;i<5;++i)out[i]=values[i];return 0;
}
extern "C" uint64_t h3v_dense_workspace_bytes(){
 uint64_t bytes=0;for(const auto& entry:states())if(entry.second->workspace)bytes+=entry.second->graph->get_workspace_size();return bytes;
}
extern "C" int h3v_dense_release_workspace(){
 try{for(auto& entry:states())entry.second->release_workspace();error.clear();return 0;}
 catch(const std::exception& e){error=e.what();return -1;}
}
