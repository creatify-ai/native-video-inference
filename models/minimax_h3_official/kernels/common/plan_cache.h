// Existing H3 environment and on-disk plan identity over the generic plan cache.
#pragma once
#include "../../../../core_infra/kernels/plan_cache.h"
namespace h3_plan_cache {
using Graph=infra_plan_cache::Graph;
inline infra_plan_cache::Configuration configuration() {
    const char* mode=std::getenv("H3_CUDNN_PLAN_CACHE_MODE");
    const char* directory=std::getenv("H3_CUDNN_PLAN_CACHE_DIR");
    return {mode?mode:"off",directory?directory:"","h3-exact-plan-v1","H3_PLAN_CACHE"};
}
inline uint64_t checksum(const std::string& value){return infra_plan_cache::checksum(value);}
inline std::string mode(){return configuration().mode;}
inline std::string full_key(const std::string& shape){return infra_plan_cache::full_key(configuration(),shape);}
inline std::string filename(const std::string& key){return infra_plan_cache::filename(configuration(),key);}
inline bool load(std::unique_ptr<Graph>& graph,cudnnHandle_t handle,const std::string& shape,std::string* tag=nullptr){return infra_plan_cache::load(configuration(),graph,handle,shape,tag);}
inline void save(Graph& graph,const std::string& shape){infra_plan_cache::save(configuration(),graph,shape);}
}
