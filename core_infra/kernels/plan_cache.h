// Experimental cache of fully built, unchanged cuDNN execution plans.
// The controller publishes a bundle only after verifying its pinned SHA256.
#pragma once
#include <cuda_runtime_api.h>
#include <cudnn.h>
#include <cudnn_frontend.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>

namespace infra_plan_cache {
struct Configuration {std::string mode,directory,key_prefix,event_label;};
// Shared operators use one policy. Model-specific cache namespaces can retain
// their own Configuration while migrating to these framework defaults.
inline Configuration configuration() {
    const char* mode=std::getenv("NATIVE_CUDNN_PLAN_CACHE_MODE");
    const char* directory=std::getenv("NATIVE_CUDNN_PLAN_CACHE_DIR");
    return {mode?mode:"",directory?directory:"","native-cudnn-plan-v1","native_plan_cache"};
}
using Graph = cudnn_frontend::graph::Graph;
// The pinned cuDNN frontend deserializer uses temporary CUDA Graph capture
// and changes its handle stream. Concurrent restores on the same device can
// invalidate capture / fail with BAD_PARAM_STREAM_MISMATCH. Serialize only
// restore and validation; file reads, native builders and weight I/O stay free.
inline std::mutex& restore_mutex() { static std::mutex mutex; return mutex; }
inline std::string full_key(const Configuration& config,const std::string& shape) {
    int device=0,driver=0,runtime=0; cudaDeviceProp prop{};
    if(cudaGetDevice(&device)!=cudaSuccess || cudaGetDeviceProperties(&prop,device)!=cudaSuccess ||
       cudaDriverGetVersion(&driver)!=cudaSuccess || cudaRuntimeGetVersion(&runtime)!=cudaSuccess) return "";
    return config.key_prefix+":"+shape+":"+std::to_string(cudnnGetVersion())+":"+
        std::to_string(driver)+":"+std::to_string(runtime)+":"+prop.name+":"+
        std::to_string(prop.major)+":"+std::to_string(prop.minor);
}
inline uint64_t checksum(const std::string& s) {
    uint64_t h=14695981039346656037ULL; for(unsigned char c:s){h^=c;h*=1099511628211ULL;} return h;
}
inline std::string filename(const Configuration& config,const std::string& key) {
    const auto& dir=config.directory; if(dir.empty() || key.empty())return "";
    std::ostringstream name;name<<dir<<"/"<<std::hex<<std::setw(16)<<std::setfill('0')<<checksum(key)<<".plan";return name.str();
}
inline void event(const Configuration& config,const char* status,const std::string& key,double seconds) {
    std::fprintf(stderr,"%s status=%s key=%016llx seconds=%.6f\n",config.event_label.c_str(),status,
        static_cast<unsigned long long>(checksum(key)),seconds);
}
inline bool load(const Configuration& config,std::unique_ptr<Graph>& graph,cudnnHandle_t handle,const std::string& shape,std::string* loaded_tag=nullptr) {
    if(config.mode!="read")return false;
    auto start=std::chrono::steady_clock::now(); const auto key=full_key(config,shape);const auto path=filename(config,key);
    auto elapsed=[&]{return std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();};
    if(path.empty())return false;
    try {
        std::ifstream f(path,std::ios::binary);if(!f){event(config,"miss",key,elapsed());return false;}
        std::string saved,tag,line;std::getline(f,saved);std::getline(f,tag);std::getline(f,line);
        if(saved!=key || saved.size()>8192 || tag.size()>8192)throw std::runtime_error("metadata mismatch");
        const auto workspace=std::stoull(line);
        std::getline(f,line);const auto expected=std::stoull(line);
        std::vector<uint8_t> data((std::istreambuf_iterator<char>(f)),{});
        if(data.empty() || data.size()>128*1024*1024 || checksum(std::string(data.begin(),data.end()))!=expected)
            throw std::runtime_error("damaged plan");
        std::lock_guard<std::mutex> restore(restore_mutex());
        auto next=std::make_unique<Graph>();auto result=next->deserialize(handle,data);
        if(!result.is_good())throw std::runtime_error(result.get_message());
        // This pinned FE revision restores execution_plans but not engine_configs.
        // get_plan_name() indexes engine_configs unchecked after deserialize.
        // Use the tag stored with the fully SHA256-verified binary plan instead.
        if(next->get_execution_plan_count()!=1 || next->get_workspace_size()!=workspace)throw std::runtime_error("plan identity mismatch");
        if(loaded_tag)*loaded_tag=tag;
        graph=std::move(next);event(config,"hit",key,elapsed());return true;
    }catch(const std::exception& error){
        event(config,"rejected",key,elapsed());
        // A rejected plan retains the unchanged builder. Keep its concrete
        // cause visible so a missing/incompatible asset cannot masquerade as
        // a cache hit in cold-start verification.
        std::string reason=error.what();
        for(char& c:reason)if(c=='\n'||c=='\r')c=' ';
        std::fprintf(stderr,"%s rejection key=%016llx reason=%s\n",config.event_label.c_str(),
            static_cast<unsigned long long>(checksum(key)),reason.c_str());
        return false;
    }
}
// Threads and isolated DSOs can record the same compatible shape concurrently.
// Keep a unique same-directory file descriptor until close/atomic publication;
// the former process-ID-only temporary name could collide across those writers.
struct PendingFile {
    std::string path;
    std::FILE* stream=nullptr;
    explicit PendingFile(const std::string& destination):path(destination+".XXXXXX") {
        const int fd=mkstemp(path.data());
        if(fd<0)throw std::runtime_error("cache temporary create failed");
        stream=fdopen(fd,"wb");
        if(!stream){close(fd);std::remove(path.c_str());throw std::runtime_error("cache temporary stream failed");}
    }
    PendingFile(const PendingFile&)=delete;
    PendingFile& operator=(const PendingFile&)=delete;
    ~PendingFile(){if(stream)std::fclose(stream);if(!path.empty())std::remove(path.c_str());}
    void publish(const std::string& destination) {
        if(std::fflush(stream)!=0)throw std::runtime_error("cache flush failed");
        auto* closing=stream;stream=nullptr;
        if(std::fclose(closing)!=0 || std::rename(path.c_str(),destination.c_str())!=0)
            throw std::runtime_error("cache write failed");
        path.clear();
    }
};
inline void save(const Configuration& config,Graph& graph,const std::string& shape) {
    if(config.mode!="record")return;
    auto start=std::chrono::steady_clock::now();const auto key=full_key(config,shape);const auto path=filename(config,key);
    auto elapsed=[&]{return std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();};
    if(path.empty())return;
    try {
        std::vector<uint8_t> data;auto result=graph.serialize(data);
        if(!result.is_good())throw std::runtime_error(result.get_message());
        std::string tag;result=graph.get_plan_name(tag);if(!result.is_good())throw std::runtime_error(result.get_message());
        PendingFile file(path);
        std::ostringstream metadata;
        metadata<<key<<'\n'<<tag<<'\n'<<graph.get_workspace_size()<<'\n'<<checksum(std::string(data.begin(),data.end()))<<'\n';
        const auto header=metadata.str();
        if(std::fwrite(header.data(),1,header.size(),file.stream)!=header.size() ||
            std::fwrite(data.data(),1,data.size(),file.stream)!=data.size())throw std::runtime_error("cache write failed");
        file.publish(path);
        event(config,"recorded",key,elapsed());
    }catch(const std::exception&){event(config,"record_failed",key,elapsed());}
}
}
