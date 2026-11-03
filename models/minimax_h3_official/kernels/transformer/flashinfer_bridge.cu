// Native adapter for the unmodified FlashInfer 0.7.0.post1 SM100 blk64 kernel.
// Kernel math is supplied by a verified CuTe AOT export; this file only moves bytes
// and constructs SGLang's heterogeneous query-block indices.
#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>
#include <cmath>
#include <atomic>
#include <mutex>
#include "h3_subblock_v07.h"
#define CHECK(call) do { int e=int(call); if(e) return e; } while(0)
namespace {
int pack_mode=0;
uint64_t baseline_calls=0,optimized_calls=0,served_calls=0;
h3_subblock_v07_Kernel_Module_t module;
std::once_flag init_once;
cudaError_t init_status=cudaSuccess;
std::mutex load_mutex;
std::atomic<uint64_t> loaded_devices{0};
int ensure_loaded(){
    std::call_once(init_once,[]{
        cudaLibrary_t* library=&module.module;
        struct {cudaLibrary_t** library;cudaError_t* ret;} args{&library,&init_status};
        _mlir_h3_subblock_v07_cuda_init(reinterpret_cast<void**>(&args));
    });
    CHECK(init_status);
    int device;CHECK(cudaGetDevice(&device));
    if(device<0||device>=64)return cudaErrorInvalidDevice;
    const uint64_t bit=uint64_t(1)<<device;
    if(loaded_devices.load(std::memory_order_acquire)&bit)return 0;
    std::lock_guard<std::mutex> lock(load_mutex);
    if(loaded_devices.load(std::memory_order_relaxed)&bit)return 0;
    cudaLibrary_t* library=&module.module;cudaError_t status=cudaSuccess;
    struct {cudaLibrary_t** library;int32_t* device;cudaError_t* ret;} args{&library,&device,&status};
    _mlir_h3_subblock_v07_cuda_load_to_device(reinterpret_cast<void**>(&args));
    CHECK(status);loaded_devices.fetch_or(bit,std::memory_order_release);return 0;
}
size_t align256(size_t n){return (n+255)/256*256;}
struct Plan {int blocks;size_t output,lse,index,counts,bytes;};
int plan(int s,int h,Plan& p){
    if(s<1||s>524288||h<1||h>128||int64_t(s)*h*128>=INT32_MAX)return cudaErrorInvalidValue;
    p.blocks=(s+63)/64;
    if(int64_t(h)*p.blocks*p.blocks>=INT32_MAX)return cudaErrorInvalidValue;
    p.output=align256(size_t(s)*h*128*2);p.lse=align256(size_t(s)*h*4);
    p.index=align256(size_t(h)*p.blocks*p.blocks*4);p.counts=align256(size_t(h)*p.blocks*4);
    p.bytes=p.output+p.lse+p.index+p.counts;return 0;
}
unsigned grid(size_t n){return unsigned((n+255)/256<65535?(n+255)/256:65535);}
__global__ void expand(const int* selected,const int* mask,int* index,int* counts,int blocks,int kept,size_t n){
    for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=size_t(gridDim.x)*blockDim.x){
        size_t row=i/blocks;int col=i%blocks;bool sparse=mask[row%blocks]!=0;
        index[i]=sparse&&col<kept?selected[row*kept+col]:col;
        if(col==0)counts[row]=sparse?kept:blocks;
    }
}
__global__ void unpack(const uint16_t* src,uint16_t* dst,int s,int h,size_t n){
    for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=size_t(gridDim.x)*blockDim.x){
        int d=i%128,head=(i/128)%h;size_t t=i/(size_t(h)*128);
        dst[i]=src[(size_t(head)*s+t)*128+d];
    }
}
__global__ void unpack16(const uint4* src,uint4* dst,int s,int h,size_t n){
    for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=size_t(gridDim.x)*blockDim.x){
        int d=i%16,head=(i/16)%h;size_t t=i/(size_t(h)*16);
        dst[i]=src[(size_t(head)*s+t)*16+d];
    }
}
template<class T> T natural(const void* data,int s,int h){
    return T{const_cast<void*>(data),{1,h,s,128},{int64_t(s)*h*128,128,int64_t(h)*128}};
}
}
extern "C" int h3_check_device(){
    int d,major,minor;CHECK(cudaGetDevice(&d));
    CHECK(cudaDeviceGetAttribute(&major,cudaDevAttrComputeCapabilityMajor,d));
    CHECK(cudaDeviceGetAttribute(&minor,cudaDevAttrComputeCapabilityMinor,d));
    return major==10&&minor==0?0:cudaErrorNotSupported;
}
extern "C" int h3_bsa_pack_set_mode(int mode){
    if(mode!=0&&mode!=1)return cudaErrorInvalidValue;
    pack_mode=mode;baseline_calls=optimized_calls=0;return 0;
}
extern "C" int h3_bsa_set_kernel(int kernel){if(kernel!=0)return cudaErrorNotSupported;served_calls=0;return 0;}
extern "C" int h3_bsa_kernel_counts(uint64_t* served,uint64_t* exact){
    if(!served||!exact)return cudaErrorInvalidValue;*served=served_calls;*exact=0;return 0;
}
extern "C" int h3_bsa_pack_counts(uint64_t* baseline,uint64_t* optimized,uint64_t* fallback){
    if(!baseline||!optimized||!fallback)return cudaErrorInvalidValue;
    *baseline=baseline_calls;*optimized=optimized_calls;*fallback=0;return 0;
}
extern "C" int h3_bsa_masked_workspace_bytes(int s,int h,size_t* bytes){
    if(!bytes)return cudaErrorInvalidValue;Plan p;CHECK(plan(s,h,p));*bytes=p.bytes;return 0;
}
extern "C" int h3_bsa_workspace_bytes(int s,int h,int kept,size_t* bytes){
    if(kept<1||kept>(s+63)/64)return cudaErrorInvalidValue;
    return h3_bsa_masked_workspace_bytes(s,h,bytes);
}
extern "C" int h3_bsa_forward_masked(const void* q,const void* k,const void* v,const int* selected,
    const int* sizes,const int* mask,void* out,void* workspace,size_t bytes,
    int s,int h,int kept,float scale,cudaStream_t stream){
    Plan p;CHECK(plan(s,h,p));
    auto aligned=[](const void* p){return p&&uintptr_t(p)%16==0;};
    if(!aligned(q)||!aligned(k)||!aligned(v)||!aligned(out)||!aligned(selected)||!aligned(sizes)||!mask||
        !workspace||uintptr_t(workspace)%256||bytes<p.bytes||kept<1||kept>p.blocks||
        !std::isfinite(scale)||scale<=0)return cudaErrorInvalidValue;
    CHECK(h3_check_device());CHECK(ensure_loaded());
    auto* base=static_cast<char*>(workspace);auto* output=base;
    auto* lse=base+p.output;auto* index=reinterpret_cast<int*>(lse+p.lse);
    auto* counts=reinterpret_cast<int*>(reinterpret_cast<char*>(index)+p.index);
    size_t n=size_t(h)*p.blocks*p.blocks;
    expand<<<grid(n),256,0,stream>>>(selected,mask,index,counts,p.blocks,kept,n);CHECK(cudaGetLastError());
    auto Q=natural<h3_subblock_v07_Tensor_mQ_t>(q,s,h);
    auto K=natural<h3_subblock_v07_Tensor_mK_t>(k,s,h);
    auto V=natural<h3_subblock_v07_Tensor_mV_t>(v,s,h);
    h3_subblock_v07_Tensor_mO_t O{output,{1,h,s,128},{int64_t(h)*s*128,int64_t(s)*128,128}};
    h3_subblock_v07_Tensor_mLSE_t L{lse,{1,h,s},{int64_t(h)*s,s}};
    h3_subblock_v07_Tensor_mBlockIndex_t I{index,{1,h,p.blocks,p.blocks},
        {int64_t(h)*p.blocks*p.blocks,int64_t(p.blocks)*p.blocks,p.blocks}};
    h3_subblock_v07_Tensor_mBlockSizes_t S{const_cast<int*>(sizes),{p.blocks}};
    h3_subblock_v07_Tensor_mBlockNums_t N{counts,{1,h,p.blocks},{int64_t(h)*p.blocks,p.blocks}};
    CHECK(cute_dsl_h3_subblock_v07_wrapper(&module,&Q,&K,&V,&O,&L,scale,&I,&S,0,&N,stream));
    CHECK(cudaGetLastError());++served_calls;
    n=size_t(s)*h*128;
    if(pack_mode){++optimized_calls;unpack16<<<grid(n/8),256,0,stream>>>(reinterpret_cast<const uint4*>(output),static_cast<uint4*>(out),s,h,n/8);}
    else {++baseline_calls;unpack<<<grid(n),256,0,stream>>>(reinterpret_cast<const uint16_t*>(output),static_cast<uint16_t*>(out),s,h,n);}
    return int(cudaGetLastError());
}
