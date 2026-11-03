// Generalized batch/group dimensions of the existing H3 BF16 GroupNorm port.
// Welford traversal follows PyTorch cf30153 group_norm_kernel.cu,
// SharedReduceOps.h and block_reduce.cuh; BSD license in licenses/PYTORCH-LICENSE.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cmath>
#include <cstdint>
namespace {
template<class T> __device__ float read(T v){return float(v);}
template<class T> __device__ T cast(float v){return T(v);}
template<> __device__ __nv_bfloat16 cast(float v){return __float2bfloat16_rn(v);}
struct W {float mean,m2;int64_t n;float nf;};
__device__ W combine(W a,W z){
    if(!a.nf)return z;if(!z.nf)return a;
    float delta=z.mean-a.mean,count=a.nf+z.nf,ratio=z.nf/count;
    return {a.mean+delta*ratio,a.m2+z.m2+delta*delta*a.nf*ratio,-1,count};
}
__device__ W warp_reduce(W v){
    for(int d=16;d;d>>=1)v=combine(v,{__shfl_down_sync(0xffffffff,v.mean,d),
        __shfl_down_sync(0xffffffff,v.m2,d),__shfl_down_sync(0xffffffff,v.n,d),__shfl_down_sync(0xffffffff,v.nf,d)});
    return v;
}
template<class T> __global__ void moments(const T* x,float* stats,int64_t count,float eps){
    W v{0,0,0,0};
    for(int64_t j=threadIdx.x;j<count;j+=blockDim.x){
        float value=read(x[int64_t(blockIdx.x)*count+j]),delta=value-v.mean;
        int64_t n=v.n+1;float nf=float(n),mean=v.mean+delta/nf;
        v={mean,v.m2+delta*(value-mean),n,nf};
    }
    __shared__ W parts[16];int lane=threadIdx.x%32,warp=threadIdx.x/32;
    v=warp_reduce(v);if(!lane)parts[warp]=v;__syncthreads();
    if(!warp){v=threadIdx.x<int(blockDim.x)/32?parts[lane]:W{0,0,0,0};v=warp_reduce(v);
        if(!lane){stats[2*blockIdx.x]=v.mean;stats[2*blockIdx.x+1]=rsqrtf(v.m2/v.nf+eps);}}
}
template<class T> __global__ void normalize(const T* x,const T* w,const T* bias,T* y,
    const float* stats,int64_t count,int64_t spatial,int channels,int groups){
    for(int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<count;i+=int64_t(gridDim.x)*blockDim.x){
        int channel=int((i/spatial)%channels);int64_t group=(i/(channels*spatial))*groups+channel/(channels/groups);
        float a=stats[2*group+1]*(w?read(w[channel]):1.0f);
        float b=-a*stats[2*group]+(bias?read(bias[channel]):0.0f);
        y[i]=cast<T>(a*read(x[i])+b);
    }
}
}
extern "C" int ltx25_group_norm(const void* x,const void* w,const void* bias,void* y,
    void* stats,int64_t batch,int channels,int64_t spatial,int groups,float eps,int dtype,cudaStream_t stream){
    if(batch<1||groups<1||channels%groups||spatial<1)return cudaErrorInvalidValue;
    int threads=(channels/groups)*spatial<512?32:512;
    int64_t n=batch*channels*spatial;unsigned blocks=unsigned((n+255)/256>65535?65535:(n+255)/256);
    if(dtype==0){
        eps=__bfloat162float(__float2bfloat16_rn(eps));
        moments<<<batch*groups,threads,0,stream>>>((const __nv_bfloat16*)x,(float*)stats,(channels/groups)*spatial,eps);
        normalize<<<blocks,256,0,stream>>>((const __nv_bfloat16*)x,(const __nv_bfloat16*)w,(const __nv_bfloat16*)bias,
            (__nv_bfloat16*)y,(float*)stats,n,spatial,channels,groups);
    }else if(dtype==1){
        moments<<<batch*groups,threads,0,stream>>>((const float*)x,(float*)stats,(channels/groups)*spatial,eps);
        normalize<<<blocks,256,0,stream>>>((const float*)x,(const float*)w,(const float*)bias,(float*)y,
            (float*)stats,n,spatial,channels,groups);
    }else return cudaErrorInvalidValue;
    return cudaGetLastError();
}
