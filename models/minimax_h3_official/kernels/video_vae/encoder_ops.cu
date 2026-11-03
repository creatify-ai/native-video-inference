// FP32 specializations of the pinned eager video VAE primitives.
// GroupNorm reductions preserve PyTorch cf30153 group_norm_kernel.cu,
// SharedReduceOps.h and block_reduce.cuh order. BSD license in vendor_reference.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <math.h>

struct Welford { float mean,m2; int64_t n; float nf; };
__device__ Welford combine(Welford a,Welford b){
    if(a.nf==0)return b;if(b.nf==0)return a;
    float delta=b.mean-a.mean,new_count=a.nf+b.nf,nb_over_n=b.nf/new_count;
    return {a.mean+delta*nb_over_n,a.m2+b.m2+delta*delta*a.nf*nb_over_n,-1,new_count};
}
__device__ Welford reduce_warp(Welford v){
    #pragma unroll
    for(int offset=16;offset>0;offset>>=1){
        Welford other{__shfl_down_sync(0xffffffff,v.mean,offset),__shfl_down_sync(0xffffffff,v.m2,offset),
            __shfl_down_sync(0xffffffff,v.n,offset),__shfl_down_sync(0xffffffff,v.nf,offset)};
        v=combine(v,other);
    }return v;
}
__global__ void moments(int64_t count,float eps,const float*x,float*mean,float*rstd){
    Welford v{0,0,0,0};
    for(int64_t j=threadIdx.x;j<count;j+=blockDim.x){
        float data=x[int64_t(blockIdx.x)*count+j];int64_t n=v.n+1;float nf=float(n);
        float delta=data-v.mean,new_mean=v.mean+delta/nf,new_delta=data-new_mean;
        v={new_mean,v.m2+delta*new_delta,n,nf};
    }
    if(blockDim.x<=32)v=reduce_warp(v);
    else{
        __shared__ Welford values[32];int tid=threadIdx.x,lane=tid%32,warp=tid/32;
        v=reduce_warp(v);__syncthreads();if(lane==0)values[warp]=v;__syncthreads();
        v=tid<int(blockDim.x)/32?values[lane]:Welford{0,0,0,0};if(warp==0)v=reduce_warp(v);
    }
    if(threadIdx.x==0){mean[blockIdx.x]=v.mean;rstd[blockIdx.x]=rsqrtf(v.m2/v.nf+eps);}
}
__global__ void norm_params(int64_t n,int64_t c,int64_t groups,const float*mean,const float*rstd,
    const float*gamma,const float*beta,float*a,float*b){
    int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i<n*c){int64_t group=i/(c/groups),channel=i%c;float scale=rstd[group]*gamma[channel];
        a[i]=scale;b[i]=-scale*mean[group]+beta[channel];}
}
__global__ void norm_output(const float*x,const float*a,const float*b,float*y,int64_t count,int64_t spatial){
    for(int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<count;i+=int64_t(gridDim.x)*blockDim.x)y[i]=a[i/spatial]*x[i]+b[i/spatial];
}
extern "C" int h3v_group_norm(const void*x,const void*gamma,const void*beta,void*y,void*mean,void*rstd,
    void*scratch,int64_t n,int64_t c,int64_t spatial,int64_t groups,float eps,void*stream){
    auto s=static_cast<cudaStream_t>(stream);float*a=static_cast<float*>(scratch),*b=a+n*c;
    int threads=(c/groups)*spatial<512?32:512;
    moments<<<n*groups,threads,0,s>>>((c/groups)*spatial,eps,static_cast<const float*>(x),static_cast<float*>(mean),static_cast<float*>(rstd));
    norm_params<<<(n*c+255)/256,256,0,s>>>(n,c,groups,static_cast<float*>(mean),static_cast<float*>(rstd),static_cast<const float*>(gamma),static_cast<const float*>(beta),a,b);
    norm_output<<<(n*c*spatial+255)/256,256,0,s>>>(static_cast<const float*>(x),a,b,static_cast<float*>(y),n*c*spatial,spatial);
    return cudaGetLastError();
}
struct Pad { int64_t shape[5],stride[5],out[5],pad[6]; };
__global__ void padding(const float*x,float*y,int64_t count,Pad p,int reflect){
    for(int64_t index=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;index<count;index+=int64_t(gridDim.x)*blockDim.x){
        int64_t left=index,source=0;bool inside=true;
        #pragma unroll
        for(int d=4;d>=0;--d){int64_t c=left%p.out[d];left/=p.out[d];if(d>=2)c-=p.pad[(4-d)*2];
            if(c<0||c>=p.shape[d]){if(reflect){c=c<0?-c:2*p.shape[d]-2-c;}else inside=false;}
            source+=c*p.stride[d];}
        y[index]=inside?x[source]:0.0f;
    }
}
extern "C" int h3v_pad(const void*x,void*y,const int64_t*shape,const int64_t*stride,const int64_t*out,
    const int64_t*pad,int reflect,void*stream){
    Pad p;int64_t count=1;for(int i=0;i<5;++i){p.shape[i]=shape[i];p.stride[i]=stride[i];p.out[i]=out[i];count*=out[i];}for(int i=0;i<6;++i)p.pad[i]=pad[i];
    padding<<<(count+255)/256,256,0,static_cast<cudaStream_t>(stream)>>>(static_cast<const float*>(x),static_cast<float*>(y),count,p,reflect);return cudaGetLastError();
}
__global__ void pointwise(const float*x,const float*z,float*y,int64_t count,int op,int64_t spatial,int64_t channels){
    for(int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<count;i+=int64_t(gridDim.x)*blockDim.x){
        float v=x[i];if(op==0)y[i]=v/(1.0f+expf(-v));else if(op==1)y[i]=__fadd_rn(v,z[i]);else y[i]=__fadd_rn(v,z[(i/spatial)%channels]);
    }
}
extern "C" int h3v_pointwise(const void*x,const void*z,void*y,int64_t count,int op,int64_t spatial,int64_t channels,void*stream){
    pointwise<<<(count+255)/256,256,0,static_cast<cudaStream_t>(stream)>>>(static_cast<const float*>(x),static_cast<const float*>(z),static_cast<float*>(y),count,op,spatial,channels);return cudaGetLastError();
}
__global__ void cast_half(const float*x,__half*y,int64_t n){
    for(int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=int64_t(gridDim.x)*blockDim.x)y[i]=__float2half_rn(x[i]);
}
extern "C" int h3v_cast_half(const void*x,void*y,int64_t n,void*stream){
    cast_half<<<(n+255)/256,256,0,static_cast<cudaStream_t>(stream)>>>(static_cast<const float*>(x),static_cast<__half*>(y),n);return cudaGetLastError();
}
__global__ void bias_half(__half*x,const __half*b,int64_t n,int64_t spatial,int64_t channels){
    for(int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=int64_t(gridDim.x)*blockDim.x)x[i]=__hadd(x[i],b[(i/spatial)%channels]);
}
extern "C" int h3v_bias_half(void*x,const void*b,int64_t n,int64_t spatial,int64_t channels,void*stream){
    bias_half<<<(n+255)/256,256,0,static_cast<cudaStream_t>(stream)>>>(static_cast<__half*>(x),static_cast<const __half*>(b),n,spatial,channels);return cudaGetLastError();
}
