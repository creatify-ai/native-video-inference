// Literal outer operations from pinned AutoencoderKLMiniMaxH3 and pipeline.
// Preserve every original FP32/FP16 rounding; see outer-gate.py for exact checks.
#include "outer.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>
#include "pixels.cuh"

namespace {
constexpr int B=256;
__device__ float pixel_mean(int c){return c==0?0.485f:(c==1?0.456f:0.406f);}
__device__ float pixel_std(int c){return c==0?0.229f:(c==1?0.224f:0.225f);}
__global__ void encode_prepare(const uint8_t* rgb,float* out,int channels_last){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=3*768*768)return;
    int c=i%3;int p=i/3;
    float x=__fmul_rn(float(rgb[i]),1.0f/255.0f);
    x=__fdiv_rn(__fsub_rn(x,pixel_mean(c)),pixel_std(c));
    out[channels_last?i:c*768*768+p]=x;
}
__global__ void extract_f32(const float* src,float* dst,int C,int T,int H,int W,
    int t0,int y0,int x0,int t,int h,int w,int last){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=C*t*h*w)return;
    int x=i%w,y=i/w%h,z=i/(w*h)%t,c=i/(w*h*t);
    int source=last?(((z+t0)*H+y+y0)*W+x+x0)*C+c:((c*T+z+t0)*H+y+y0)*W+x+x0;
    int destination=last?((z*h+y)*w+x)*C+c:i;dst[destination]=src[source];
}
template<class T> __device__ T blend(T a,T b,int p,int extent);
template<> __device__ float blend(float a,float b,int p,int extent){
    // PyTorch div-by-host-scalar uses a FP32 reciprocal multiply.
    float wb=__fmul_rn(float(p),__fdiv_rn(1.0f,float(extent)));
    float wa=__fsub_rn(1.0f,wb);
    return __fadd_rn(__fmul_rn(a,wa),__fmul_rn(b,wb));
}
template<> __device__ __half blend(__half a,__half b,int p,int extent){
    __half wb=__float2half_rn(__fmul_rn(float(p),__fdiv_rn(1.0f,float(extent))));
    __half wa=__hsub_rn(__float2half_rn(1.0f),wb);
    // The _rn forms also prohibit contraction into half FMA. Plain __hmul/
    // __hadd can fuse despite returning __half, losing PyTorch's tensor-boundary
    // rounding between its two multiply kernels and final add kernel.
    return __hadd_rn(__hmul_rn(a,wa),__hmul_rn(b,wb));
}
template<class T> __global__ void stitch(const T* tiles,T* out,int C,int frames,int K){
    int width=3*K,i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=C*frames*width*width)return;
    int x=i%width,y=i/width%width,ct=i/(width*width);
    // Original _split_tiles distributes four 16-pixel increments round-robin.
    int unit=K/16;int starts[4]={0,10*unit,21*unit,32*unit};
    int overlaps[3]={6*unit,5*unit,5*unit};
    int r=y<starts[1]?0:y<starts[2]?1:y<starts[3]?2:3;
    int c=x<starts[1]?0:x<starts[2]?1:x<starts[3]?2:3;
    int ly=y-starts[r],lx=x-starts[c],tile_elements=C*frames*K*K;
    int local=ct*K*K+ly*K+lx;
    T value=tiles[(r*4+c)*tile_elements+local];
    if(r>0&&ly<overlaps[r-1]){
        T a=tiles[((r-1)*4+c)*tile_elements+ct*K*K+(K-overlaps[r-1]+ly)*K+lx];
        value=blend(a,value,ly,overlaps[r-1]);
    }
    if(c>0&&lx<overlaps[c-1]){
        // Horizontal blend reads the original previous tile, not a vertically
        // blended temporary; this is the released _stitch_tiles evaluation order.
        T a=tiles[(r*4+c-1)*tile_elements+ct*K*K+ly*K+(K-overlaps[c-1]+lx)];
        value=blend(a,value,lx,overlaps[c-1]);
    }
    out[i]=value;
}
__global__ void denorm(const float* src,const float* mean,const float* std,float* dst){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=24*102*48*48)return;
    int c=i/(102*48*48);dst[i]=__fadd_rn(__fmul_rn(src[i],std[c]),mean[c]);
}
__global__ void posterior(const float* moments,const float* noise,const float* mean,const float* std,float* dst){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=24*48*48)return;
    float logvar=fminf(20.0f,fmaxf(-30.0f,moments[i+24*48*48]));
    float sigma=expf(__fmul_rn(0.5f,logvar));
    float latent=__fadd_rn(moments[i],__fmul_rn(sigma,noise[i]));
    float rounded=__half2float(__float2half_rn(latent));
    int c=i/(48*48);dst[i]=__fdiv_rn(__fsub_rn(rounded,mean[c]),std[c]);
}
__global__ void temporal_current(const __half* clip,const __half* overlap,__half* output,int chunk){
    constexpr int P=768*768;int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=3*17*P)return;
    int p=i%P,t=i/P%17,c=i/(17*P);__half value=clip[(c*28+t+3)*P+p];
    if(chunk>0&&t<5)value=blend(overlap[(c*5+t)*P+p],value,t,5);
    output[(c*345+chunk*17+t)*P+p]=value;
}
__global__ void temporal_tail(const __half* clip,__half* overlap,__half* output,int chunk){
    constexpr int P=768*768;int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=3*5*P)return;
    int p=i%P,t=i/P%5,c=i/(5*P);__half value=clip[(c*28+t+23)*P+p];
    overlap[i]=value;if(chunk==19)output[(c*345+340+t)*P+p]=value;
}
__global__ void pt8(const __half* input,uint8_t* output){
    constexpr int P=768*768;int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=3*345*P)return;
    int c=i%3,p=i/3,t=p/P;
    output[i]=h3_public_rgb_pixel(input[(c*345+t)*P+p%P],c);
}
__global__ void half_nonfinite(const __half* input,uint32_t* counter,int64_t count){
    int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i<count&&(__half_as_ushort(input[i])&0x7c00u)==0x7c00u)atomicAdd(counter,1u);
}
}
extern "C" int h3v_encode_prepare(const uint8_t* rgb,float* pixels,int channels_last,void* stream){
    if(channels_last!=0&&channels_last!=1)return cudaErrorInvalidValue;
    encode_prepare<<<(3*768*768+B-1)/B,B,0,(cudaStream_t)stream>>>(rgb,pixels,channels_last);return cudaGetLastError();
}
extern "C" int h3v_extract_f32(const float* input,float* tile,int C,int T,int H,int W,
    int t0,int y0,int x0,int t,int h,int w,int channels_last,void* stream){
    if(C<=0||T<=0||H<=0||W<=0||t<=0||h<=0||w<=0||t0<0||y0<0||x0<0||t0+t>T||y0+h>H||x0+w>W||(channels_last!=0&&channels_last!=1))return cudaErrorInvalidValue;
    extract_f32<<<(C*t*h*w+B-1)/B,B,0,(cudaStream_t)stream>>>(input,tile,C,T,H,W,t0,y0,x0,t,h,w,channels_last);return cudaGetLastError();
}
extern "C" int h3v_stitch(const void* tiles,void* output,int C,int T,int K,int dtype,void* stream){
    if(!((K==16&&C==48&&T==1)||(K==256&&C==3&&T==28))||(dtype!=0&&dtype!=1))return cudaErrorInvalidValue;
    int grid=(C*T*9*K*K+B-1)/B;
    if(dtype==0)stitch<<<grid,B,0,(cudaStream_t)stream>>>((const float*)tiles,(float*)output,C,T,K);
    else stitch<<<grid,B,0,(cudaStream_t)stream>>>((const __half*)tiles,(__half*)output,C,T,K);
    return cudaGetLastError();
}
extern "C" int h3v_decode_denorm(const float* input,const float* mean,const float* std,float* output,void* stream){
    denorm<<<(24*102*48*48+B-1)/B,B,0,(cudaStream_t)stream>>>(input,mean,std,output);return cudaGetLastError();
}
extern "C" int h3v_posterior(const float* moments,const float* noise,const float* mean,const float* std,float* output,void* stream){
    posterior<<<(24*48*48+B-1)/B,B,0,(cudaStream_t)stream>>>(moments,noise,mean,std,output);return cudaGetLastError();
}
extern "C" int h3v_temporal_merge(const void* clip,void* overlap,void* output,int chunk,void* stream){
    if(chunk<0||chunk>=20)return cudaErrorInvalidValue;
    temporal_current<<<(3*17*768*768+B-1)/B,B,0,(cudaStream_t)stream>>>((const __half*)clip,(const __half*)overlap,(__half*)output,chunk);
    auto code=cudaGetLastError();if(code!=cudaSuccess)return code;
    temporal_tail<<<(3*5*768*768+B-1)/B,B,0,(cudaStream_t)stream>>>((const __half*)clip,(__half*)overlap,(__half*)output,chunk);return cudaGetLastError();
}
extern "C" int h3v_pt8(const void* input,uint8_t* output,void* stream){
    pt8<<<(3*345*768*768+B-1)/B,B,0,(cudaStream_t)stream>>>((const __half*)input,output);return cudaGetLastError();
}
extern "C" int h3v_half_nonfinite(const void* input,uint32_t* counter,int64_t count,void* stream){
    if(count<0||count>int64_t(UINT32_MAX))return cudaErrorInvalidValue;
    if(count==0)return cudaSuccess;
    half_nonfinite<<<(count+B-1)/B,B,0,(cudaStream_t)stream>>>((const __half*)input,counter,count);return cudaGetLastError();
}
