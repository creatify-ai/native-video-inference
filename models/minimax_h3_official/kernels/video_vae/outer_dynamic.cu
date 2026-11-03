// Shape-general forms of the already validated literal original outer operations.
// Original per-operation FP32 and FP16 rounding is retained; no fused blend FMA.
#include "outer_dynamic.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>
#include <limits.h>
#include "delivery.cuh"
#include "pixels.cuh"
namespace {
constexpr int B=256;
__device__ float pm(int c){return c==0?.485f:c==1?.456f:.406f;}
__device__ float ps(int c){return c==0?.229f:c==1?.224f:.225f;}
template<class T> __device__ T blend(T a,T b,int p,int n);
template<> __device__ float blend(float a,float b,int p,int n){
 float wb=__fmul_rn(float(p),__fdiv_rn(1.f,float(n))),wa=__fsub_rn(1.f,wb);
 return __fadd_rn(__fmul_rn(a,wa),__fmul_rn(b,wb));
}
template<> __device__ __half blend(__half a,__half b,int p,int n){
 __half wb=__float2half_rn(__fmul_rn(float(p),__fdiv_rn(1.f,float(n))));
 __half wa=__hsub_rn(__float2half_rn(1.f),wb);
 return __hadd_rn(__hmul_rn(a,wa),__hmul_rn(b,wb));
}
__global__ void prepare(const uint8_t* in,float* out,int P){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=3*P)return;
 int c=i%3,p=i/3;float x=__fmul_rn(float(in[i]),1.f/255.f);
 out[c*P+p]=__fdiv_rn(__fsub_rn(x,pm(c)),ps(c));
}
__global__ void extract(const float* in,float* out,int C,int T,int H,int W,int t0,int y0,int x0,int t,int h,int w){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=C*t*h*w)return;
 int x=i%w,y=i/w%h,z=i/(w*h)%t,c=i/(w*h*t);
 int source_t=min(z+t0,T-1); // exact original repeated-final-latent temporal padding
 out[i]=in[((int64_t(c)*T+source_t)*H+y+y0)*W+x+x0];
}
template<class T> __global__ void stitch(const T* tiles,T* out,int C,int F,int K,int H,int W,
 int ny,int nx,const int* ys,const int* xs,const int* yo,const int* xo){
 int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=int64_t(C)*F*H*W)return;
 int x=i%W,y=i/W%H,ct=i/(int64_t(W)*H),r=0,c=0;
 while(r+1<ny&&y>=ys[r+1])++r;while(c+1<nx&&x>=xs[c+1])++c;
 int ly=y-ys[r],lx=x-xs[c];int64_t elements=int64_t(C)*F*K*K;
 T v=tiles[(r*nx+c)*elements+int64_t(ct)*K*K+ly*K+lx];
 if(r>0&&ly<yo[r-1])v=blend(tiles[((r-1)*nx+c)*elements+int64_t(ct)*K*K+(K-yo[r-1]+ly)*K+lx],v,ly,yo[r-1]);
 if(c>0&&lx<xo[c-1])v=blend(tiles[(r*nx+c-1)*elements+int64_t(ct)*K*K+ly*K+K-xo[c-1]+lx],v,lx,xo[c-1]);
 out[i]=v;
}
__global__ void denorm(const float* in,const float* mean,const float* std,float* out,int64_t P){
 int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=24*P)return;
 out[i]=__fadd_rn(__fmul_rn(in[i],std[i/P]),mean[i/P]);
}
__global__ void posterior(const float* in,const float* noise,const float* mean,const float* std,float* out,int64_t P){
 int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=24*P)return;
 float logvar=fminf(20.f,fmaxf(-30.f,in[i+24*P]));
 float sigma=expf(__fmul_rn(.5f,logvar));
 float sample=__fadd_rn(in[i],__fmul_rn(sigma,noise[i]));
 out[i]=__fdiv_rn(__fsub_rn(__half2float(__float2half_rn(sample)),mean[i/P]),std[i/P]);
}
__global__ void current(const __half* clip,const __half* prev,__half* out,int chunk,int P,int F){
 int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=int64_t(3)*17*P)return;
 int p=i%P,t=i/P%17,c=i/(int64_t(17)*P),target=chunk*17+t;
 if(target>=F)return;
 __half v=clip[(int64_t(c)*28+t+3)*P+p];if(chunk>0&&t<5)v=blend(prev[(c*5+t)*P+p],v,t,5);
 out[(int64_t(c)*F+target)*P+p]=v;
}
__global__ void tail(const __half* clip,__half* prev,__half* out,int chunk,int chunks,int P,int F){
 int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=int64_t(3)*5*P)return;
 int p=i%P,t=i/P%5,c=i/(int64_t(5)*P),target=chunks*17+t;
 __half v=clip[(int64_t(c)*28+t+23)*P+p];prev[i]=v;
 if(chunk==chunks-1&&target<F)out[(int64_t(c)*F+target)*P+p]=v;
}
__global__ void pt8(const __half* in,uint8_t* out,int P,int F){
 int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=int64_t(3)*F*P)return;
 int c=i%3,p=(i/3)%P,t=i/(int64_t(3)*P);
 out[i]=h3_public_rgb_pixel(in[(int64_t(c)*F+t)*P+p],c);
}
bool extent(int64_t n){return n>0&&n<=INT_MAX;}
bool video_extent(int64_t n){return n>0&&n<=int64_t(UINT32_MAX);}
int grid(int64_t n){return int((n+B-1)/B);}
}
extern "C" int h3vg_encode_prepare(const uint8_t* in,float* out,int H,int W,void* s){
 if(H<=0||W<=0||!extent(int64_t(3)*H*W))return cudaErrorInvalidValue;
 prepare<<<grid(int64_t(3)*H*W),B,0,(cudaStream_t)s>>>(in,out,H*W);return cudaGetLastError();}
extern "C" int h3vg_extract(const float* in,float* out,int C,int T,int H,int W,int t0,int y0,int x0,int t,int h,int w,void* s){
 if(C<=0||T<=0||H<=0||W<=0||t<=0||h<=0||w<=0||t0<0||t0>=T||y0<0||x0<0||y0+h>H||x0+w>W||!extent(int64_t(C)*t*h*w))return cudaErrorInvalidValue;
 extract<<<grid(int64_t(C)*t*h*w),B,0,(cudaStream_t)s>>>(in,out,C,T,H,W,t0,y0,x0,t,h,w);return cudaGetLastError();}
extern "C" int h3vg_stitch(const void* in,void* out,int C,int F,int K,int H,int W,int ny,int nx,const int* ys,const int* xs,const int* yo,const int* xo,int dtype,void* s){
 if(C<=0||F<=0||K<=0||H<K||W<K||ny<=0||nx<=0||!extent(int64_t(C)*F*H*W)||(dtype!=0&&dtype!=1))return cudaErrorInvalidValue;
 if(dtype==0)stitch<<<grid(int64_t(C)*F*H*W),B,0,(cudaStream_t)s>>>((const float*)in,(float*)out,C,F,K,H,W,ny,nx,ys,xs,yo,xo);
 else stitch<<<grid(int64_t(C)*F*H*W),B,0,(cudaStream_t)s>>>((const __half*)in,(__half*)out,C,F,K,H,W,ny,nx,ys,xs,yo,xo);return cudaGetLastError();}
extern "C" int h3vg_denorm(const float* in,const float* m,const float* st,float* out,int64_t P,void* s){
 if(!extent(24*P))return cudaErrorInvalidValue;denorm<<<grid(24*P),B,0,(cudaStream_t)s>>>(in,m,st,out,P);return cudaGetLastError();}
extern "C" int h3vg_posterior(const float* in,const float* n,const float* m,const float* st,float* out,int64_t P,void* s){
 if(!extent(24*P))return cudaErrorInvalidValue;posterior<<<grid(24*P),B,0,(cudaStream_t)s>>>(in,n,m,st,out,P);return cudaGetLastError();}
extern "C" int h3vg_temporal_merge(const void* in,void* prev,void* out,int chunk,int chunks,int H,int W,int F,void* s){
 if(chunks<=0||chunk<0||chunk>=chunks||H<=0||W<=0||F<=0||F>chunks*17+5||!extent(int64_t(3)*28*H*W)||!video_extent(int64_t(3)*F*H*W))return cudaErrorInvalidValue;
 current<<<grid(int64_t(3)*17*H*W),B,0,(cudaStream_t)s>>>((const __half*)in,(const __half*)prev,(__half*)out,chunk,H*W,F);
 auto code=cudaGetLastError();if(code!=cudaSuccess)return code;
 tail<<<grid(int64_t(3)*5*H*W),B,0,(cudaStream_t)s>>>((const __half*)in,(__half*)prev,(__half*)out,chunk,chunks,H*W,F);return cudaGetLastError();}
extern "C" int h3vg_pt8(const void* in,uint8_t* out,int H,int W,int F,void* s){
 if(H<=0||W<=0||F<=0||!extent(int64_t(H)*W)||!video_extent(int64_t(3)*F*H*W))return cudaErrorInvalidValue;
 pt8<<<grid(int64_t(3)*F*H*W),B,0,(cudaStream_t)s>>>((const __half*)in,out,H*W,F);return cudaGetLastError();}
