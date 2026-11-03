#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <climits>
namespace {
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
template<class T> __global__ void stitch_rect(const T* tiles,T* out,int C,int F,int KH,int KW,int H,int W,
 int ny,int nx,const int* ys,const int* xs,const int* yo,const int* xo){
 int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=int64_t(C)*F*H*W)return;
 int x=i%W,y=i/W%H,ct=i/(int64_t(W)*H),r=0,c=0;
 while(r+1<ny&&y>=ys[r+1])++r;while(c+1<nx&&x>=xs[c+1])++c;
 int ly=y-ys[r],lx=x-xs[c];int64_t elements=int64_t(C)*F*KH*KW;
 T v=tiles[(r*nx+c)*elements+int64_t(ct)*KH*KW+ly*KW+lx];
 if(r>0&&ly<yo[r-1])v=blend(tiles[((r-1)*nx+c)*elements+int64_t(ct)*KH*KW+(KH-yo[r-1]+ly)*KW+lx],v,ly,yo[r-1]);
 if(c>0&&lx<xo[c-1])v=blend(tiles[(r*nx+c-1)*elements+int64_t(ct)*KH*KW+ly*KW+KW-xo[c-1]+lx],v,lx,xo[c-1]);
 out[i]=v;
}
}

extern "C" int h3vg_stitch(const void*,void*,int,int,int,int,int,int,int,const int*,const int*,const int*,const int*,int,void*);
extern "C" int h3vgs_stitch(const void* in,void* out,int C,int F,int KH,int KW,int H,int W,int ny,int nx,const int* ys,const int* xs,const int* yo,const int* xo,int dtype,void* s){
 if(KH==KW&&(KH==16||KH==256))return h3vg_stitch(in,out,C,F,KH,H,W,ny,nx,ys,xs,yo,xo,dtype,s);
 const int64_t count=int64_t(C)*F*H*W;
 if(C<=0||F<=0||KH<=0||KW<=0||H<KH||W<KW||ny<=0||nx<=0||count<=0||count>INT_MAX||(dtype!=0&&dtype!=1))return cudaErrorInvalidValue;
 if(dtype==0)stitch_rect<<<(count+255)/256,256,0,(cudaStream_t)s>>>((const float*)in,(float*)out,C,F,KH,KW,H,W,ny,nx,ys,xs,yo,xo);
 else stitch_rect<<<(count+255)/256,256,0,(cudaStream_t)s>>>((const __half*)in,(__half*)out,C,F,KH,KW,H,W,ny,nx,ys,xs,yo,xo);
 return cudaGetLastError();
}
