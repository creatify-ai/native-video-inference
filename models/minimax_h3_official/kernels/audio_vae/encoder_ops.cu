// Native FP32 audio encoder primitives; Welford follows pinned PyTorch 2.13.
// Derived Welford portions retain ../../licenses/PYTORCH-LICENSE.
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
namespace {
struct W { float mean, sigma2, count; };
__device__ W online(float val, W a) {
    float delta = val - a.mean;
    float count = a.count + 1.f;
    float mean = a.mean + delta * (1.f / count);
    return {mean, a.sigma2 + delta * (val - mean), count};
}
__device__ W combine(W b, W a) {
    float delta = b.mean - a.mean;
    float count = a.count + b.count;
    if (count <= 0) return {0, 0, count};
    float coef = 1.f / count, na = a.count * coef, nb = b.count * coef;
    return {na * a.mean + nb * b.mean, a.sigma2 + b.sigma2 + delta * delta * a.count * nb, count};
}
__global__ void layernorm(const float* x, const float* weight, const float* bias, float* out, int cols, float eps) {
    __shared__ float smem[8];
    int lane = threadIdx.x, warp = threadIdx.y, tid = lane + 32 * warp;
    int64_t offset = int64_t(blockIdx.x) * cols;
    W wd{0, 0, 0};
    for (int vector = tid; vector < cols / 4; vector += 128) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) wd = online(x[offset + vector * 4 + j], wd);
    }
    for (int delta = 16; delta; delta >>= 1) {
        W other{__shfl_down_sync(0xffffffff, wd.mean, delta),
                __shfl_down_sync(0xffffffff, wd.sigma2, delta),
                __shfl_down_sync(0xffffffff, wd.count, delta)};
        wd = combine(wd, other);
    }
    for (int delta = 2; delta; delta >>= 1) {
        if (!lane && warp >= delta && warp < 2 * delta) {
            smem[2 * (warp - delta)] = wd.mean;
            smem[2 * (warp - delta) + 1] = wd.sigma2;
            smem[4 + warp - delta] = wd.count;
        }
        __syncthreads();
        if (!lane && warp < delta) wd = combine(wd, {smem[2 * warp], smem[2 * warp + 1], smem[4 + warp]});
        __syncthreads();
    }
    if (!lane && !warp) { smem[0] = wd.mean; smem[1] = wd.sigma2 / float(cols); }
    __syncthreads();
    float rstd = rsqrtf(smem[1] + eps);
    for (int col = tid; col < cols; col += 128)
        out[offset + col] = weight[col] * (rstd * (x[offset + col] - smem[0])) + bias[col];
}

struct Layout { int rank; int64_t shape[5],stride[5]; };
__global__ void copy_strided(const float* x,float* out,int64_t count,Layout l,int64_t offset){
 for(int64_t i=int64_t(blockIdx.x)*256+threadIdx.x;i<count;i+=int64_t(gridDim.x)*256){
  int64_t t=i,j=offset;for(int d=l.rank-1;d>=0;--d){j+=(t%l.shape[d])*l.stride[d];t/=l.shape[d];}out[i]=x[j];
 }
}
__global__ void pad_zero(const float* x,float* out,int64_t count,int in,int n,int left){
 for(int64_t i=int64_t(blockIdx.x)*256+threadIdx.x;i<count;i+=int64_t(gridDim.x)*256){int c=i%n-left;out[i]=c<0||c>=in?0.f:x[(i/n)*in+c];}
}
__global__ void gelu(const float* x,float* out,int64_t count){
 for(int64_t i=int64_t(blockIdx.x)*256+threadIdx.x;i<count;i+=int64_t(gridDim.x)*256){
 float v=x[i];constexpr float beta=0.79788456080286535587989211986876f,kappa=0.044715f;
 float cube=v*v*v;float inner=beta*(v+kappa*cube);out[i]=0.5f*v*(1.f+tanhf(inner));}
}
__global__ void heads_mean(const float* x,float* out,int64_t count,int heads,int width){
 for(int64_t i=int64_t(blockIdx.x)*256+threadIdx.x;i<count;i+=int64_t(gridDim.x)*256){
  int64_t base=(i/width)*heads*width+i%width;
  // Reduce.cuh accumulates four interleaved lanes for this strided reduction.
  float a[4]={0,0,0,0};for(int h=0;h<heads;h++)a[h%4]+=x[base+h*width];
  out[i]=((a[0]+a[1])+a[2])+a[3];out[i]*=1.f/heads;
 }
}
__global__ void pool(const float* x,float* out,int64_t count,int in,int n){
 for(int64_t i=int64_t(blockIdx.x)*256+threadIdx.x;i<count;i+=int64_t(gridDim.x)*256){
  int c=i%n;int lo=(c*in)/n,hi=((c+1)*in+n-1)/n;float sum=0.f;
  for(int j=lo;j<hi;j++)sum+=x[(i/n)*in+j];out[i]=sum/(hi-lo);
 }
}
}
extern "C" int h3_audio_copy5(const void* x,void* y,int rank,const int64_t* shape,const int64_t* stride,int64_t offset,void* stream){
 Layout l{};l.rank=rank;int64_t count=1;for(int i=0;i<rank;i++){l.shape[i]=shape[i];l.stride[i]=stride[i];count*=shape[i];}
 copy_strided<<<unsigned((count+255)/256),256,0,(cudaStream_t)stream>>>((const float*)x,(float*)y,count,l,offset);return cudaGetLastError();
}
extern "C" int h3_audio_layernorm(const void* x,const void* w,const void* b,void* y,int rows,int cols,float eps,void* stream){
 if(cols%4||cols<4)return -1;
 layernorm<<<rows,dim3(32,4),0,(cudaStream_t)stream>>>((const float*)x,(const float*)w,(const float*)b,(float*)y,cols,eps);return cudaGetLastError();
}
extern "C" int h3_audio_encoder_op(int op,const void* x,void* y,int64_t count,int a,int b,int c,void* stream){
 auto s=(cudaStream_t)stream;int grid=(count+255)/256;
 if(op==0)pad_zero<<<grid,256,0,s>>>((const float*)x,(float*)y,count,a,b,c);
 else if(op==1)gelu<<<grid,256,0,s>>>((const float*)x,(float*)y,count);
 else if(op==2)heads_mean<<<grid,256,0,s>>>((const float*)x,(float*)y,count,a,b);
 else if(op==3)pool<<<grid,256,0,s>>>((const float*)x,(float*)y,count,a,b);
 else return -1;return cudaGetLastError();
}
