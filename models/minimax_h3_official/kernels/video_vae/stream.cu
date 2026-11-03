// Literal existing pt8 transform, restricted to an already finalized frame range.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <climits>
#include "pixels.cuh"
namespace {
__global__ void pt8_range(const __half* in,uint8_t* out,int P,int F,int first,int count){
 int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=int64_t(3)*count*P)return;
 int c=i%3,p=(i/3)%P,t=i/(int64_t(3)*P)+first;
 out[i]=h3_public_rgb_pixel(in[(int64_t(c)*F+t)*P+p],c);
}
}
extern "C" int h3vg_pt8_range(const void* input,uint8_t* output,int H,int W,int F,int first,int count,void* stream){
 int64_t bytes=int64_t(3)*count*H*W;
 if(H<=0||W<=0||F<=0||first<0||count<=0||first>F-count||bytes>INT_MAX)return cudaErrorInvalidValue;
 pt8_range<<<(bytes+255)/256,256,0,(cudaStream_t)stream>>>((const __half*)input,output,H*W,F,first,count);
 return cudaGetLastError();
}

// Output health telemetry. Exact integer sums avoid an extra RGB copy
// and are downloaded before the existing frame-transfer synchronization.
__global__ void frame_sums(const uint8_t* input,uint64_t* output,int64_t bytes){
 __shared__ uint64_t sums[256];uint64_t sum=0;
 for(int64_t i=threadIdx.x;i<bytes;i+=256)sum+=input[int64_t(blockIdx.x)*bytes+i];
 sums[threadIdx.x]=sum;__syncthreads();
 for(int stride=128;stride;stride/=2){if(threadIdx.x<stride)sums[threadIdx.x]+=sums[threadIdx.x+stride];__syncthreads();}
 if(threadIdx.x==0)output[blockIdx.x]=sums[0];
}
extern "C" int h3vg_frame_sums(const uint8_t* input,uint64_t* output,int count,int64_t bytes,void* stream){
 if(count<=0||count>128||bytes<=0)return cudaErrorInvalidValue;
 frame_sums<<<count,256,0,(cudaStream_t)stream>>>(input,output,bytes);return cudaGetLastError();
}
