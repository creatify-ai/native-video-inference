#include "linear.h"
#include <cuda_runtime.h>
extern "C" int h3_audio_encoder_linear(const void* x,const void* w,const void* b,void* y,int rows,int cols,int inner,int tf32,void* stream){
    void* workspace=nullptr;void* context=nullptr;
    // Match the pinned Torch handles: 32 MiB legacy BLAS, 1 MiB cuBLASLt.
    // Increasing Lt's budget can select a different reduction algorithm.
    constexpr size_t blas_bytes=32u<<20,lt_bytes=1u<<20;
    int status=cudaMalloc(&workspace,blas_bytes+lt_bytes);if(status)return status;
    status=h3_linear_create(stream,workspace,blas_bytes,(char*)workspace+blas_bytes,lt_bytes,tf32,1,&context);
    if(!status)status=h3_linear_forward(context,x,w,b,y,rows,cols,inner,1);
    int sync=cudaStreamSynchronize((cudaStream_t)stream);
    if(context)h3_linear_destroy(context);cudaFree(workspace);
    return status?status:sync;
}
