// The shared contiguous implementation plus LTX's transposed video-token
// input projection. PyTorch Linear.cpp uses matmul followed by a separate bias
// add for this noncontiguous rank-three input (cf30153c4c131c81).
#include "../../../core_infra/kernels/linear.cu"

extern "C" int ltx25_linear_column_major(void* handle,const void* x,const void* weight,
    void* y,int rows,int out_features,int in_features,int dtype) {
    if(!handle || rows<=0 || out_features<=0 || in_features<=0)return -1;
    auto* ctx=static_cast<Context*>(handle);
    const float alpha=1.0f,beta=0.0f;
    if(dtype==0) {
        auto flags=ctx->reduced?CUBLAS_DEFAULT_MATH:
            static_cast<cublasMath_t>(CUBLAS_DEFAULT_MATH|CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION);
        BLAS(cublasSetMathMode(ctx->blas,flags));
        auto status=cublasGemmEx(ctx->blas,CUBLAS_OP_T,CUBLAS_OP_T,out_features,rows,in_features,
            &alpha,weight,CUDA_R_16BF,in_features,x,CUDA_R_16BF,rows,
            &beta,y,CUDA_R_16BF,out_features,CUDA_R_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        auto restored=cublasSetMathMode(ctx->blas,CUBLAS_DEFAULT_MATH);
        if(status!=CUBLAS_STATUS_SUCCESS)return fail("column-major cublasGemmEx",int(status));
        if(restored!=CUBLAS_STATUS_SUCCESS)return fail("column-major restore math mode",int(restored));
    } else if(dtype==1) {
        BLAS(cublasSetMathMode(ctx->blas,CUBLAS_DEFAULT_MATH));
        BLAS(cublasSgemm(ctx->blas,CUBLAS_OP_T,CUBLAS_OP_T,out_features,rows,in_features,
            &alpha,(const float*)weight,in_features,(const float*)x,rows,
            &beta,(float*)y,out_features));
    } else return -1;
    return 0;
}
extern "C" int ltx25_bmm_nn_f32(void* handle,const float* left,const float* right,float* out,
    int batch,int rows,int cols,int inner) {
    if(!handle || batch<1 || rows<1 || cols<1 || inner<1)return -1;
    auto* ctx=static_cast<Context*>(handle);const float alpha=1.0f,beta=0.0f;
    BLAS(cublasSetMathMode(ctx->blas,CUBLAS_DEFAULT_MATH));
    BLAS(cublasSgemmStridedBatched(ctx->blas,CUBLAS_OP_N,CUBLAS_OP_N,cols,rows,inner,&alpha,
        right,cols,int64_t(inner)*cols,left,inner,0,&beta,out,cols,int64_t(rows)*cols,batch));
    return 0;
}
