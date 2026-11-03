// Native linear algebra boundary for the pinned Torch 2.13 CUDA backend.
// No ATen/Python headers or linkage. Row-major X[M,K], W[N,K], Y[M,N].
#include "linear.h"
#include <cublas_v2.h>
#include <cublasLt.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <new>
#include <algorithm>
#include <cstdlib>
#include <cstring>

namespace {
thread_local char error_message[256] = "";
int fail(const char* call, int code) {
    snprintf(error_message, sizeof(error_message), "%s: status %d", call, code);
    return code ? code : -1;
}
#define BLAS(call) do { auto code = (call); if (code != CUBLAS_STATUS_SUCCESS) return fail(#call, code); } while(0)

struct Context {
    cublasHandle_t blas = nullptr;
    cublasLtHandle_t lt = nullptr;
    cudaStream_t stream = nullptr;
    void* workspace = nullptr;
    size_t workspace_bytes = 0;
    bool tf32 = false;
    bool reduced = true;
    ~Context() { if (blas) cublasDestroy(blas); }
};
struct Descriptors {
    cublasLtMatmulDesc_t op = nullptr;
    cublasLtMatrixLayout_t a = nullptr, b = nullptr, c = nullptr;
    cublasLtMatmulPreference_t preference = nullptr;
    ~Descriptors() {
        if (preference) cublasLtMatmulPreferenceDestroy(preference);
        if (c) cublasLtMatrixLayoutDestroy(c);
        if (b) cublasLtMatrixLayoutDestroy(b);
        if (a) cublasLtMatrixLayoutDestroy(a);
        if (op) cublasLtMatmulDescDestroy(op);
    }
};
uint32_t alignment(const void* ptr) {
    uint32_t result = 256;
    while ((reinterpret_cast<uintptr_t>(ptr) % result) != 0) result /= 2;
    return result;
}
}

// Native form of pinned PyTorch addmm fallback. See vendor_reference/LICENSE
// and vendor_reference/{Blas,CUDABlas}.cpp for source and operation ordering.
// Bias is copied into C before GEMM(beta=1); no intermediate GEMM output cast.
namespace h3_linear_generic {
thread_local uint64_t counts[5]={0,0,0,0,0};
enum Reason { LtSuccess=0, Forced=1, SmallDimension=2, EmptyHeuristic=3, LtFailure=4 };
template<class Bits>
__global__ void broadcast(const Bits* bias,Bits* output,int64_t elements,int columns) {
    for(int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<elements;i+=int64_t(gridDim.x)*blockDim.x)
        output[i]=bias[i%columns];
}
int force_mode() {
    const char* value=std::getenv("H3_LINEAR_FORCE_FALLBACK");
    if(!value || std::strcmp(value,"0")==0)return 0;
    if(std::strcmp(value,"1")==0)return 1;
    return -1;
}
int fallback(Context* ctx,const void* input,const void* weight,const void* bias,void* output,
             int rows,int out_features,int in_features,int dtype,Reason reason) {
    ++counts[reason];
    int64_t elements=int64_t(rows)*out_features;
    unsigned blocks=unsigned(std::min<int64_t>((elements+255)/256,65535));
    if((dtype==0 || dtype==2))broadcast<<<blocks,256,0,ctx->stream>>>((const uint16_t*)bias,(uint16_t*)output,elements,out_features);
    else broadcast<<<blocks,256,0,ctx->stream>>>((const uint32_t*)bias,(uint32_t*)output,elements,out_features);
    auto copied=cudaGetLastError();
    if(copied!=cudaSuccess)return fail("fallback bias broadcast",int(copied));
    const float alpha=1.0f,beta=1.0f;
    if((dtype==0 || dtype==2)) {
        auto flags=ctx->reduced ? CUBLAS_DEFAULT_MATH :
            static_cast<cublasMath_t>(CUBLAS_DEFAULT_MATH | CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION);
        BLAS(cublasSetMathMode(ctx->blas,flags));
        auto status=cublasGemmEx(ctx->blas,CUBLAS_OP_T,CUBLAS_OP_N,out_features,rows,in_features,
            &alpha,weight,(dtype==0 ? CUDA_R_16BF : CUDA_R_16F),in_features,input,(dtype==0 ? CUDA_R_16BF : CUDA_R_16F),in_features,
            &beta,output,(dtype==0 ? CUDA_R_16BF : CUDA_R_16F),out_features,CUDA_R_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        auto restore=cublasSetMathMode(ctx->blas,CUBLAS_DEFAULT_MATH);
        if(status!=CUBLAS_STATUS_SUCCESS)return fail("fallback cublasGemmEx",int(status));
        if(restore!=CUBLAS_STATUS_SUCCESS)return fail("fallback restore math mode",int(restore));
    }else{
        BLAS(cublasSetMathMode(ctx->blas,ctx->tf32 ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH));
        BLAS(cublasSgemm(ctx->blas,CUBLAS_OP_T,CUBLAS_OP_N,out_features,rows,in_features,
            &alpha,(const float*)weight,in_features,(const float*)input,in_features,
            &beta,(float*)output,out_features));
    }
    error_message[0]='\0';
    return 0;
}
} // namespace h3_linear_generic
extern "C" void h3_linear_fallback_reset(){for(auto& count:h3_linear_generic::counts)count=0;}
extern "C" int h3_linear_fallback_counts(uint64_t* out,int size){
    if(!out||size!=5)return -1;
    for(int i=0;i<5;++i)out[i]=h3_linear_generic::counts[i];
    return 0;
}

extern "C" const char* h3_linear_error() { return error_message; }
extern "C" int h3_linear_create(void* stream, void* blas_workspace, size_t blas_workspace_bytes,
    void* lt_workspace, size_t lt_workspace_bytes, int allow_tf32, int allow_reduced, void** out) {
    if (!out || !blas_workspace || !lt_workspace) return fail("invalid context arguments", -1);
    Context* ctx = new(std::nothrow) Context;
    if (!ctx) return fail("context allocation", -1);
    ctx->stream = static_cast<cudaStream_t>(stream);
    ctx->workspace = lt_workspace;
    ctx->workspace_bytes = lt_workspace_bytes;
    ctx->tf32 = allow_tf32 != 0;
    ctx->reduced = allow_reduced != 0;
    auto a = cublasCreate(&ctx->blas);
    if (a != CUBLAS_STATUS_SUCCESS) { delete ctx; return fail("cublasCreate", a); }
    // Torch's CUDA handle pool shares the cuBLAS handle with the Lt API.
    ctx->lt = reinterpret_cast<cublasLtHandle_t>(ctx->blas);
    auto c = cublasSetStream(ctx->blas, ctx->stream);
    if (c != CUBLAS_STATUS_SUCCESS) { delete ctx; return fail("cublasSetStream", c); }
    auto d = cublasSetWorkspace(ctx->blas, blas_workspace, blas_workspace_bytes);
    if (d != CUBLAS_STATUS_SUCCESS) { delete ctx; return fail("cublasSetWorkspace", d); }
    *out = ctx;
    return 0;
}

extern "C" void h3_linear_destroy(void* handle) { delete static_cast<Context*>(handle); }
extern "C" int h3_linear_cublas_version(void* handle, int* version) {
    BLAS(cublasGetVersion(static_cast<Context*>(handle)->blas, version)); return 0;
}

// The compiled decoder's first projection keeps NCTHW's column-major token
// view. Preserve the original cuBLAS transpose contract instead of copying it
// into the contiguous layout used by the remaining projection layers.
extern "C" int h3v_mm(void* handle, const void* input, const void* weight,
    void* output, int rows, int out_features, int in_features, int input_column_major) {
    auto* ctx = static_cast<Context*>(handle);
    const float alpha=1.0f,beta=0.0f;
    auto flags=ctx->reduced?CUBLAS_DEFAULT_MATH:
        static_cast<cublasMath_t>(CUBLAS_DEFAULT_MATH|CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION);
    BLAS(cublasSetMathMode(ctx->blas,flags));
    BLAS(cublasGemmEx(ctx->blas,CUBLAS_OP_T,input_column_major?CUBLAS_OP_T:CUBLAS_OP_N,
        out_features,rows,in_features,&alpha,weight,CUDA_R_16F,in_features,
        input,CUDA_R_16F,input_column_major?rows:in_features,&beta,output,CUDA_R_16F,
        out_features,CUDA_R_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    BLAS(cublasSetMathMode(ctx->blas,CUBLAS_DEFAULT_MATH));return 0;
}

extern "C" int h3_linear_forward(void* handle, const void* input, const void* weight,
    const void* bias, void* output, int rows, int out_features, int in_features, int dtype) {
    if (!handle || !input || !weight || !output || rows <= 0 || out_features < 1 ||
        in_features < 1 || (dtype != 0 && dtype != 1 && dtype != 2)) return fail("unsupported Linear arguments", -1);
    auto* ctx = static_cast<Context*>(handle);
    const int m = out_features, n = rows, k = in_features;
    const int lda = k, ldb = k, ldc = m;
    const float alpha = 1.0f, beta = 0.0f;
    if (!bias) {
        if (dtype == 0 || dtype == 2) {
            auto flags = ctx->reduced ? CUBLAS_DEFAULT_MATH :
                static_cast<cublasMath_t>(CUBLAS_DEFAULT_MATH | CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION);
            BLAS(cublasSetMathMode(ctx->blas, flags));
            // Torch's CUDA_R_32F overload selects the same legacy compute contract.
            BLAS(cublasGemmEx(ctx->blas, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k,
                &alpha, weight, dtype == 0 ? CUDA_R_16BF : CUDA_R_16F, lda,
                input, dtype == 0 ? CUDA_R_16BF : CUDA_R_16F, ldb,
                &beta, output, dtype == 0 ? CUDA_R_16BF : CUDA_R_16F, ldc,
                CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
            BLAS(cublasSetMathMode(ctx->blas, CUBLAS_DEFAULT_MATH));
        } else {
            BLAS(cublasSetMathMode(ctx->blas, ctx->tf32 ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH));
            BLAS(cublasSgemm(ctx->blas, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k,
                &alpha, static_cast<const float*>(weight), lda, static_cast<const float*>(input), ldb,
                &beta, static_cast<float*>(output), ldc));
        }
        return 0;
    }
    const int force_fallback=h3_linear_generic::force_mode();
    if(force_fallback<0)return fail("H3_LINEAR_FORCE_FALLBACK must be 0 or 1",-1);
    if(force_fallback || out_features==1 || in_features==1)
        return h3_linear_generic::fallback(ctx,input,weight,bias,output,rows,out_features,in_features,dtype,
            force_fallback ? h3_linear_generic::Forced : h3_linear_generic::SmallDimension);
    const auto type = dtype == 0 ? CUDA_R_16BF : dtype == 2 ? CUDA_R_16F : CUDA_R_32F;
    const auto compute = dtype == 1 && ctx->tf32 ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;
    Descriptors desc;
    BLAS(cublasLtMatmulDescCreate(&desc.op, compute, CUDA_R_32F));
    const cublasOperation_t transa = CUBLAS_OP_T, transb = CUBLAS_OP_N;
    BLAS(cublasLtMatmulDescSetAttribute(desc.op, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
    BLAS(cublasLtMatmulDescSetAttribute(desc.op, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));
    const cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;
    BLAS(cublasLtMatmulDescSetAttribute(desc.op, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));
    BLAS(cublasLtMatmulDescSetAttribute(desc.op, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias)));
    BLAS(cublasLtMatrixLayoutCreate(&desc.a, type, k, m, lda));
    BLAS(cublasLtMatrixLayoutCreate(&desc.b, type, k, n, ldb));
    BLAS(cublasLtMatrixLayoutCreate(&desc.c, type, m, n, ldc));
    BLAS(cublasLtMatmulPreferenceCreate(&desc.preference));
    BLAS(cublasLtMatmulPreferenceSetAttribute(desc.preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &ctx->workspace_bytes, sizeof(ctx->workspace_bytes)));
    if ((dtype == 0 || dtype == 2) && !ctx->reduced) {
        uint32_t mask = CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE | CUBLASLT_REDUCTION_SCHEME_NONE;
        BLAS(cublasLtMatmulPreferenceSetAttribute(desc.preference, CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK, &mask, sizeof(mask)));
    }
    const void* pointers[] = {weight, input, output, bias};
    const cublasLtMatmulPreferenceAttributes_t attrs[] = {
        CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES,
        CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES};
    for (int i = 0; i != 4; ++i) {
        uint32_t value = alignment(pointers[i]);
        BLAS(cublasLtMatmulPreferenceSetAttribute(desc.preference, attrs[i], &value, sizeof(value)));
    }
    cublasLtMatmulHeuristicResult_t heuristic{};
    int count = 0;
    auto heuristic_status=cublasLtMatmulAlgoGetHeuristic(ctx->lt, desc.op, desc.a, desc.b, desc.c, desc.c,
        desc.preference, 1, &heuristic, &count);
    if(heuristic_status==CUBLAS_STATUS_NOT_SUPPORTED)
        return h3_linear_generic::fallback(ctx,input,weight,bias,output,rows,out_features,in_features,dtype,
            h3_linear_generic::EmptyHeuristic);
    if(heuristic_status!=CUBLAS_STATUS_SUCCESS)return fail("cublasLtMatmulAlgoGetHeuristic",int(heuristic_status));
    if (count != 1) return h3_linear_generic::fallback(ctx,input,weight,bias,output,
        rows,out_features,in_features,dtype,h3_linear_generic::EmptyHeuristic);
    auto lt_status=cublasLtMatmul(ctx->lt, desc.op, &alpha, weight, desc.a, input, desc.b, &beta,
        output, desc.c, output, desc.c, &heuristic.algo, ctx->workspace, ctx->workspace_bytes, ctx->stream);
    if(lt_status==CUBLAS_STATUS_NOT_SUPPORTED)return h3_linear_generic::fallback(ctx,input,weight,bias,output,
        rows,out_features,in_features,dtype,h3_linear_generic::LtFailure);
    if(lt_status!=CUBLAS_STATUS_SUCCESS)return fail("cublasLtMatmul",int(lt_status));
    ++h3_linear_generic::counts[h3_linear_generic::LtSuccess];
    return 0;
}
