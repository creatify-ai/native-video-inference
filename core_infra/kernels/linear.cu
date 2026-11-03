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
#include <map>
#include <string>
#include <tuple>
#include <vector>
#include <array>
#include <chrono>

namespace {
thread_local char error_message[256] = "";
int fail(const char* call, int code) {
    snprintf(error_message, sizeof(error_message), "%s: status %d", call, code);
    return code ? code : -1;
}
#define BLAS(call) do { auto code = (call); if (code != CUBLAS_STATUS_SUCCESS) return fail(#call, code); } while(0)
// cuBLAS element type of a Linear dtype code (linear.h).
inline cudaDataType_t io_type(int dtype) { return dtype == 0 ? CUDA_R_16BF : dtype == 2 ? CUDA_R_16F : CUDA_R_32F; }

// Reference-pinned algorithm selection for shapes whose accepted result is
// computed elsewhere (infra_linear_tune_against). cuBLASLt heuristic candidates
// are accepted only when their output is bitwise identical to that reference;
// the fastest such candidate is then used through infra_linear_forward_pinned.
// Ordinary infra_linear_forward calls never consult these decisions.
struct TuneKey {
    int m, n, k, dtype, bias; uint32_t align;
    bool operator<(const TuneKey& o) const { return std::tie(m,n,k,dtype,bias,align) < std::tie(o.m,o.n,o.k,o.dtype,o.bias,o.align); }
};
struct TuneChoice { bool use_lt = false, pinned = false; cublasLtMatmulAlgo_t algo{}; double default_us = 0, best_us = 0; int candidates = 0, identical = 0, chosen = -1; };
struct Tuner {
    std::map<TuneKey, TuneChoice> choices;
    void* scratch = nullptr; size_t scratch_bytes = 0;
    void* workspace = nullptr; size_t workspace_bytes = 64u << 20;
    unsigned* mismatch = nullptr;
    cudaEvent_t start = nullptr, stop = nullptr;
    std::string report = "[";
    ~Tuner() {
        if (scratch) cudaFree(scratch); if (workspace) cudaFree(workspace); if (mismatch) cudaFree(mismatch);
        if (start) cudaEventDestroy(start); if (stop) cudaEventDestroy(stop);
    }
};
// One tuner per process: decisions survive layout rebuilds (new contexts), and a
// single host thread drives every context, so the shared scratch is never contended.
Tuner* global_tuner() { static Tuner* tuner = new(std::nothrow) Tuner; return tuner; }
__global__ void count_mismatch(const uint32_t* a, const uint32_t* b, size_t words, unsigned* out) {
    unsigned local = 0;
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < words; i += size_t(gridDim.x) * blockDim.x) local += (a[i] != b[i]);
    if (local) atomicAdd(out, local);
}
struct Context {
    Tuner* tuner = nullptr;
    cublasHandle_t blas = nullptr;
    cublasLtHandle_t lt = nullptr;
    cudaStream_t stream = nullptr;
    void* workspace = nullptr;
    size_t workspace_bytes = 0;
    bool tf32 = false;
    bool reduced = true;
    int (*fallback_policy)() = nullptr;
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
namespace infra_linear_generic {
thread_local uint64_t counts[5]={0,0,0,0,0};
enum Reason { LtSuccess=0, Forced=1, SmallDimension=2, EmptyHeuristic=3, LtFailure=4 };
template<class Bits>
__global__ void broadcast(const Bits* bias,Bits* output,int64_t elements,int columns) {
    for(int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<elements;i+=int64_t(gridDim.x)*blockDim.x)
        output[i]=bias[i%columns];
}
int force_mode(Context* ctx) {return ctx->fallback_policy ? ctx->fallback_policy() : 0;}

int fallback(Context* ctx,const void* input,const void* weight,const void* bias,void* output,
             int rows,int out_features,int in_features,int dtype,Reason reason) {
    ++counts[reason];
    int64_t elements=int64_t(rows)*out_features;
    unsigned blocks=unsigned(std::min<int64_t>((elements+255)/256,65535));
    if(dtype!=1)broadcast<<<blocks,256,0,ctx->stream>>>((const uint16_t*)bias,(uint16_t*)output,elements,out_features);
    else broadcast<<<blocks,256,0,ctx->stream>>>((const uint32_t*)bias,(uint32_t*)output,elements,out_features);
    auto copied=cudaGetLastError();
    if(copied!=cudaSuccess)return fail("fallback bias broadcast",int(copied));
    const float alpha=1.0f,beta=1.0f;
    if(dtype!=1) {
        auto flags=ctx->reduced ? CUBLAS_DEFAULT_MATH :
            static_cast<cublasMath_t>(CUBLAS_DEFAULT_MATH | CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION);
        BLAS(cublasSetMathMode(ctx->blas,flags));
        auto status=cublasGemmEx(ctx->blas,CUBLAS_OP_T,CUBLAS_OP_N,out_features,rows,in_features,
            &alpha,weight,io_type(dtype),in_features,input,io_type(dtype),in_features,
            &beta,output,io_type(dtype),out_features,CUDA_R_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP);
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
} // namespace infra_linear_generic
extern "C" void infra_linear_fallback_reset(){for(auto& count:infra_linear_generic::counts)count=0;}
extern "C" int infra_linear_fallback_counts(uint64_t* out,int size){
    if(!out||size!=5)return -1;
    for(int i=0;i<5;++i)out[i]=infra_linear_generic::counts[i];
    return 0;
}

extern "C" const char* infra_linear_error() { return error_message; }
extern "C" int infra_linear_create(void* stream, void* blas_workspace, size_t blas_workspace_bytes,
    void* lt_workspace, size_t lt_workspace_bytes, int allow_tf32, int allow_reduced, void** out, int (*fallback_policy)()) {
    if (!out || !blas_workspace || !lt_workspace) return fail("invalid context arguments", -1);
    Context* ctx = new(std::nothrow) Context;
    if (!ctx) return fail("context allocation", -1);
    ctx->stream = static_cast<cudaStream_t>(stream);
    ctx->workspace = lt_workspace;
    ctx->workspace_bytes = lt_workspace_bytes;
    ctx->tf32 = allow_tf32 != 0;
    ctx->reduced = allow_reduced != 0;
    ctx->fallback_policy = fallback_policy;
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

extern "C" void infra_linear_destroy(void* handle) { delete static_cast<Context*>(handle); }
extern "C" int infra_linear_cublas_version(void* handle, int* version) {
    BLAS(cublasGetVersion(static_cast<Context*>(handle)->blas, version)); return 0;
}

namespace {
int forward_default(Context* ctx, const void* input, const void* weight,
    const void* bias, void* output, int rows, int out_features, int in_features, int dtype) {
    const int m = out_features, n = rows, k = in_features;
    const int lda = k, ldb = k, ldc = m;
    const float alpha = 1.0f, beta = 0.0f;
    if (!bias) {
        if (dtype != 1) {
            auto flags = ctx->reduced ? CUBLAS_DEFAULT_MATH :
                static_cast<cublasMath_t>(CUBLAS_DEFAULT_MATH | CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION);
            BLAS(cublasSetMathMode(ctx->blas, flags));
            // Torch's CUDA_R_32F overload selects the same legacy compute contract.
            BLAS(cublasGemmEx(ctx->blas, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k,
                &alpha, weight, io_type(dtype), lda, input, io_type(dtype), ldb,
                &beta, output, io_type(dtype), ldc, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
            BLAS(cublasSetMathMode(ctx->blas, CUBLAS_DEFAULT_MATH));
        } else {
            BLAS(cublasSetMathMode(ctx->blas, ctx->tf32 ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH));
            BLAS(cublasSgemm(ctx->blas, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k,
                &alpha, static_cast<const float*>(weight), lda, static_cast<const float*>(input), ldb,
                &beta, static_cast<float*>(output), ldc));
        }
        return 0;
    }
    const int force_fallback=infra_linear_generic::force_mode(ctx);
    if(force_fallback<0)return fail("linear fallback policy must be 0 or 1",-1);
    if(force_fallback || out_features==1 || in_features==1)
        return infra_linear_generic::fallback(ctx,input,weight,bias,output,rows,out_features,in_features,dtype,
            force_fallback ? infra_linear_generic::Forced : infra_linear_generic::SmallDimension);
    const auto type = io_type(dtype);
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
    if (dtype != 1 && !ctx->reduced) {
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
        return infra_linear_generic::fallback(ctx,input,weight,bias,output,rows,out_features,in_features,dtype,
            infra_linear_generic::EmptyHeuristic);
    if(heuristic_status!=CUBLAS_STATUS_SUCCESS)return fail("cublasLtMatmulAlgoGetHeuristic",int(heuristic_status));
    if (count != 1) return infra_linear_generic::fallback(ctx,input,weight,bias,output,
        rows,out_features,in_features,dtype,infra_linear_generic::EmptyHeuristic);
    auto lt_status=cublasLtMatmul(ctx->lt, desc.op, &alpha, weight, desc.a, input, desc.b, &beta,
        output, desc.c, output, desc.c, &heuristic.algo, ctx->workspace, ctx->workspace_bytes, ctx->stream);
    if(lt_status==CUBLAS_STATUS_NOT_SUPPORTED)return infra_linear_generic::fallback(ctx,input,weight,bias,output,
        rows,out_features,in_features,dtype,infra_linear_generic::LtFailure);
    if(lt_status!=CUBLAS_STATUS_SUCCESS)return fail("cublasLtMatmul",int(lt_status));
    ++infra_linear_generic::counts[infra_linear_generic::LtSuccess];
    return 0;
}

// One cuBLASLt matmul with an explicit algorithm and the tuner workspace.
int run_lt_algo(Context* ctx, const void* input, const void* weight, const void* bias, void* output,
                int rows, int out_features, int in_features, int dtype, const cublasLtMatmulAlgo_t& algo) {
    const int m = out_features, n = rows, k = in_features;
    const int lda = k, ldb = k, ldc = m;
    const float alpha = 1.0f, beta = 0.0f;
    const auto type = io_type(dtype);
    const auto compute = dtype == 1 && ctx->tf32 ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;
    Descriptors desc;
    BLAS(cublasLtMatmulDescCreate(&desc.op, compute, CUDA_R_32F));
    const cublasOperation_t transa = CUBLAS_OP_T, transb = CUBLAS_OP_N;
    BLAS(cublasLtMatmulDescSetAttribute(desc.op, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
    BLAS(cublasLtMatmulDescSetAttribute(desc.op, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));
    if (bias) {
        const cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;
        BLAS(cublasLtMatmulDescSetAttribute(desc.op, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));
        BLAS(cublasLtMatmulDescSetAttribute(desc.op, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias)));
    }
    BLAS(cublasLtMatrixLayoutCreate(&desc.a, type, k, m, lda));
    BLAS(cublasLtMatrixLayoutCreate(&desc.b, type, k, n, ldb));
    BLAS(cublasLtMatrixLayoutCreate(&desc.c, type, m, n, ldc));
    auto status = cublasLtMatmul(ctx->lt, desc.op, &alpha, weight, desc.a, input, desc.b, &beta,
        output, desc.c, output, desc.c, &algo, ctx->tuner->workspace, ctx->tuner->workspace_bytes, ctx->stream);
    if (status != CUBLAS_STATUS_SUCCESS) return fail("tuned cublasLtMatmul", int(status));
    return 0;
}

int tune_shape(Context* ctx, const TuneKey& key, const void* input, const void* weight, const void* bias,
               const void* reference, int rows, int out_features, int in_features, int dtype) {
    Tuner& t = *ctx->tuner;
    const size_t bytes = size_t(rows) * out_features * (dtype == 1 ? 4 : 2);
    if (!t.workspace) { if (cudaMalloc(&t.workspace, t.workspace_bytes) != cudaSuccess) return fail("tuner workspace", -1); }
    if (!t.mismatch) { if (cudaMalloc(&t.mismatch, sizeof(unsigned)) != cudaSuccess) return fail("tuner counter", -1); }
    if (!t.start) { if (cudaEventCreate(&t.start) != cudaSuccess || cudaEventCreate(&t.stop) != cudaSuccess) return fail("tuner events", -1); }
    if (t.scratch_bytes < bytes) {
        if (t.scratch) cudaFree(t.scratch);
        t.scratch = nullptr; t.scratch_bytes = 0;
        if (cudaMalloc(&t.scratch, bytes) != cudaSuccess) return fail("tuner scratch", -1);
        t.scratch_bytes = bytes;
    }
    TuneChoice choice;
    auto time_of = [&](auto&& run) -> double {
        double best = 1e30;
        for (int rep = 0; rep < 2; ++rep) {
            if (cudaEventRecord(t.start, ctx->stream) != cudaSuccess) return -1;
            if (run() != 0) return -1;
            if (cudaEventRecord(t.stop, ctx->stream) != cudaSuccess || cudaEventSynchronize(t.stop) != cudaSuccess) return -1;
            float ms = 0; cudaEventElapsedTime(&ms, t.start, t.stop); best = std::min(best, double(ms) * 1000.0);
        }
        return best;
    };
    choice.default_us = time_of([&] { return forward_default(ctx, input, weight, bias, t.scratch, rows, out_features, in_features, dtype); });
    // Candidate algorithms from the heuristic with the larger tuner workspace.
    const int m = out_features, n = rows, k = in_features;
    const int lda = k, ldb = k, ldc = m;
    const auto type = io_type(dtype);
    const auto compute = dtype == 1 && ctx->tf32 ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;
    std::vector<cublasLtMatmulHeuristicResult_t> results(32);
    int count = 0;
    {
        Descriptors desc;
        BLAS(cublasLtMatmulDescCreate(&desc.op, compute, CUDA_R_32F));
        const cublasOperation_t transa = CUBLAS_OP_T, transb = CUBLAS_OP_N;
        BLAS(cublasLtMatmulDescSetAttribute(desc.op, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
        BLAS(cublasLtMatmulDescSetAttribute(desc.op, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));
        if (bias) {
            const cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;
            BLAS(cublasLtMatmulDescSetAttribute(desc.op, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));
            BLAS(cublasLtMatmulDescSetAttribute(desc.op, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias)));
        }
        BLAS(cublasLtMatrixLayoutCreate(&desc.a, type, k, m, lda));
        BLAS(cublasLtMatrixLayoutCreate(&desc.b, type, k, n, ldb));
        BLAS(cublasLtMatrixLayoutCreate(&desc.c, type, m, n, ldc));
        BLAS(cublasLtMatmulPreferenceCreate(&desc.preference));
        BLAS(cublasLtMatmulPreferenceSetAttribute(desc.preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &t.workspace_bytes, sizeof(t.workspace_bytes)));
        if (dtype != 1 && !ctx->reduced) {
            uint32_t mask = CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE | CUBLASLT_REDUCTION_SCHEME_NONE;
            BLAS(cublasLtMatmulPreferenceSetAttribute(desc.preference, CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK, &mask, sizeof(mask)));
        }
        const void* pointers[] = {weight, input, t.scratch, bias ? bias : t.scratch};
        const cublasLtMatmulPreferenceAttributes_t attrs[] = {
            CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES,
            CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES};
        for (int i = 0; i != 4; ++i) {
            uint32_t value = std::min(key.align, alignment(pointers[i]));
            BLAS(cublasLtMatmulPreferenceSetAttribute(desc.preference, attrs[i], &value, sizeof(value)));
        }
        auto status = cublasLtMatmulAlgoGetHeuristic(ctx->lt, desc.op, desc.a, desc.b, desc.c, desc.c,
            desc.preference, int(results.size()), results.data(), &count);
        if (status != CUBLAS_STATUS_SUCCESS) count = 0;
    }
    choice.candidates = count;
    const size_t words = bytes / 4;
    for (int i = 0; i < count; ++i) {
        if (results[i].state != CUBLAS_STATUS_SUCCESS) continue;
        if (run_lt_algo(ctx, input, weight, bias, t.scratch, rows, out_features, in_features, dtype, results[i].algo) != 0) continue;
        if (cudaMemsetAsync(t.mismatch, 0, sizeof(unsigned), ctx->stream) != cudaSuccess) return fail("tuner memset", -1);
        unsigned blocks = unsigned(std::min<size_t>((words + 255) / 256, 4096));
        count_mismatch<<<blocks, 256, 0, ctx->stream>>>(static_cast<const uint32_t*>(reference), static_cast<const uint32_t*>(t.scratch), words, t.mismatch);
        unsigned mismatches = 1;
        if (cudaMemcpyAsync(&mismatches, t.mismatch, sizeof(unsigned), cudaMemcpyDeviceToHost, ctx->stream) != cudaSuccess ||
            cudaStreamSynchronize(ctx->stream) != cudaSuccess) return fail("tuner compare", -1);
        if (mismatches != 0) continue;
        ++choice.identical;
        const double us = time_of([&] { return run_lt_algo(ctx, input, weight, bias, t.scratch, rows, out_features, in_features, dtype, results[i].algo); });
        if (us > 0 && (choice.chosen < 0 || us < choice.best_us)) { choice.best_us = us; choice.chosen = i; choice.algo = results[i].algo; }
    }
    // Any identical candidate is acceptable for a pinned shape; the fastest one is kept.
    choice.use_lt = choice.chosen >= 0;
    char line[320];
    snprintf(line, sizeof(line), "%s{\"m\":%d,\"n\":%d,\"k\":%d,\"dtype\":%d,\"bias\":%d,\"candidates\":%d,\"identical\":%d,\"default_us\":%.1f,\"best_identical_us\":%.1f,\"chosen\":%d,\"tuned\":%s}",
        t.choices.empty() ? "" : ",", key.m, key.n, key.k, key.dtype, key.bias, choice.candidates, choice.identical,
        choice.default_us, choice.chosen >= 0 ? choice.best_us : -1.0, choice.chosen, choice.use_lt ? "true" : "false");
    t.report += line;
    t.choices[key] = choice;
    error_message[0] = '\0';
    return 0;
}
} // namespace

extern "C" int infra_linear_forward(void* handle, const void* input, const void* weight,
    const void* bias, void* output, int rows, int out_features, int in_features, int dtype) {
    if (!handle || !input || !weight || !output || rows <= 0 || out_features < 1 ||
        in_features < 1 || (dtype < 0 || dtype > 2)) return fail("unsupported Linear arguments", -1);
    return forward_default(static_cast<Context*>(handle), input, weight, bias, output, rows, out_features, in_features, dtype);
}
// Pin an algorithm for a shape whose accepted result is computed elsewhere (for
// example the adapter down projection at full sequence length): `output` holds
// that reference for these rows; candidates must reproduce it bit for bit.
// *exact_found is 1 when a candidate was pinned for later infra_linear_forward calls
// with this shape, else 0 (the caller must keep its own accepted path).
extern "C" int infra_linear_tune_against(void* handle, const void* input, const void* weight, const void* bias,
    void* output, int rows, int out_features, int in_features, int dtype, int* exact_found) {
    if (!handle || !input || !weight || !output || !exact_found || rows <= 0 || out_features < 1 ||
        in_features < 1 || (dtype < 0 || dtype > 2)) return fail("unsupported tune_against arguments", -1);
    auto* ctx = static_cast<Context*>(handle);
    *exact_found = 0;
    if (!ctx->tuner) { ctx->tuner = global_tuner(); if (!ctx->tuner) return fail("tuner allocation", -1); }
    const uint32_t align = std::min(std::min(alignment(input), alignment(weight)), std::min(alignment(output), bias ? alignment(bias) : 256u));
    const TuneKey key{out_features, rows, in_features, dtype, bias ? 1 : 0, align};
    auto existing = ctx->tuner->choices.find(key);
    if (existing != ctx->tuner->choices.end() && existing->second.pinned) { *exact_found = existing->second.use_lt ? 1 : 0; return 0; }
    int status = tune_shape(ctx, key, input, weight, bias, output, rows, out_features, in_features, dtype);
    if (status != 0) return status;
    auto& choice = ctx->tuner->choices[key];
    choice.pinned = true;
    // Without an exact candidate the caller must keep its accepted path for this shape.
    ctx->tuner->report += choice.use_lt ? ",{\"pinned_against_reference\":true}" : ",{\"pinned_against_reference\":false}";
    *exact_found = choice.use_lt ? 1 : 0;
    return 0;
}
// Run only through an algorithm pinned by infra_linear_tune_against for this exact
// shape/alignment; returns 2 when none is pinned so the caller keeps its accepted path.
extern "C" int infra_linear_forward_pinned(void* handle, const void* input, const void* weight, const void* bias,
    void* output, int rows, int out_features, int in_features, int dtype) {
    if (!handle || !input || !weight || !output || rows <= 0 || out_features < 1 ||
        in_features < 1 || (dtype < 0 || dtype > 2)) return fail("unsupported pinned arguments", -1);
    auto* ctx = static_cast<Context*>(handle);
    if (!ctx->tuner) ctx->tuner = global_tuner();
    if (!ctx->tuner) return fail("tuner allocation", -1);
    const uint32_t align = std::min(std::min(alignment(input), alignment(weight)), std::min(alignment(output), bias ? alignment(bias) : 256u));
    const TuneKey key{out_features, rows, in_features, dtype, bias ? 1 : 0, align};
    auto found = ctx->tuner->choices.find(key);
    if (found == ctx->tuner->choices.end() || !found->second.pinned || !found->second.use_lt) return 2;
    return run_lt_algo(ctx, input, weight, bias, output, rows, out_features, in_features, dtype, found->second.algo);
}
// Bitwise comparison of two device buffers on the context stream (synchronizes).
extern "C" int infra_linear_bitwise_equal(void* handle, const void* a, const void* b, size_t bytes, int* equal) {
    if (!handle || !a || !b || !equal || bytes % 4 != 0) return fail("unsupported compare arguments", -1);
    auto* ctx = static_cast<Context*>(handle);
    if (!ctx->tuner) ctx->tuner = global_tuner();
    if (!ctx->tuner) return fail("tuner allocation", -1);
    Tuner& t = *ctx->tuner;
    if (!t.mismatch && cudaMalloc(&t.mismatch, sizeof(unsigned)) != cudaSuccess) return fail("compare counter", -1);
    if (cudaMemsetAsync(t.mismatch, 0, sizeof(unsigned), ctx->stream) != cudaSuccess) return fail("compare memset", -1);
    const size_t words = bytes / 4;
    unsigned blocks = unsigned(std::min<size_t>((words + 255) / 256, 4096));
    count_mismatch<<<blocks, 256, 0, ctx->stream>>>(static_cast<const uint32_t*>(a), static_cast<const uint32_t*>(b), words, t.mismatch);
    unsigned mismatches = 1;
    if (cudaMemcpyAsync(&mismatches, t.mismatch, sizeof(unsigned), cudaMemcpyDeviceToHost, ctx->stream) != cudaSuccess ||
        cudaStreamSynchronize(ctx->stream) != cudaSuccess) return fail("compare copy", -1);
    *equal = mismatches == 0 ? 1 : 0;
    error_message[0] = '\0';
    return 0;
}
extern "C" int infra_linear_pinned_report(void* handle, char* out, int size) {
    auto* ctx = static_cast<Context*>(handle);
    if (!ctx || !out || size < 3) return -1;
    std::string text = ctx->tuner ? ctx->tuner->report + "]" : "[]";
    if (int(text.size()) + 1 > size) return -2;
    std::memcpy(out, text.c_str(), text.size() + 1);
    return 0;
}

// First-use library/module registration only. The graph is discarded without
// instantiation or launch; the dummy allocation is never read or written.
extern "C" int infra_linear_prepare(void* handle, const InfraLinearShape* shapes, size_t count, const char* label) {
    auto* ctx=static_cast<Context*>(handle);
    if(!ctx || (!shapes && count))return -1;
    const auto started=std::chrono::steady_clock::now();
    void* dummy=nullptr;
    auto allocated=cudaMalloc(&dummy,1u<<20);
    if(allocated!=cudaSuccess)return fail("metadata dummy allocation",int(allocated));
    uint64_t counts[5];std::copy(infra_linear_generic::counts,infra_linear_generic::counts+5,counts);
    int captures=0,errors=0;
    auto capture=[&](int rows,int in,int out,bool bias){
        auto begun=cudaStreamBeginCapture(ctx->stream,cudaStreamCaptureModeThreadLocal);
        if(begun!=cudaSuccess){++errors;cudaGetLastError();return;}
        int code=infra_linear_forward(handle,dummy,dummy,bias?dummy:nullptr,dummy,rows,out,in,0);
        cudaGraph_t graph=nullptr;auto ended=cudaStreamEndCapture(ctx->stream,&graph);
        if(graph)cudaGraphDestroy(graph);
        if(code==0&&ended==cudaSuccess)++captures;else ++errors;
        cudaGetLastError();
    };
    for(size_t i=0;i<count;++i)capture(shapes[i].rows,shapes[i].in_features,shapes[i].out_features,shapes[i].bias!=0);
    std::copy(counts,counts+5,infra_linear_generic::counts);
    cudaFree(dummy);
    double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
    std::fprintf(stderr,"%s seconds=%.6f captures=%d errors=%d graphs_launched=0\n",label?label:"NATIVE_LINEAR_METADATA",seconds,captures,errors);
    return errors?-1:0;
}
