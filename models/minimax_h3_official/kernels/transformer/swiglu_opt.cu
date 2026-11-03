#include "swiglu_opt.cuh"
extern "C" int h3_pointwise_opt_enabled();
extern "C" int h3_swiglu_bf16(const void*, void*, int64_t, int, cudaStream_t);


#include <cstdlib>
#include <cstring>
#include <mutex>
#include <unordered_map>
namespace {
using B=__nv_bfloat16;
// SiLU is materialized in BF16 before multiplication in the accepted kernel.
// This table computes that identical mapping for every possible BF16 bit pattern.
__device__ B h3_silu_bf16_mapping[65536];
__global__ void initialize_silu_mapping() {
    unsigned bits=blockIdx.x*blockDim.x+threadIdx.x;
    float gate=__bfloat162float(__ushort_as_bfloat16(static_cast<unsigned short>(bits)));
    h3_silu_bf16_mapping[bits]=__float2bfloat16_rn(gate/(1.f+expf(-gate)));
}
struct MappingState {cudaEvent_t ready=nullptr;cudaStream_t producer=nullptr;};
int prepare_silu_mapping(cudaStream_t stream) {
    static std::mutex mutex;
    static std::unordered_map<int,MappingState> devices;
    int device=0;cudaError_t status=cudaGetDevice(&device);if(status)return status;
    std::lock_guard<std::mutex> guard(mutex);
    auto& state=devices[device];
    if(!state.ready) {
        cudaEvent_t event=nullptr;
        status=cudaEventCreateWithFlags(&event,cudaEventDisableTiming);if(status)return status;
        initialize_silu_mapping<<<256,256,0,stream>>>();
        status=cudaGetLastError();if(status){cudaEventDestroy(event);return status;}
        status=cudaEventRecord(event,stream);if(status){cudaEventDestroy(event);return status;}
        // One tiny event/table per CUDA device, retained for this module's lifetime.
        state.ready=event;state.producer=stream;
    }
    return state.producer==stream?cudaSuccess:cudaStreamWaitEvent(stream,state.ready,0);
}
__device__ __forceinline__ unsigned mapped_pair(unsigned value,unsigned gate) {
    unsigned result=0;
    #pragma unroll
    for(int j=0;j<2;++j) {
        unsigned gb=(gate>>(16*j))&65535,vb=(value>>(16*j))&65535;
        B activation=__ldg(h3_silu_bf16_mapping+gb);
        B output=__float2bfloat16_rn(__bfloat162float(__ushort_as_bfloat16(static_cast<unsigned short>(vb)))*__bfloat162float(activation));
        result|=unsigned(__bfloat16_as_ushort(output))<<(16*j);
    }
    return result;
}
__global__ void mapped_swiglu_row(const uint4* x,uint4* out) {
    unsigned row=blockIdx.x;
    uint64_t input=uint64_t(row)*2*h3_swiglu_opt::vectors_per_row;
    uint64_t output=uint64_t(row)*h3_swiglu_opt::vectors_per_row;
    for(unsigned c=threadIdx.x;c<h3_swiglu_opt::vectors_per_row;c+=128) {
        const uint4 v=x[input+c],g=x[input+c+h3_swiglu_opt::vectors_per_row];
        out[output+c]=make_uint4(mapped_pair(v.x,g.x),mapped_pair(v.y,g.y),mapped_pair(v.z,g.z),mapped_pair(v.w,g.w));
    }
}
bool mapping_enabled() {
    static const bool enabled=[](){const char* v=std::getenv("H3_SWIGLU_LUT");return !v||std::strcmp(v,"0");}();
    return enabled;
}
}

// Selected from the predeclared row/flat microbenchmark candidates. Both use
// the identical accepted arithmetic; only launch/indexing differs.
extern "C" int h3_opt_swiglu_bf16(const void* x, void* out, int64_t rows, int cols, cudaStream_t stream) {
    if (!h3_pointwise_opt_enabled() || !h3_swiglu_opt::supported(x, out, rows, cols))
        return h3_swiglu_bf16(x, out, rows, cols, stream);
    if(rows>4096 && mapping_enabled()) {
        int status=prepare_silu_mapping(stream);if(status)return status;
        mapped_swiglu_row<<<unsigned(rows),128,0,stream>>>((const uint4*)x,(uint4*)out);
        return cudaGetLastError();
    }
    if (rows <= 4096) return h3_swiglu_opt::launch<4>(x, out, rows, stream);
    return h3_swiglu_opt::launch<1>(x, out, rows, stream);
}
