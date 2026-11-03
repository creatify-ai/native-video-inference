// Exact host-inference candidate: same RMS reduction and BF16 boundaries as ops.cu.
// PyTorch-derived RMS traversal/reduction: see ../../../ops/NOTICE.md and LICENSE.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cstdlib>
#include <cstring>

extern "C" int h3_pointwise_opt_enabled();
extern "C" int h3_rms_bf16(const void*,const void*,void*,int64_t,int,float,cudaStream_t);
extern "C" int h3_modulate_bf16(void*,const void*,const void*,const int64_t*,int64_t,int,int,cudaStream_t);
extern "C" int h3_gated_bf16(const void*,const void*,const void*,const int64_t*,void*,int64_t,int,int,cudaStream_t);

namespace h3_pw_norm_mod {
using B = __nv_bfloat16;
constexpr int C = 5376;
__device__ __forceinline__ float f(B x) { return __bfloat162float(x); }
__device__ __forceinline__ B b(float x) { return __float2bfloat16_rn(x); }
__device__ __forceinline__ float r(float x) { return f(b(x)); }

template<bool Gated,bool FloatOutput=false,bool SplitIndex=false>
__global__ void fused_scalar(const B* input, const B* branch, const B* gate,
                      const B* weight, const B* scale, const B* shift,
                      const int64_t* index, B* hidden, B* out, int stride, float eps,
                      const int64_t* mod_index=nullptr,int mod_stride=0) {
    __shared__ float sums[4];
    const int lane=threadIdx.x, warp=threadIdx.y, tid=lane+32*warp;
    const int64_t row=int64_t(blockIdx.x)*C;
    const int64_t table=index[blockIdx.x]*int64_t(stride);
    const int64_t mod_table=SplitIndex?mod_index[blockIdx.x]*int64_t(mod_stride):table;
    float sum=0.f;
    // Do not change this traversal: each thread accumulates consecutive groups
    // of four with explicit FMA, exactly as the accepted standalone RMS kernel.
    for(int v=tid; v<C/4; v+=128) {
        #pragma unroll
        for(int j=0;j<4;++j) {
            const int col=4*v+j;
            float value=f(input[row+col]);
            if constexpr(Gated) {
                value=r(__fadd_rn(value,r(__fmul_rn(f(gate[table+col]),f(branch[row+col])))));
                hidden[row+col]=b(value);
            }
            sum=fmaf(value,value,sum);
        }
    }
    for(int offset=16;offset;offset>>=1)
        sum=__shfl_down_sync(0xffffffff,sum,offset)+sum;
    for(int offset=2;offset;offset>>=1) {
        if(!lane && warp>=offset && warp<offset*2) sums[warp-offset]=sum;
        __syncthreads();
        if(!lane && warp<offset) sum=sums[warp]+sum;
        __syncthreads();
    }
    if(!lane && !warp) sums[0]=__fdiv_rn(sum,float(C));
    __syncthreads();
    const float rstd=rsqrtf(sums[0]+eps);
    for(int v=tid;v<C/4;v+=128) {
        #pragma unroll
        for(int j=0;j<4;++j) {
            const int col=4*v+j;
            const float value=f((Gated?hidden:input)[row+col]);
            const float norm=r(__fmul_rn(f(weight[col]),__fmul_rn(rstd,value)));
            const float factor=r(__fadd_rn(1.f,f(scale[mod_table+col])));
            const float product=r(__fmul_rn(norm,factor));
            const B result=b(__fadd_rn(product,f(shift[mod_table+col])));
            if constexpr(FloatOutput) reinterpret_cast<float*>(out)[row+col]=f(result);
            else out[row+col]=result;
        }
    }
}

__device__ __forceinline__ void load4(const B* p,float (&v)[4]) {
    uint2 x=*reinterpret_cast<const uint2*>(p);
    v[0]=f(__ushort_as_bfloat16(x.x));v[1]=f(__ushort_as_bfloat16(x.x>>16));
    v[2]=f(__ushort_as_bfloat16(x.y));v[3]=f(__ushort_as_bfloat16(x.y>>16));
}
__device__ __forceinline__ uint2 pack4(const B (&v)[4]) {
    return make_uint2(unsigned(__bfloat16_as_ushort(v[0]))|(unsigned(__bfloat16_as_ushort(v[1]))<<16),
        unsigned(__bfloat16_as_ushort(v[2]))|(unsigned(__bfloat16_as_ushort(v[3]))<<16));
}
__device__ __forceinline__ void unpack4(uint2 x,float (&v)[4]) {
    v[0]=f(__ushort_as_bfloat16(x.x));v[1]=f(__ushort_as_bfloat16(x.x>>16));
    v[2]=f(__ushort_as_bfloat16(x.y));v[3]=f(__ushort_as_bfloat16(x.y>>16));
}
template<bool Gated,bool FloatOutput=false,bool SplitIndex=false>
__global__ void fused_vector(const B* input,const B* branch,const B* gate,const B* weight,
    const B* scale,const B* shift,const int64_t* index,B* hidden,B* out,int stride,float eps,
    const int64_t* mod_index=nullptr,int mod_stride=0) {
    __shared__ float sums[4];
    const int lane=threadIdx.x,warp=threadIdx.y,tid=lane+32*warp;
    const int64_t row=int64_t(blockIdx.x)*C,table=index[blockIdx.x]*int64_t(stride);
    const int64_t mod_table=SplitIndex?mod_index[blockIdx.x]*int64_t(mod_stride):table;
    float sum=0.f;
    #pragma unroll
    for(int iteration=0;iteration<11;++iteration) {
        const int vector=tid+128*iteration;
        if(vector<C/4) {
            const int col=4*vector;
            float values[4];load4(input+row+col,values);
            if constexpr(Gated) {
                float branches[4],gates[4];B rounded[4];
                load4(branch+row+col,branches);load4(gate+table+col,gates);
                #pragma unroll
                for(int j=0;j<4;++j) {
                    values[j]=r(__fadd_rn(values[j],r(__fmul_rn(gates[j],branches[j]))));
                    rounded[j]=b(values[j]);
                }
                *reinterpret_cast<uint2*>(hidden+row+col)=pack4(rounded);
            }
            #pragma unroll
            for(int j=0;j<4;++j)sum=fmaf(values[j],values[j],sum);

        }
    }
    // Preserve the exact four-warp RMS traversal and reduction from the owner.
    for(int offset=16;offset;offset>>=1)sum=__shfl_down_sync(0xffffffff,sum,offset)+sum;
    for(int offset=2;offset;offset>>=1) {
        if(!lane && warp>=offset && warp<offset*2)sums[warp-offset]=sum;
        __syncthreads();
        if(!lane && warp<offset)sum=sums[warp]+sum;
        __syncthreads();
    }
    if(!lane && !warp)sums[0]=__fdiv_rn(sum,float(C));
    __syncthreads();
    const float rstd=rsqrtf(sums[0]+eps);
    #pragma unroll
    for(int iteration=0;iteration<11;++iteration) {
        const int vector=tid+128*iteration;
        if(vector<C/4) {
            const int col=4*vector;
            float values[4],weights[4],scales[4],shifts[4];B result[4];
            load4((Gated?hidden:input)+row+col,values);
            load4(weight+col,weights);load4(scale+mod_table+col,scales);load4(shift+mod_table+col,shifts);
            #pragma unroll
            for(int j=0;j<4;++j) {
                float norm=r(__fmul_rn(weights[j],__fmul_rn(rstd,values[j])));
                float factor=r(__fadd_rn(1.f,scales[j]));
                float product=r(__fmul_rn(norm,factor));
                result[j]=b(__fadd_rn(product,shifts[j]));
            }
            if constexpr(FloatOutput) {
                #pragma unroll
                for(int j=0;j<4;++j)reinterpret_cast<float*>(out)[row+col+j]=f(result[j]);
            } else *reinterpret_cast<uint2*>(out+row+col)=pack4(result);
        }
    }
}

bool vector_enabled() {
    static const bool enabled=[](){const char* v=std::getenv("H3_NORM_VECTOR");return !v||std::strcmp(v,"0");}();
    return enabled;
}
template<bool Gated,bool FloatOutput=false,bool SplitIndex=false>
void launch_fused(const B* input,const B* branch,const B* gate,const B* weight,
    const B* scale,const B* shift,const int64_t* index,B* hidden,B* out,int64_t rows,int stride,
    float eps,cudaStream_t stream,const int64_t* mod_index=nullptr,int mod_stride=0) {
    uintptr_t addresses=reinterpret_cast<uintptr_t>(input)|reinterpret_cast<uintptr_t>(branch)|
        reinterpret_cast<uintptr_t>(gate)|reinterpret_cast<uintptr_t>(weight)|reinterpret_cast<uintptr_t>(scale)|
        reinterpret_cast<uintptr_t>(shift)|reinterpret_cast<uintptr_t>(out);
    // Preserve the original general ABI for non-vector-aligned external views.
    if(vector_enabled() && !(addresses&7) && stride%4==0 && (!SplitIndex || mod_stride%4==0))
        fused_vector<Gated,FloatOutput,SplitIndex><<<unsigned(rows),dim3(32,4),0,stream>>>(input,branch,gate,weight,
            scale,shift,index,hidden,out,stride,eps,mod_index,mod_stride);
    else fused_scalar<Gated,FloatOutput,SplitIndex><<<unsigned(rows),dim3(32,4),0,stream>>>(input,branch,gate,weight,
            scale,shift,index,hidden,out,stride,eps,mod_index,mod_stride);
}

__global__ void gated_row(const B* input,const B* branch,const B* gate,
                         const int64_t* index,B* out,int stride) {
    const int64_t row=int64_t(blockIdx.x)*C;
    const int64_t table=index[blockIdx.x]*int64_t(stride);
    for(int col=threadIdx.x;col<C;col+=blockDim.x)
        out[row+col]=b(__fadd_rn(f(input[row+col]),r(__fmul_rn(f(gate[table+col]),f(branch[row+col])))));
}

bool valid(int64_t rows,int cols,int stride) {
    return rows>0 && rows<=0x7fffffff && cols==C && stride>=C;
}
}

extern "C" int h3_pw_rms_mod_bf16(const void* x,const void* weight,const void* scale,
                                  const void* shift,const int64_t* index,void* out,
                                  int64_t rows,int cols,int stride,float eps,cudaStream_t stream) {
    using namespace h3_pw_norm_mod;
    if(!valid(rows,cols,stride)||!x||!weight||!scale||!shift||!index||!out||x==out||!(eps>0)) return cudaErrorInvalidValue;
    if(!h3_pointwise_opt_enabled()) {
        int error=h3_rms_bf16(x,weight,out,rows,cols,eps,stream);
        return error ? error : h3_modulate_bf16(out,scale,shift,index,rows,cols,stride,stream);
    }
    launch_fused<false>((const B*)x,nullptr,nullptr,(const B*)weight,(const B*)scale,(const B*)shift,index,
        nullptr,(B*)out,rows,stride,eps,stream);
    return cudaGetLastError();
}

extern "C" int h3_pw_gated_rms_mod_bf16(void* hidden,const void* branch,const void* gate,
                                        const void* weight,const void* scale,const void* shift,
                                        const int64_t* index,void* out,int64_t rows,int cols,
                                        int stride,float eps,cudaStream_t stream) {
    using namespace h3_pw_norm_mod;
    if(!valid(rows,cols,stride)||!hidden||!branch||!gate||!weight||!scale||!shift||!index||!out||
       hidden==out||hidden==branch||out==branch||!(eps>0)) return cudaErrorInvalidValue;
    if(!h3_pointwise_opt_enabled()) {
        int error=h3_gated_bf16(hidden,branch,gate,index,hidden,rows,cols,stride,stream);
        if(error) return error;
        error=h3_rms_bf16(hidden,weight,out,rows,cols,eps,stream);
        return error ? error : h3_modulate_bf16(out,scale,shift,index,rows,cols,stride,stream);
    }
    // One CTA owns a row. Each thread writes and subsequently reads only its
    // own columns. All reduction barriers are uniform; no inter-CTA aliasing.
    launch_fused<true>((const B*)hidden,(const B*)branch,(const B*)gate,(const B*)weight,(const B*)scale,
        (const B*)shift,index,(B*)hidden,(B*)out,rows,stride,eps,stream);
    return cudaGetLastError();
}

extern "C" int h3_pw_gated_bf16(const void* hidden,const void* branch,const void* gate,
                                const int64_t* index,void* out,int64_t rows,int cols,
                                int stride,cudaStream_t stream) {
    using namespace h3_pw_norm_mod;
    if(!valid(rows,cols,stride)||!hidden||!branch||!gate||!index||!out||out==branch) return cudaErrorInvalidValue;
    if(!h3_pointwise_opt_enabled()) return h3_gated_bf16(hidden,branch,gate,index,out,rows,cols,stride,stream);
    gated_row<<<unsigned(rows),256,0,stream>>>((const B*)hidden,(const B*)branch,(const B*)gate,index,(B*)out,stride);
    return cudaGetLastError();
}

// Final block: gate uses modality-expanded AdaLN indices; final modulation uses
// timestep indices. Both BF16 boundaries and the original RMS tree are retained.
extern "C" int h3_pw_final_gated_norm_mod_cast(void* hidden,const void* branch,const void* gate,
    const void* weight,const void* scale,const void* shift,const int64_t* gate_index,
    const int64_t* mod_index,float* out,int64_t rows,int cols,int gate_stride,int mod_stride,
    float eps,cudaStream_t stream) {
    using namespace h3_pw_norm_mod;
    if(!valid(rows,cols,gate_stride)||mod_stride<cols||!hidden||!branch||!gate||!weight||!scale||!shift||
       !gate_index||!mod_index||!out||hidden==out||hidden==branch||out==branch||!(eps>0))return cudaErrorInvalidValue;
    launch_fused<true,true,true>((const B*)hidden,(const B*)branch,(const B*)gate,(const B*)weight,(const B*)scale,
        (const B*)shift,gate_index,(B*)hidden,(B*)out,rows,gate_stride,eps,stream,mod_index,mod_stride);
    return cudaGetLastError();
}
