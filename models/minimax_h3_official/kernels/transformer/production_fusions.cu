// Fused pointwise operations matching the original Triton fusion reference.
// FP32 intermediates and RoPE tables, with BF16 stores only at the fused outputs.
// Historical eager kernels remain separate for pinned old reference artifacts.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cmath>
namespace {
using B=__nv_bfloat16;
__device__ __forceinline__ float f(B x){return __bfloat162float(x);}
__device__ __forceinline__ B b(float x){return __float2bfloat16_rn(x);}
constexpr int C=5376, THREADS=512, BLOCK=8192;
__device__ __forceinline__ float lane8(uint4 v,int j){
    const unsigned word=j<2?v.x:j<4?v.y:j<6?v.z:v.w;
    return f(__ushort_as_bfloat16(uint16_t(word>>((j&1)*16))));
}
__device__ __forceinline__ uint4 pack8(const float* values){
    uint4 out;
    out.x=unsigned(__bfloat16_as_ushort(b(values[0])))|(unsigned(__bfloat16_as_ushort(b(values[1])))<<16);
    out.y=unsigned(__bfloat16_as_ushort(b(values[2])))|(unsigned(__bfloat16_as_ushort(b(values[3])))<<16);
    out.z=unsigned(__bfloat16_as_ushort(b(values[4])))|(unsigned(__bfloat16_as_ushort(b(values[5])))<<16);
    out.w=unsigned(__bfloat16_as_ushort(b(values[6])))|(unsigned(__bfloat16_as_ushort(b(values[7])))<<16);
    return out;
}
__global__ void add_vectorized(const B* x,const B* y,B* out,int64_t vectors){
    for(int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<vectors;i+=int64_t(gridDim.x)*blockDim.x){
        const uint4 a=reinterpret_cast<const uint4*>(x)[i];
        const uint4 c=reinterpret_cast<const uint4*>(y)[i];
        float result[8];
        #pragma unroll
        for(int j=0;j<8;j++)result[j]=__fadd_rn(lane8(a,j),lane8(c,j));
        reinterpret_cast<uint4*>(out)[i]=pack8(result);
    }
}
__global__ void add_scalar(const B* x,const B* y,B* out,int64_t n){
    for(int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=int64_t(gridDim.x)*blockDim.x)
        out[i]=b(__fadd_rn(f(x[i]),f(y[i])));
}
__device__ __forceinline__ float full_div(float x,float y){
    float out;asm("div.full.f32 %0, %1, %2;" : "=f"(out) : "f"(x),"f"(y));return out;
}
__device__ __forceinline__ float triton_rsqrt(float x){
    float out;asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(out) : "f"(x));return out;
}
template<bool Gated,bool Vectorized>
__global__ void norm_mod(const B* input,const B* branch,const B* gate,const B* weight,
    const B* scale,const B* shift,const int64_t* indices,B* hidden,B* out,int cols,int stride,float eps){
    __shared__ float sums[16];
    const int tid=threadIdx.x,lane=tid%32,warp=tid/32;
    const int64_t row=int64_t(blockIdx.x)*C,table=indices[blockIdx.x]*int64_t(stride);
    float value[BLOCK/THREADS];
    if constexpr(Vectorized){
        #pragma unroll
        for(int tile=0;tile<2;tile++){
            const int col=tid*8+tile*THREADS*8;
            uint4 x{},br{},g{};
            if(col<C){
                x=*reinterpret_cast<const uint4*>(input+row+col);
                if constexpr(Gated){
                    br=*reinterpret_cast<const uint4*>(branch+row+col);
                    g=*reinterpret_cast<const uint4*>(gate+table+col);
                }
            }
            #pragma unroll
            for(int j=0;j<8;j++){
                float v=lane8(x,j);
                if constexpr(Gated)v=fmaf(lane8(g,j),lane8(br,j),v);
                value[tile*8+j]=v;
            }
            if constexpr(Gated)if(col<C)*reinterpret_cast<uint4*>(hidden+row+col)=pack8(value+tile*8);
        }
    }else{
    #pragma unroll
    for(int j=0;j<BLOCK/THREADS;j++){
        // The reference's aligned Triton layout gives each thread eight adjacent
        // values in each 4096-column tile, then reduces its registers in order.
        int col=tid*8+j%8+(j/8)*THREADS*8;
        value[j]=0;
        if(col<C){
            value[j]=f(input[row+col]);
            if constexpr(Gated){value[j]=fmaf(f(gate[table+col]),f(branch[row+col]),value[j]);hidden[row+col]=b(value[j]);}
        }
    }
    }
    float sum=__fmul_rn(value[1],value[1]);
    sum=fmaf(value[0],value[0],sum);
    #pragma unroll
    for(int j=2;j<BLOCK/THREADS;j++){
        if constexpr(Gated)sum=fmaf(value[j],value[j],sum);
        else if(j%2==0)sum=fmaf(value[j],value[j],sum);
        else sum=__fadd_rn(sum,__fmul_rn(value[j],value[j]));
    }
    #pragma unroll
    for(int offset=16;offset;offset/=2)sum+=__shfl_xor_sync(0xffffffff,sum,offset);
    if(lane==0)sums[warp]=sum;
    __syncthreads();
    if(warp==0){
        sum=lane<16?sums[lane]:0;
        #pragma unroll
        for(int offset=8;offset;offset/=2)sum+=__shfl_xor_sync(0xffffffff,sum,offset);
        // Match the original div.full + add.f32 sequence. Explicit add.rn
        // inhibits ptxas contraction that the original kernel permits.
        if(lane==0)sums[0]=triton_rsqrt(full_div(sum,float(cols))+eps);
    }
    __syncthreads();
    const float inv=sums[0];
    if constexpr(Vectorized){
        #pragma unroll
        for(int tile=0;tile<2;tile++){
            const int col=tid*8+tile*THREADS*8;
            if(col<C){
                const uint4 w=*reinterpret_cast<const uint4*>(weight+col);
                const uint4 sc=*reinterpret_cast<const uint4*>(scale+table+col);
                const uint4 sh=*reinterpret_cast<const uint4*>(shift+table+col);
                float result[8];
                #pragma unroll
                for(int j=0;j<8;j++){
                    const float norm=(value[tile*8+j]*inv)*lane8(w,j);
                    result[j]=fmaf(norm,1.f+lane8(sc,j),lane8(sh,j));
                }
                *reinterpret_cast<uint4*>(out+row+col)=pack8(result);
            }
        }
    }else{
    #pragma unroll
    for(int j=0;j<BLOCK/THREADS;j++){
        int col=tid*8+j%8+(j/8)*THREADS*8;
        if(col<C){
            const float norm=(value[j]*inv)*f(weight[col]);
            out[row+col]=b(fmaf(norm,1.f+f(scale[table+col]),f(shift[table+col])));
        }
    }
    }
}
__global__ void swiglu(const B* x,B* out,int64_t n){
    constexpr int cols=14336;
    for(int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=int64_t(gridDim.x)*blockDim.x){
        const int64_t offset=(i/cols)*(2*int64_t(cols))+i%cols;
        const float gate=f(x[offset+cols]);
        // Match the pinned Triton PTX, including its exp approximation and
        // div.full rounding. libdevice exp2f / C++ division cross BF16 ties
        // differently on real activations, even when random small probes pass.
        float exponential,sigmoid;
        const float exponent=__fmul_rn(-gate,1.4426950408889634f);
        asm("ex2.approx.f32 %0, %1;" : "=f"(exponential) : "f"(exponent));
        const float denominator=__fadd_rn(1.f,exponential);
        asm("div.full.f32 %0, %1, %2;" : "=f"(sigmoid) : "f"(1.f),"f"(denominator));
        out[i]=b(f(x[offset])*(gate*sigmoid));
    }
}
__global__ void swiglu_vectorized(const B* x,B* out,unsigned vectors){
    constexpr int cols=14336,words=cols/8;
    // rows<=2^20 gives at most 1,879,048,192 vectors; only byte offsets need 64 bits.
    for(unsigned i=blockIdx.x*blockDim.x+threadIdx.x;i<vectors;i+=gridDim.x*blockDim.x){
        const int64_t offset=int64_t(i/words)*(2*cols)+(i%words)*8;
        const uint4 value=*reinterpret_cast<const uint4*>(x+offset);
        const uint4 gates=*reinterpret_cast<const uint4*>(x+offset+cols);
        float result[8];
        #pragma unroll
        for(int j=0;j<8;j++){
            const float gate=lane8(gates,j);
            float exponential,sigmoid;
            const float exponent=__fmul_rn(-gate,1.4426950408889634f);
            asm("ex2.approx.f32 %0, %1;" : "=f"(exponential) : "f"(exponent));
            const float denominator=__fadd_rn(1.f,exponential);
            asm("div.full.f32 %0, %1, %2;" : "=f"(sigmoid) : "f"(1.f),"f"(denominator));
            result[j]=lane8(value,j)*(gate*sigmoid);
        }
        *reinterpret_cast<uint4*>(out+int64_t(i)*8)=pack8(result);
    }
}
// Eight independent 128-column heads share a CTA. Each warp reproduces the
// four original 32-lane reductions in registers, including lane-zero broadcast.
// This removes shared-memory barriers without changing the FP32 reduction tree.
__global__ void qk_rope_warp(const B* q,const B* k,const B* qw,const B* kw,
    const float* cosine,const float* sine,B* qn,B* kn,B* qo,B* ko,
    int64_t rows,int /*heads*/,float qe,float ke){
    const int lane=threadIdx.x%32;
    // The public bound is tokens<=2^20, heads=56, so row IDs fit uint32.
    const unsigned combined=blockIdx.x*8+threadIdx.x/32;
    const unsigned row_count=unsigned(rows);
    if(combined>=2*row_count)return;
    const bool is_k=combined>=row_count;
    const unsigned row=is_k?combined-row_count:combined;
    const int64_t base=int64_t(row)*128;
    const B* x=is_k?k:q;const B* weight=is_k?kw:qw;
    B* norm=is_k?kn:qn;B* out=is_k?ko:qo;
    float value[4],partial[4],normalized[4];
    #pragma unroll
    for(int group=0;group<4;group++){
        const float v=f(x[base+group*32+lane]);value[group]=v;
        float sum=__fmul_rn(v,v);
        sum=fmaf(v,v,__shfl_xor_sync(0xffffffff,sum,16));
        #pragma unroll
        for(int offset=8;offset;offset/=2)sum+=__shfl_xor_sync(0xffffffff,sum,offset);
        partial[group]=__shfl_sync(0xffffffff,sum,0);
    }
    const float variance=full_div((partial[0]+partial[2])+(partial[1]+partial[3]),128.f);
    const float inv=triton_rsqrt(__fadd_rn(variance,is_k?ke:qe));
    #pragma unroll
    for(int group=0;group<4;group++)normalized[group]=(value[group]*inv)*f(weight[group*32+lane]);
    const float p0=__shfl_xor_sync(0xffffffff,normalized[0],16);
    const float p1=__shfl_xor_sync(0xffffffff,normalized[1],16);
    const float p2=__shfl_xor_sync(0xffffffff,normalized[2],16);
    #pragma unroll
    for(int group=0;group<4;group++){
        const int col=group*32+lane;
        float rotated_value=normalized[group];
        // Every lane participated in the shuffles, including bypass columns.
        if(col<96){
            const int partner=(col<48?col+48:col-48)/32;
            float rotated=partner==0?p0:partner==1?p1:p2;
            if(col<48)rotated=-rotated;
            const int64_t pos=int64_t(row/56)*96+col;
            rotated_value=fmaf(normalized[group],cosine[pos],rotated*sine[pos]);
        }
        norm[base+col]=b(normalized[group]);out[base+col]=b(rotated_value);
    }
}
}
extern "C" int h3_prod_add_bf16(const void* x,const void* y,void* out,int64_t n,cudaStream_t stream){
    if(!x||!y||!out||n<1)return cudaErrorInvalidValue;
    const bool vectorized=n%8==0 &&
        ((reinterpret_cast<uintptr_t>(x)|reinterpret_cast<uintptr_t>(y)|reinterpret_cast<uintptr_t>(out))&15u)==0;
    const int64_t count=vectorized?n/8:n;
    const unsigned blocks=unsigned(count/256+(count%256!=0)>65535?65535:count/256+(count%256!=0));
    if(vectorized)add_vectorized<<<blocks,256,0,stream>>>((const B*)x,(const B*)y,(B*)out,count);
    else add_scalar<<<blocks,256,0,stream>>>((const B*)x,(const B*)y,(B*)out,count);
    return cudaGetLastError();
}
extern "C" int h3_prod_rms_mod_bf16(const void* x,const void* weight,const void* scale,
    const void* shift,const int64_t* indices,void* out,int64_t rows,int cols,int stride,float eps,cudaStream_t stream){
    if(!x||!weight||!scale||!shift||!indices||!out||x==out||rows<1||rows>1048576||cols!=C||stride<C||!(eps>0))return cudaErrorInvalidValue;
    if(((reinterpret_cast<uintptr_t>(x)|reinterpret_cast<uintptr_t>(weight)|reinterpret_cast<uintptr_t>(scale)|reinterpret_cast<uintptr_t>(shift)|reinterpret_cast<uintptr_t>(out)|uintptr_t(stride)*2)&15u)==0)
        norm_mod<false,true><<<unsigned(rows),THREADS,0,stream>>>((const B*)x,nullptr,nullptr,(const B*)weight,(const B*)scale,(const B*)shift,indices,nullptr,(B*)out,cols,stride,eps);
    else norm_mod<false,false><<<unsigned(rows),THREADS,0,stream>>>((const B*)x,nullptr,nullptr,(const B*)weight,(const B*)scale,(const B*)shift,indices,nullptr,(B*)out,cols,stride,eps);
    return cudaGetLastError();
}
extern "C" int h3_prod_gated_rms_mod_bf16(void* hidden,const void* branch,const void* gate,
    const void* weight,const void* scale,const void* shift,const int64_t* indices,void* out,
    int64_t rows,int cols,int stride,float eps,cudaStream_t stream){
    if(!hidden||!branch||!gate||!weight||!scale||!shift||!indices||!out||hidden==out||hidden==branch||out==branch||rows<1||rows>1048576||cols!=C||stride<C||!(eps>0))return cudaErrorInvalidValue;
    if(((reinterpret_cast<uintptr_t>(hidden)|reinterpret_cast<uintptr_t>(branch)|reinterpret_cast<uintptr_t>(gate)|reinterpret_cast<uintptr_t>(weight)|reinterpret_cast<uintptr_t>(scale)|reinterpret_cast<uintptr_t>(shift)|reinterpret_cast<uintptr_t>(out)|uintptr_t(stride)*2)&15u)==0)
        norm_mod<true,true><<<unsigned(rows),THREADS,0,stream>>>((const B*)hidden,(const B*)branch,(const B*)gate,(const B*)weight,(const B*)scale,(const B*)shift,indices,(B*)hidden,(B*)out,cols,stride,eps);
    else norm_mod<true,false><<<unsigned(rows),THREADS,0,stream>>>((const B*)hidden,(const B*)branch,(const B*)gate,(const B*)weight,(const B*)scale,(const B*)shift,indices,(B*)hidden,(B*)out,cols,stride,eps);
    return cudaGetLastError();
}
extern "C" int h3_prod_swiglu_bf16(const void* x,void* out,int64_t rows,int cols,cudaStream_t stream){
    if(!x||!out||x==out||rows<1||rows>1048576||cols!=14336)return cudaErrorInvalidValue;
    const int64_t n=rows*cols;unsigned blocks=unsigned((n+255)/256);if(blocks>65535)blocks=65535;
    if(((reinterpret_cast<uintptr_t>(x)|reinterpret_cast<uintptr_t>(out))&15u)==0){
        const int64_t vectors=n/8;blocks=unsigned((vectors+255)/256);if(blocks>65535)blocks=65535;
        swiglu_vectorized<<<blocks,256,0,stream>>>((const B*)x,(B*)out,vectors);
    }else swiglu<<<blocks,256,0,stream>>>((const B*)x,(B*)out,n);
    return cudaGetLastError();
}
extern "C" int h3_prod_qk_rope(const void* q,const void* k,const void* qw,const void* kw,
    const void* cs,const void* sn,void* qn,void* kn,void* qo,void* ko,int64_t tokens,int heads,float qe,float ke,cudaStream_t stream){
    if(!q||!k||!qw||!kw||!cs||!sn||!qn||!kn||!qo||!ko||tokens<1||tokens>1048576||heads!=56||!(qe>0)||!(ke>0))return cudaErrorInvalidValue;
    if(q==k||qo==k||ko==q||qn==q||qn==k||kn==q||kn==k||qn==kn||qn==qo||qn==ko||kn==qo||kn==ko||qo==ko)return cudaErrorInvalidValue;
    qk_rope_warp<<<unsigned((2*tokens*heads+7)/8),256,0,stream>>>((const B*)q,(const B*)k,(const B*)qw,(const B*)kw,(const float*)cs,(const float*)sn,(B*)qn,(B*)kn,(B*)qo,(B*)ko,tokens*heads,heads,qe,ke);
    return cudaGetLastError();
}
