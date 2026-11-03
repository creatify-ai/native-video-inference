// Exact H3 RMSNorm(128) + partial RoPE(96), paired Q/K. No ATen dependency.
// Reduction sequence follows the accepted ops/ops.cu implementation.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cmath>
#include <cstdint>

using H3QKB = __nv_bfloat16;
extern "C" int h3_pointwise_opt_enabled();
extern "C" int h3_rms_bf16(const void*,const void*,void*,int64_t,int,float,cudaStream_t);
extern "C" int h3_rope_bf16(const void*,const void*,const void*,void*,int64_t,int,int,cudaStream_t);

namespace {
__device__ __forceinline__ float qkf(H3QKB x) { return __bfloat162float(x); }
__device__ __forceinline__ H3QKB qkb(float x) { return __float2bfloat16_rn(x); }

template<int WARPS, bool SGLANG_WARP = false>
__global__ void h3_qk_rope_warp_exact_v1(
    const H3QKB* q,const H3QKB* k,const H3QKB* qw,const H3QKB* kw,
    const H3QKB* cosine,const H3QKB* sine,H3QKB* qn,H3QKB* kn,
    H3QKB* qo,H3QKB* ko,int64_t head_rows,float qeps,float keps) {
    const int lane=threadIdx.x & 31;
    const int64_t combined=int64_t(blockIdx.x)*WARPS+(threadIdx.x>>5);
    if(combined>=2*head_rows) return; // Entire warp follows the same branch.
    const bool is_k=combined>=head_rows;
    const int64_t row=is_k?combined-head_rows:combined;
    const auto* x=is_k?k:q;
    const auto* weight=is_k?kw:qw;
    auto* norm=is_k?kn:qn;
    auto* output=is_k?ko:qo;
    float values[4];
    float sum=0.f;
    #pragma unroll
    for(int j=0;j<4;++j) {
        values[j]=qkf(x[row*128+lane*4+j]);
        sum=fmaf(values[j],values[j],sum);
    }
    // The original 128-thread CTA only loads values in warp 0 for width128.
    // Preserve its exact four-value lane order, downward tree, and the two
    // additions of all-zero warp partials. No reassociation or fast math.
    float variance;
    if constexpr (SGLANG_WARP) {
        // Pinned SGLang impl/norm.cuh apply_norm_warp, used by its H3
        // fused QK norm + RoPE with round_norm_before_rope=true.
        #pragma unroll
        for(int offset=16;offset;offset>>=1)
            sum=__fadd_rn(sum,__shfl_xor_sync(0xffffffff,sum,offset));
        variance=__fmul_rn(sum,1.f/128.f);
    } else {
        #pragma unroll
        for(int offset=16;offset;offset>>=1)
            sum=__fadd_rn(__shfl_down_sync(0xffffffff,sum,offset),sum);
        if(lane==0) {
            sum=__fadd_rn(0.f,sum); // original warp reduction offset 2
            sum=__fadd_rn(0.f,sum); // original warp reduction offset 1
            sum=sum/128.f;
        }
        variance=__shfl_sync(0xffffffff,sum,0);
    }
    const float rstd=rsqrtf(__fadd_rn(variance,is_k?keps:qeps));
    float normalized[4];
    #pragma unroll
    for(int j=0;j<4;++j) {
        const int col=lane*4+j;
        // Materialize the same BF16 normalization boundary in registers.
        normalized[j]=qkf(qkb(__fmul_rn(qkf(weight[col]),__fmul_rn(rstd,values[j]))));
    }
    const int partner_lane=lane<12?lane+12:lane-12;
    const int64_t position=(row/56)*96;
    #pragma unroll
    for(int j=0;j<4;++j) {
        const int col=lane*4+j;
        // Every lane participates, including the unrotated last32 channels.
        float partner=__shfl_sync(0xffffffff,normalized[j],partner_lane & 31);
        if(lane<12) partner=-partner;
        H3QKB value=qkb(normalized[j]);
        if(col<96) {
            if constexpr (SGLANG_WARP) {
                // Match SGLang rotary_add/rotary_sub's typed BF16 expression.
                // Do not impose the eager path's extra materialization barriers.
                if(lane<12)
                    value=qkb(normalized[j])*cosine[position+col]-qkb(-partner)*sine[position+col];
                else
                    value=qkb(normalized[j])*cosine[position+col]+qkb(partner)*sine[position+col];
            } else {
                const float left=qkf(qkb(__fmul_rn(normalized[j],qkf(cosine[position+col]))));
                const float right=qkf(qkb(__fmul_rn(partner,qkf(sine[position+col]))));
                value=qkb(__fadd_rn(left,right));
            }
        }
        // No global partner reads follow the stores. Exact x==output alias is
        // safe because the entire head and all partners are already registers.
        norm[row*128+col]=qkb(normalized[j]);
        output[row*128+col]=value;
    }
}

int check(const void* q,const void* k,const void* qw,const void* kw,const void* cs,const void* sn,
          void* qn,void* kn,void* qo,void* ko,int64_t tokens,int heads,float qe,float ke) {
    if(!q||!k||!qw||!kw||!cs||!sn||!qn||!kn||!qo||!ko||tokens<1||tokens>1048576||heads!=56||!(qe>0)||!(ke>0))
        return cudaErrorInvalidValue;
    if(q==k||qo==k||ko==q||qn==q||qn==k||kn==q||kn==k||qn==kn||qn==qo||qn==ko||kn==qo||kn==ko||qo==ko)
        return cudaErrorInvalidValue;
    return cudaSuccess;
}
}

extern "C" int h3_sglang_qk_rope_v1(
    const void* q,const void* k,const void* qw,const void* kw,const void* cs,const void* sn,
    void* qn,void* kn,void* qo,void* ko,int64_t tokens,int heads,float qe,float ke,cudaStream_t stream) {
    int status=check(q,k,qw,kw,cs,sn,qn,kn,qo,ko,tokens,heads,qe,ke);
    if(status) return status;
    h3_qk_rope_warp_exact_v1<8,true><<<unsigned((2*tokens*heads+7)/8),256,0,stream>>>(
        (const H3QKB*)q,(const H3QKB*)k,(const H3QKB*)qw,(const H3QKB*)kw,
        (const H3QKB*)cs,(const H3QKB*)sn,(H3QKB*)qn,(H3QKB*)kn,
        (H3QKB*)qo,(H3QKB*)ko,tokens*heads,qe,ke);
    return cudaGetLastError();
}

namespace {
__device__ __forceinline__ void load4(const H3QKB* input, float (&value)[4]) {
    const uint2 packed=*reinterpret_cast<const uint2*>(input);
    value[0]=qkf(__ushort_as_bfloat16(static_cast<unsigned short>(packed.x)));
    value[1]=qkf(__ushort_as_bfloat16(static_cast<unsigned short>(packed.x>>16)));
    value[2]=qkf(__ushort_as_bfloat16(static_cast<unsigned short>(packed.y)));
    value[3]=qkf(__ushort_as_bfloat16(static_cast<unsigned short>(packed.y>>16)));
}
__device__ __forceinline__ uint2 pack4(const H3QKB (&value)[4]) {
    return make_uint2(unsigned(__bfloat16_as_ushort(value[0]))|(unsigned(__bfloat16_as_ushort(value[1]))<<16),
                      unsigned(__bfloat16_as_ushort(value[2]))|(unsigned(__bfloat16_as_ushort(value[3]))<<16));
}
// Keep the accepted per-lane four-value accumulation, XOR reduction and BF16
// boundaries. Only memory transactions and optional diagnostic stores change.
// Packed QKV is split in this kernel, avoiding a separate full-tensor copy.
template<bool PackedQkv, bool KeepNorm>
__global__ void sglang_qk_vectorized(const H3QKB* q,const H3QKB* k,
    const H3QKB* qw,const H3QKB* kw,const H3QKB* cosine,const H3QKB* sine,
    H3QKB* qn,H3QKB* kn,H3QKB* qo,H3QKB* ko,H3QKB* vo,
    int64_t head_rows,float qeps,float keps) {
    const int lane=threadIdx.x&31;
    const int64_t combined=int64_t(blockIdx.x)*8+(threadIdx.x>>5);
    if(combined>=2*head_rows)return;
    const bool is_k=combined>=head_rows;
    const int64_t row=is_k?combined-head_rows:combined;
    const int64_t token=row/56,head=row%56;
    const int col=lane*4;
    const int64_t input_offset=PackedQkv?(token*3*56*128+(is_k?56*128:0)+head*128+col):row*128+col;
    const H3QKB* input=PackedQkv?q:(is_k?k:q);
    float values[4],weights[4];load4(input+input_offset,values);load4((is_k?kw:qw)+col,weights);
    float sum=0.f;
    #pragma unroll
    for(int j=0;j<4;++j)sum=fmaf(values[j],values[j],sum);
    #pragma unroll
    for(int offset=16;offset;offset>>=1)sum=__fadd_rn(sum,__shfl_xor_sync(0xffffffff,sum,offset));
    const float rstd=rsqrtf(__fadd_rn(__fmul_rn(sum,1.f/128.f),is_k?keps:qeps));
    float normalized[4];H3QKB rounded[4],result[4];
    #pragma unroll
    for(int j=0;j<4;++j){
        rounded[j]=qkb(__fmul_rn(weights[j],__fmul_rn(rstd,values[j])));
        normalized[j]=qkf(rounded[j]);
    }
    const int partner_lane=lane<12?lane+12:lane-12;
    #pragma unroll
    for(int j=0;j<4;++j){
        float partner=__shfl_sync(0xffffffff,normalized[j],partner_lane&31);
        if(lane<12)partner=-partner;
        H3QKB value=rounded[j];
        if(col+j<96){
            const int64_t position=token*96+col+j;
            if(lane<12)value=rounded[j]*cosine[position]-qkb(-partner)*sine[position];
            else value=rounded[j]*cosine[position]+qkb(partner)*sine[position];
        }
        result[j]=value;
    }
    *reinterpret_cast<uint2*>((is_k?ko:qo)+row*128+col)=pack4(result);
    if constexpr(KeepNorm)*reinterpret_cast<uint2*>((is_k?kn:qn)+row*128+col)=pack4(rounded);
    if constexpr(PackedQkv){
        if(!is_k)*reinterpret_cast<uint2*>(vo+row*128+col)=
            *reinterpret_cast<const uint2*>(q+token*3*56*128+2*56*128+head*128+col);
    }
}
int fast_check(const void* q,const void* k,const void* qw,const void* kw,const void* cs,const void* sn,
    void* qn,void* kn,void* qo,void* ko,int64_t tokens,int heads,float qe,float ke){
    auto aligned=[](const void* p){return p&&uintptr_t(p)%8==0;};
    if(!aligned(q)||!aligned(k)||!aligned(qw)||!aligned(kw)||!aligned(cs)||!aligned(sn)||
       !aligned(qo)||!aligned(ko)||tokens<1||tokens>1048576||heads!=56||!(qe>0)||!(ke>0)||
       (qn==nullptr)!=(kn==nullptr)||qo==ko)return cudaErrorInvalidValue;
    if(qn&&(!aligned(qn)||!aligned(kn)||qn==kn||qn==q||qn==k||kn==q||kn==k||
             qn==qo||qn==ko||kn==qo||kn==ko))return cudaErrorInvalidValue;
    return cudaSuccess;
}
template<bool PackedQkv>
int launch_fast(const void* q,const void* k,const void* qw,const void* kw,const void* cs,const void* sn,
    void* qn,void* kn,void* qo,void* ko,void* vo,int64_t tokens,int heads,float qe,float ke,cudaStream_t stream){
    int status=fast_check(q,k,qw,kw,cs,sn,qn,kn,qo,ko,tokens,heads,qe,ke);if(status)return status;
    if constexpr(PackedQkv){
        if(!vo||uintptr_t(vo)%8||vo==q||vo==qo||vo==ko||qo==q||ko==q)return cudaErrorInvalidValue;
    }else if(q==k||qo==k||ko==q)return cudaErrorInvalidValue;
    if(qn)sglang_qk_vectorized<PackedQkv,true><<<unsigned((2*tokens*heads+7)/8),256,0,stream>>>(
        (const H3QKB*)q,(const H3QKB*)k,(const H3QKB*)qw,(const H3QKB*)kw,(const H3QKB*)cs,(const H3QKB*)sn,
        (H3QKB*)qn,(H3QKB*)kn,(H3QKB*)qo,(H3QKB*)ko,(H3QKB*)vo,tokens*heads,qe,ke);
    else sglang_qk_vectorized<PackedQkv,false><<<unsigned((2*tokens*heads+7)/8),256,0,stream>>>(
        (const H3QKB*)q,(const H3QKB*)k,(const H3QKB*)qw,(const H3QKB*)kw,(const H3QKB*)cs,(const H3QKB*)sn,
        nullptr,nullptr,(H3QKB*)qo,(H3QKB*)ko,(H3QKB*)vo,tokens*heads,qe,ke);
    return cudaGetLastError();
}
}
extern "C" int h3_sglang_qk_rope_v2(const void* q,const void* k,const void* qw,const void* kw,
    const void* cs,const void* sn,void* qn,void* kn,void* qo,void* ko,
    int64_t tokens,int heads,float qe,float ke,cudaStream_t stream){
    return launch_fast<false>(q,k,qw,kw,cs,sn,qn,kn,qo,ko,nullptr,tokens,heads,qe,ke,stream);
}
extern "C" int h3_sglang_qkv_rope_v2(const void* qkv,const void* qw,const void* kw,
    const void* cs,const void* sn,void* qn,void* kn,void* qo,void* ko,void* vo,
    int64_t tokens,int heads,float qe,float ke,cudaStream_t stream){
    return launch_fast<true>(qkv,qkv,qw,kw,cs,sn,qn,kn,qo,ko,vo,tokens,heads,qe,ke,stream);
}

extern "C" int h3_qk_rope_opt_candidate_v1(
    const void* q,const void* k,const void* qw,const void* kw,const void* cs,const void* sn,
    void* qn,void* kn,void* qo,void* ko,int64_t tokens,int heads,float qe,float ke,
    int warps,cudaStream_t stream) {
    int status=check(q,k,qw,kw,cs,sn,qn,kn,qo,ko,tokens,heads,qe,ke);
    if(status) return status;
    const int64_t rows=tokens*heads;
    if(warps==4)
        h3_qk_rope_warp_exact_v1<4><<<unsigned((2*rows+3)/4),128,0,stream>>>(
            (const H3QKB*)q,(const H3QKB*)k,(const H3QKB*)qw,(const H3QKB*)kw,
            (const H3QKB*)cs,(const H3QKB*)sn,(H3QKB*)qn,(H3QKB*)kn,(H3QKB*)qo,(H3QKB*)ko,rows,qe,ke);
    else if(warps==8)
        h3_qk_rope_warp_exact_v1<8><<<unsigned((2*rows+7)/8),256,0,stream>>>(
            (const H3QKB*)q,(const H3QKB*)k,(const H3QKB*)qw,(const H3QKB*)kw,
            (const H3QKB*)cs,(const H3QKB*)sn,(H3QKB*)qn,(H3QKB*)kn,(H3QKB*)qo,(H3QKB*)ko,rows,qe,ke);
    else return cudaErrorInvalidValue;
    return cudaGetLastError();
}

extern "C" int h3_qk_rope_opt_dispatch_v1(
    const void* q,const void* k,const void* qw,const void* kw,const void* cs,const void* sn,
    void* qn,void* kn,void* qo,void* ko,int64_t tokens,int heads,float qe,float ke,cudaStream_t stream) {
    int status=check(q,k,qw,kw,cs,sn,qn,kn,qo,ko,tokens,heads,qe,ke);
    if(status) return status;
    if(h3_pointwise_opt_enabled())
        return h3_qk_rope_opt_candidate_v1(q,k,qw,kw,cs,sn,qn,kn,qo,ko,tokens,heads,qe,ke,4,stream);
    // Keep the original cross-Q/K launch order, including scratch contents.
    status=h3_rms_bf16(q,qw,qn,tokens*heads,128,qe,stream); if(status) return status;
    status=h3_rms_bf16(k,kw,kn,tokens*heads,128,ke,stream); if(status) return status;
    status=h3_rope_bf16(qn,cs,sn,qo,tokens,heads,96,stream); if(status) return status;
    return h3_rope_bf16(kn,cs,sn,ko,tokens,heads,96,stream);
}

namespace {
template<bool Interleaved, bool PackedShuffle>
__global__ void h3_qkv_u32_inplace(const H3QKB* q,const H3QKB* k,
    const H3QKB* qw,const H3QKB* kw,const H3QKB* cosine,const H3QKB* sine,
    H3QKB* qn,H3QKB* kn,H3QKB* qo,H3QKB* ko,H3QKB* vo,
    int64_t head_rows,float qeps,float keps) {
    const int lane=threadIdx.x&31;
    for(uint32_t combined=int64_t(blockIdx.x)*8+(threadIdx.x>>5);combined<2*head_rows;combined+=int64_t(gridDim.x)*8) {
    const bool is_k=Interleaved?(combined%112>=56):(combined>=head_rows);
    const uint32_t row=Interleaved?((combined/112)*56+combined%56):(is_k?combined-uint32_t(head_rows):combined);
    const uint32_t token=row/56,head=row%56;
    const int col=lane*4;
    const int64_t input_offset=true?(int64_t(token)*3*56*128+(is_k?56*128:0)+head*128+col):int64_t(row)*128+col;
    const H3QKB* input=true?q:(is_k?k:q);
    float values[4],weights[4];load4(input+input_offset,values);load4((is_k?kw:qw)+col,weights);
    float sum=0.f;
    #pragma unroll
    for(int j=0;j<4;++j)sum=fmaf(values[j],values[j],sum);
    #pragma unroll
    for(int offset=16;offset;offset>>=1)sum=__fadd_rn(sum,__shfl_xor_sync(0xffffffff,sum,offset));
    const float rstd=rsqrtf(__fadd_rn(__fmul_rn(sum,1.f/128.f),is_k?keps:qeps));
    float normalized[4];H3QKB rounded[4],result[4];
    #pragma unroll
    for(int j=0;j<4;++j){
        rounded[j]=qkb(__fmul_rn(weights[j],__fmul_rn(rstd,values[j])));
        normalized[j]=qkf(rounded[j]);
    }
    const int partner_lane=lane<12?lane+12:lane-12;
    uint2 partners;
    if constexpr(PackedShuffle) {
        const uint2 bits=pack4(rounded);
        partners=make_uint2(__shfl_sync(0xffffffff,bits.x,partner_lane&31),__shfl_sync(0xffffffff,bits.y,partner_lane&31));
    }
    #pragma unroll
    for(int j=0;j<4;++j){
        float partner;
        if constexpr(PackedShuffle) {
            const unsigned bits=j<2?partners.x:partners.y;
            partner=qkf(__ushort_as_bfloat16(static_cast<unsigned short>(bits>>((j%2)*16))));
        } else partner=__shfl_sync(0xffffffff,normalized[j],partner_lane&31);
        if(lane<12)partner=-partner;
        H3QKB value=rounded[j];
        if(col+j<96){
            const int64_t position=int64_t(token)*96+col+j;
            if(lane<12)value=rounded[j]*cosine[position]-qkb(-partner)*sine[position];
            else value=rounded[j]*cosine[position]+qkb(partner)*sine[position];
        }
        result[j]=value;
    }
    *reinterpret_cast<uint2*>(const_cast<H3QKB*>(q)+input_offset)=pack4(result);
    }

}

int launch_u32_inplace(void* qkv,const void* qw,const void* kw,const void* cs,const void* sn,
    int64_t tokens,float qe,float ke,cudaStream_t stream) {
    // Public H3 bounds guarantee fewer than 2^32 worker rows. Byte offsets
    // remain int64. Cache occupancy per host thread/device, not across devices.
    static thread_local int cached_device=-1,cached_grid=0;
    int device=0;cudaError_t status=cudaGetDevice(&device);if(status)return status;
    if(device!=cached_device || !cached_grid) {
        int sm=0,occupancy=0;
        status=cudaDeviceGetAttribute(&sm,cudaDevAttrMultiProcessorCount,device);if(status)return status;
        status=cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy,h3_qkv_u32_inplace<false,false>,256,0);if(status)return status;
        cached_device=device;cached_grid=sm*occupancy;
    }
    unsigned blocks=unsigned((2*tokens*56+7)/8);if(blocks>unsigned(cached_grid))blocks=cached_grid;
    h3_qkv_u32_inplace<false,false><<<blocks,256,0,stream>>>((const H3QKB*)qkv,(const H3QKB*)qkv,
        (const H3QKB*)qw,(const H3QKB*)kw,(const H3QKB*)cs,(const H3QKB*)sn,
        nullptr,nullptr,nullptr,nullptr,nullptr,tokens*56,qe,ke);
    return cudaGetLastError();
}
}

extern "C" int h3_sglang_qkv_rope_inplace(void* qkv,const void* qw,const void* kw,
    const void* cs,const void* sn,int64_t tokens,int heads,float qe,float ke,cudaStream_t stream) {
    if(!qkv||!qw||!kw||!cs||!sn||tokens<1||tokens>1048576||heads!=56||!(qe>0)||!(ke>0)||uintptr_t(qkv)%8)
        return cudaErrorInvalidValue;
    return launch_u32_inplace(qkv,qw,kw,cs,sn,tokens,qe,ke,stream);
}
