// Instantiate the exact original FP32 Cutlass specialization. Headers are build
// inputs from the pinned Torch wheel; the shared object links no Torch/ATen code.
#include <ATen/native/transformers/cuda/mem_eff_attention/kernel_forward.h>
#include <cuda_runtime.h>
using AK=PyTorchMemEffAttention::AttentionKernel<float,cutlass::arch::Sm80,true,32,128,65536,true,true>;
extern "C" int h3_audio_encoder_attention(const void* q,const void* k,const void* v,void* out,int batch,int heads,int tokens,int dim,void* stream){
    if(batch!=2||heads!=8||dim!=256||tokens<1)return -1;
    AK::Params p;
    p.query_ptr=(const float*)q;p.key_ptr=(const float*)k;p.value_ptr=(const float*)v;
    p.output_ptr=(float*)out;p.output_accum_ptr=(float*)out;
    p.head_dim=dim;p.head_dim_value=dim;p.num_queries=tokens;p.num_keys=tokens;p.num_keys_absolute=tokens;
    p.num_batches=batch;p.num_heads=heads;p.scale=1.f/sqrtf(float(dim));p.custom_mask_type=AK::CausalFromTopLeft;
    // Native intermediate tensors use contiguous B,H,T,D. Output kernel's
    // contract is B,T,H,D and is transposed by the Rust caller after execution.
    p.q_strideM=p.k_strideM=p.v_strideM=dim;
    p.q_strideH=p.k_strideH=p.v_strideH=tokens*dim;
    p.q_strideB=p.k_strideB=p.v_strideB=int64_t(heads)*tokens*dim;
    p.o_strideM=heads*dim;
    auto kernel=PyTorchMemEffAttention::attention_kernel_batched_impl<AK>;
    constexpr int shared=sizeof(AK::SharedStorage);
    if(shared>0xc000){auto err=cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,shared);if(err)return err;}
    kernel<<<p.getBlocksGrid(),p.getThreadsGrid(),shared,(cudaStream_t)stream>>>(p);
    return cudaGetLastError();
}
