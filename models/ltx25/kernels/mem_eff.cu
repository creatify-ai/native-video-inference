// Gemma efficient-attention specialization selected by pinned PyTorch 2.13 on B200:
// fmha_cutlassF_bf16_aligned_32x128_gmem_sm80, i.e. AttentionKernel<bf16, Sm80, aligned,
// 32, 128, 65536, dropout, bias>. Instantiated from the pinned wheel's headers; the
// parameters follow attention.cu _efficient_attention_forward for BMHK views.
#include <ATen/native/transformers/cuda/mem_eff_attention/kernel_forward.h>
#include <cuda_runtime.h>
#include <cmath>
using AK = PyTorchMemEffAttention::AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 65536, true, true>;

// q/k/v: BMHK strides (in elements); out: contiguous [B, M, H, Kv] bf16; accum: fp32 of the
// same shape when the kernel needs it (bf16 output).
extern "C" int ltx25_mem_eff_attention(const void* q, const void* k, const void* v, void* out, void* accum, int64_t batch,
                                     int64_t queries, int64_t keys, int64_t heads, int64_t head_dim, int64_t q_sb,
                                     int64_t q_sm, int64_t q_sh, int64_t k_sb, int64_t k_sm, int64_t k_sh, int64_t v_sb,
                                     int64_t v_sm, int64_t v_sh, const void* bias, int64_t bias_sb, int64_t bias_sh, int64_t bias_sm, float scale, void* stream) {
    AK::Params p;
    p.query_ptr = (const AK::scalar_t*)q; p.key_ptr = (const AK::scalar_t*)k; p.value_ptr = (const AK::scalar_t*)v;
    p.logsumexp_ptr = nullptr;
    p.output_accum_ptr = AK::kNeedsOutputAccumulatorBuffer ? (AK::output_accum_t*)accum : nullptr;
    p.output_ptr = (AK::output_t*)out;
    p.num_heads = heads; p.head_dim = head_dim; p.head_dim_value = head_dim;
    p.num_queries = queries; p.num_keys = keys; p.num_batches = batch;
    p.custom_mask_type = AK::NoCustomMask;
    p.seqlen_k_ptr = nullptr;
    p.scale = scale;
    p.attn_bias_ptr = (const AK::scalar_t*)bias;
    p.bias_strideB = bias_sb; p.bias_strideH = bias_sh; p.bias_strideM = bias_sm;
    p.q_strideB = q_sb; p.k_strideB = k_sb; p.v_strideB = v_sb;
    p.q_strideM = q_sm; p.k_strideM = k_sm; p.v_strideM = v_sm;
    p.q_strideH = q_sh; p.k_strideH = k_sh; p.v_strideH = v_sh;
    p.o_strideM = heads * head_dim;
    p.use_dropout = false;
    auto kernel = PyTorchMemEffAttention::attention_kernel_batched_impl<AK>;
    constexpr int smem = sizeof(AK::SharedStorage);
    if (smem > 0xc000) {
        auto err = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        if (err) return err;
    }
    // AK::check_supported uses TORCH_CHECK (libc10); the caller guarantees aligned BMHK views.
    if (head_dim % AK::kAlignmentQ || q_sm % AK::kAlignmentQ || k_sm % AK::kAlignmentK || v_sm % AK::kAlignmentV) return -1;
    kernel<<<p.getBlocksGrid(), p.getThreadsGrid(), smem, (cudaStream_t)stream>>>(p);
    return cudaGetLastError();
}
extern "C" int ltx25_mem_eff_needs_accum() { return AK::kNeedsOutputAccumulatorBuffer ? 1 : 0; }
