//! SGLang's JIT CUDA kernels (python/sglang/kernels/jit/csrc) and flashinfer's CuTe-DSL norm,
//! launched from the reference CUBINs with the parameter structs their host code builds.
use crate::cuda::{LaunchAttribute, Stream};
use crate::runtime::{attr_pdl, p_f32, p_i32, p_i64, p_u64, Kernels, Launch, Pod};
use crate::Result;

/// CU_LAUNCH_ATTRIBUTE_COOPERATIVE = 0, as cute-dsl's launcher passes it.
fn attr_not_cooperative() -> LaunchAttribute { LaunchAttribute { id: 2, pad: [0; 4], value: [0; 64] } }

const QUANT_FLAT: &str = "_ZN6sglang33per_token_group_quant_flat_kernelINS_10QuantTraitI13__nv_bfloat1613__nv_fp8_e4m3Lj128ELb1ELb0ELb1ELb0ELb0EEELb1ELb0EEEvNSt11conditionalIXT1_ENS_23SortedQuantKernelParamsENS_17QuantKernelParamsEE4typeE";

/// Bytes of the packed UE8M0 activation scales DeepGEMM reads for an [m, k] FP8 activation:
/// int32 [ceil(k/128/4)] x align4(m), MN-major.
pub fn act_scale_bytes(m: usize, k: usize) -> usize { k.div_ceil(512) * m.div_ceil(4) * 4 * 4 }

/// `per_token_group_quant_fp8` (128-wide groups, UE8M0, column-major packed scales):
/// x [m, k] bf16 (row stride `x_stride` elements) -> q [m, k] fp8 (contiguous) + packed scales.
pub fn quant_fp8(k: &Kernels, s: Stream, x: u64, x_stride: usize, q: u64, scales: u64, m: usize, hidden: usize) -> Result<()> {
    let groups = hidden / 128;
    let aligned_m = m.div_ceil(4) * 4;
    let mut p = Pod::default();
    p.u64(x).i64(0).i64(x_stride as i64);               // input TensorArgs
    p.u64(q).i64(0).i64(hidden as i64);                 // output TensorArgs
    p.u64(scales).u32(0).u32(4).u32((aligned_m * 4) as u32).u32(groups as u32); // ScaleStoreArgs (col-major ue8m0)
    p.u32(m as u32).u32(hidden as u32);
    let threads = m * groups * 8;
    // The trait's third flag is off when the group count is not a multiple of 4 (a partly filled
    // packed scale word): DeepSeek-V4-Pro's rank-local shared expert (384 wide, 3 groups).
    let name = if groups % 4 == 0 { QUANT_FLAT.to_string() } else { QUANT_FLAT.replace("Lj128ELb1ELb0ELb1ELb0ELb0E", "Lj128ELb1ELb0ELb0ELb0ELb0E") };
    Launch { name: &name, grid: [threads.div_ceil(256) as u32, 1, 1], block: [256, 1, 1], smem: 0,
             attrs: vec![attr_pdl()], params: vec![p.take()] }.run(k, s)
}

/// flashinfer `rmsnorm_cute` for H = 1024 bf16 rows (q_norm). A [m, 1024] input whose row stride is
/// not 1024 (and m > 1) takes the strided variant; otherwise the compact one, as torch's
/// `is_contiguous` decides.
#[allow(clippy::too_many_arguments)]
pub fn rmsnorm_1024(k: &Kernels, s: Stream, x: u64, x_stride: usize, w: u64, out: u64, m: usize, eps: f32) -> Result<()> {
    // H = Q_LORA (1024; DeepSeek-V4-Pro 1536).
    let h = crate::model::Q_LORA;
    let contiguous = m == 1 || x_stride == h;
    let (name, x_arg, out_arg) = if contiguous {
        (format!("kernel_cutlass_kernel_flashinfernormkernelsrmsnormRMSNormKernel_object_at__tensorptrbf16gmemalign128oi64{h}{h}1_tensorptrbf16gmemalign16o{h}1_tensorptrbf16gmemalign128oi64{h}{h}1___T_0"),
         Pod::default().u64(x).i64(m as i64).take(), Pod::default().u64(out).i64(m as i64).take())
    } else {
        (format!("kernel_cutlass_kernel_flashinfernormkernelsrmsnormRMSNormKernel_object_at__tensorptrbf16gmemalign16oi64{h}i64div81_tensorptrbf16gmemalign16o{h}1_tensorptrbf16gmemalign16oi64{h}i64div81_0"),
         Pod::default().u64(x).i64(m as i64).i64(x_stride as i64).take(), Pod::default().u64(out).i64(m as i64).i64(h as i64).take())
    };
    // 32 threads per row, 128 threads -> 4 rows per block; tile 4 x H bf16 + reductions.
    Launch { name: &name, grid: [m.div_ceil(4) as u32, 1, 1], block: [128, 1, 1], smem: (4 * h * 2 + 16) as u32,
             attrs: vec![attr_pdl(), attr_not_cooperative()],
             params: vec![x_arg, p_u64(w), out_arg, p_i64(m as i64), p_f32(eps)] }.run(k, s)
}

/// flashinfer `rmsnorm_cute` for H = 4096 bf16 rows, contiguous (the final norm).
pub fn rmsnorm_4096(k: &Kernels, s: Stream, x: u64, w: u64, out: u64, m: usize, eps: f32) -> Result<()> {
    // H = HIDDEN. 4096: 64 threads per row -> 2 rows per block; 7168 (DeepSeek-V4-Pro): one row
    // per block. The tile is rows x H bf16 + reductions.
    let h = crate::model::HIDDEN;
    let rows = if h > 4096 { 1 } else { 2 };
    let name = format!("kernel_cutlass_kernel_flashinfernormkernelsrmsnormRMSNormKernel_object_at__tensorptrbf16gmemalign128oi64{h}{h}1_tensorptrbf16gmemalign16o{h}1_tensorptrbf16gmemalign128oi64{h}{h}1___T_0");
    Launch { name: &name,
             grid: [m.div_ceil(rows) as u32, 1, 1], block: [128, 1, 1], smem: (rows * h * 2 + 16) as u32, attrs: vec![attr_pdl(), attr_not_cooperative()],
             params: vec![Pod::default().u64(x).i64(m as i64).take(), p_u64(w), Pod::default().u64(out).i64(m as i64).take(),
                          p_i64(m as i64), p_f32(eps)] }.run(k, s)
}

/// `fused_q_norm_rope` (head 512, rope 64, int64 positions, PDL, norm): per-head RMSNorm without
/// weight, then RoPE on the last 64 dims, q_in [b, h, 512] -> q_out.
#[allow(clippy::too_many_arguments)]
pub fn fused_q_norm_rope(k: &Kernels, s: Stream, q_in: u64, q_out: u64, freqs: u64, positions: u64, in_stride: usize,
                         out_stride: usize, batch: usize, heads: usize, eps: f32) -> Result<()> {
    let mut p = Pod::default();
    p.u64(q_in).u64(q_out).u64(freqs).u64(positions).i64(in_stride as i64).i64(out_stride as i64)
     .u32(batch as u32).u32(heads as u32).f32(eps).pad_to(64);
    Launch { name: "_ZN6sglang17fused_q_norm_ropeI13__nv_bfloat16Ll512ELl64ElLb1ELb1EEEvNS_20FusedQNormRopeParamsE",
             grid: [(batch * heads).div_ceil(8) as u32, 1, 1], block: [256, 1, 1], smem: 0, attrs: vec![attr_pdl()],
             params: vec![p.take()] }.run(k, s)
}

/// `fused_k_norm_rope_flashmla` (head 512, rope 64, int64 positions, page size 8, V4 layout):
/// RMSNorm with weight + RoPE + FP8 quantization of each token's KV, stored into its FlashMLA page.
#[allow(clippy::too_many_arguments)]
pub fn fused_k_norm_rope_flashmla(k: &Kernels, s: Stream, kv: u64, kv_weight: u64, freqs: u64, positions: u64,
                                  out_loc: u64, kvcache: u64, kv_stride: usize, batch: usize, eps: f32) -> Result<()> {
    let mut p = Pod::default();
    p.u64(kv).u64(kv_weight).u64(freqs).u64(positions).u64(out_loc).u64(kvcache).i64(kv_stride as i64)
     .u32(batch as u32).f32(eps).u64(0).i64(0).i64(0).u32(0).pad_to(96);
    Launch { name: "_ZN6sglang26fused_k_norm_rope_flashmlaI13__nv_bfloat16Ll512ELl64ElLi8ELNS_11deepseek_v48KVLayoutE0ELb1ELb0EEEvNS_28FusedKNormRopeFlashMLAParamsE",
             grid: [batch as u32, 1, 1], block: [256, 1, 1], smem: 0, attrs: vec![attr_pdl()], params: vec![p.take()] }.run(k, s)
}

/// `deepseek_rope_kernel` (PDL, inverse, int64 positions) on q only: un-rotates the attention
/// output's RoPE dims in place. Strides in bytes.
#[allow(clippy::too_many_arguments)]
pub fn inverse_rope_q(k: &Kernels, s: Stream, q: u64, freqs: u64, positions: u64, q_stride_batch: usize,
                      q_stride_head: usize, heads: usize, batch: usize) -> Result<()> {
    let mut p = Pod::default();
    p.u64(q).u64(0).u64(freqs).u64(positions).i64(q_stride_batch as i64).i64(0).i64(q_stride_head as i64).i64(0)
     .u32(heads as u32).u32(0).u32(batch as u32).pad_to(80);
    Launch { name: "_ZN6sglang20deepseek_rope_kernelILb1ELb1ElEEvNS_17FusedQKRopeParamsE",
             grid: [(batch * heads).div_ceil(4) as u32, 1, 1], block: [128, 1, 1], smem: 0, attrs: vec![attr_pdl()],
             params: vec![p.take()] }.run(k, s)
}

/// `fp8_wo_a_group_major_quant_ue8m0`: x [t, g, d] bf16 (token stride `x_stride` elements) ->
/// fp8 [t, g, d] + packed scales [g, ceil(d/512), align4(t)] int32.
#[allow(clippy::too_many_arguments)]
pub fn wo_a_group_quant(k: &Kernels, s: Stream, x: u64, q: u64, scales: u64, t: usize, groups: usize, d: usize, x_stride: usize) -> Result<()> {
    let hidden_groups = d / 128;
    let total = t * groups * hidden_groups;
    Launch { name: "_ZN6sglang39fp8_wo_a_group_major_quant_ue8m0_kernelI13__nv_bfloat16Lb1EEEvPKT_P13__nv_fp8_e4m3Piliiill",
             grid: [total.div_ceil(16) as u32, 1, 1], block: [128, 1, 1], smem: 0, attrs: vec![attr_pdl()],
             params: vec![p_u64(x), p_u64(q), p_u64(scales), p_i64(total as i64), p_i32(hidden_groups as i32),
                          p_i32(hidden_groups.div_ceil(4) as i32), p_i32(groups as i32), p_i64(t.div_ceil(4) as i64 * 4),
                          p_i64(x_stride as i64)] }.run(k, s)
}

/// `dequantize_k_cache_paged_dsv4` (V4 layout): the listed cache slots -> bf16 rows [n, 512] (row
/// stride `out_stride` elements). Triton (out, fp8 view, bf16 view, u8 view, slots, out_stride, n),
/// 4 tokens per program. The page size is a constexpr: `kind` (the cache: swa / c4 / c128) selects
/// the SWA (256), C4 (64) or C128 (2) page CUBIN.
pub fn dequant_k_cache(k: &Kernels, s: Stream, out: u64, cache: u64, slots: u64, out_stride: usize, n: usize, kind: &str) -> Result<()> {
    use crate::runtime::{triton, TArg};
    if n == 0 { return Ok(()); }
    triton(k, s, "_dequantize_k_cache_paged_kernel", [n.div_ceil(4) as u32, 1, 1], 4, 4096,
           &[TArg::Ptr(out), TArg::Ptr(cache), TArg::Ptr(cache), TArg::Ptr(cache), TArg::Ptr(slots), TArg::I32(out_stride as i32),
             TArg::I32(n as i32)], &format!("cache={kind}"))
}

/// `tiny_n_gemm` (the MoE router): out [m, 256] f32 = x [m, 4096] bf16 (row stride `x_stride`) times
/// W [256, 4096]^T bf16. One template instantiation per token count m.
pub fn router_tiny_gemm(k: &Kernels, s: Stream, out: u64, x: u64, w: u64, m: usize, x_stride: usize) -> Result<()> {
    // GEMMTraitN<experts, hidden, 2 (Pro: 3), 32>.
    let (n, h) = (crate::model::EXPERTS, crate::model::HIDDEN);
    let third = if h > 4096 { 3 } else { 2 };
    let name = format!("_ZN6sglang18tiny_n_gemm_kernelINS_10GEMMTraitNILj{n}ELj{h}ELj{third}ELj32E13__nv_bfloat16EELj{m}EfLb1ES2_EEvNS_14TinyGEMMParamsE");
    let mut p = Pod::default();
    p.u64(out).u64(x).u64(w).i64(x_stride as i64);
    // GEMMTraitN<256, 4096, 2, 32>: 128 blocks of (4096 * 2 / 32) threads.
    Launch { name: &name, grid: [128, 1, 1], block: [(h * 2 / 32) as u32, 1, 1], smem: 0, attrs: vec![attr_pdl()], params: vec![p.take()] }.run(k, s)
}

/// `moe_hash_topk_fused<act_sqrt_softplus>` (hash-routed layers): experts from tid2eid[input_id],
/// weights from the router logits. ids int32 / weights float32 [t, topk].
#[allow(clippy::too_many_arguments)]
pub fn hash_topk(k: &Kernels, s: Stream, logits: u64, input_ids: u64, tid2eid: u64, topk_ids: u64, topk_weights: u64, t: usize,
                 topk: usize, experts: usize, routed_scaling_factor: f32) -> Result<()> {
    let mut p = Pod::default();
    p.u64(logits).u64(input_ids).u64(tid2eid).u64(topk_ids).u64(topk_weights).u32(t as u32).u32(topk as u32)
     .u32(experts as u32).u32(0).f32(routed_scaling_factor).pad_to(64);
    Launch { name: "_ZN6sglang19moe_hash_topk_fusedIXadL_ZNS_17act_sqrt_softplusEfEELb1EEEvNS_17MoEHashTopKParamsE",
             grid: [t.div_ceil(4) as u32, 1, 1], block: [128, 1, 1], smem: 0, attrs: vec![attr_pdl()], params: vec![p.take()] }.run(k, s)
}

/// `silu_mul_clamp` (the shared expert's activation): out [t, d] = silu(min(gate, limit)) *
/// clamp(up, -limit, limit) over input [t, 2d] = (gate | up), bf16.
pub fn silu_mul_clamp(k: &Kernels, s: Stream, input: u64, output: u64, t: usize, out_dim: usize, limit: f32) -> Result<()> {
    let out_vecs = out_dim / 8;
    let threads = out_vecs.min(1024);
    let blocks_per_row = out_vecs.div_ceil(threads);
    let mut p = Pod::default();
    p.u64(input).u64(output).f32(limit).u32(out_vecs as u32).u32(blocks_per_row as u32);
    Launch { name: "_ZN6sglang21silu_mul_clamp_kernelI13__nv_bfloat16Lb1EEEvNS_21SiluAndMulClampParamsE",
             grid: [(t * blocks_per_row) as u32, 1, 1], block: [threads as u32, 1, 1], smem: 0, attrs: vec![attr_pdl()],
             params: vec![p.take()] }.run(k, s)
}

/// flashinfer `nvfp4_quantize` (CuTe-DSL, linear scale layout, global scale tensor) for K = 4096
/// bf16 rows: x [m, 4096] -> packed e2m1 [m, 2048] + e4m3 block scales [m * 256].
pub fn nvfp4_quantize_4096(k: &Kernels, s: Stream, x: u64, out: u64, scales: u64, m: usize, global_scale: u64, sms: usize) -> Result<()> {
    let h = crate::model::HIDDEN;
    let total = m * h / 16;
    let blocks = total.div_ceil(512).min(sms * 4);
    let t = |ptr: u64, n: usize| { let mut p = Pod::default(); p.u64(ptr).i32(n as i32).pad_to(16); p.take() };
    let name = format!("kernel_cutlass_kernel_flashinferquantizationkernelsnvfp4_quantizeNVFP4QuantizeLinearKernel_object_at__tensorptrbf16gmemalign16o{h}{h}1_tensorptri8gmemalign16o{0}{0}1_tensorptri8gmemal_0", h / 2);
    Launch { name: &name,
             grid: [blocks as u32, 1, 1], block: [512, 1, 1], smem: 0, attrs: vec![attr_pdl(), attr_not_cooperative()],
             params: vec![t(x, m), t(out, m), t(scales, total), p_i32(m as i32), p_i32(total as i32), p_u64(global_scale)] }.run(k, s)
}

/// flashinfer's MXFP8 quantization (cute-dsl MXFP8QuantizeLinearKernel, hidden 4096): x bf16 [m, 4096]
/// -> e4m3 [m, 4096] with E8M0 scales [m, 128] in the linear layout (one per 32 elements), as the
/// MTP draft's MXFP4 experts take their input (`flashinfer_mxfp8_quantize(x, False, alignment=4096)`).
pub fn mxfp8_quantize_4096(k: &Kernels, s: Stream, x: u64, out: u64, scales: u64, m: usize, sms: usize) -> Result<()> {
    let total = m * 4096 / 32;
    let t = |ptr: u64, n: usize| { let mut p = Pod::default(); p.u64(ptr).i32(n as i32).pad_to(16); p.take() };
    Launch { name: "kernel_cutlass_kernel_flashinferquantizationkernelsmxfp8_quantizeMXFP8QuantizeLinearKernel_object_at__tensorptrbf16gmemalign16o409640961_tensorptri8gmemalign16o409640961_tensorptri8gmemal_0",
             grid: [m.min(sms * 4) as u32, 1, 1], block: [512, 1, 1], smem: 0, attrs: vec![attr_pdl(), attr_not_cooperative()],
             params: vec![t(x, m), t(out, m), t(scales, total), p_i32(total as i32)] }.run(k, s)
}

/// `_router_triton_kernel` (score-routed MoE layers): sqrt-softplus scores + e_score_correction
/// bias, top-6 of 256, renormalized and scaled by the routed scaling factor; weights float32 /
/// ids int32 [m, 6]. `packed` is the unused int32 output slot SGLang passes (a dummy buffer).
#[allow(clippy::too_many_arguments)]
pub fn router_triton(k: &Kernels, s: Stream, logits: u64, bias: u64, weights: u64, ids: u64, packed: u64, m: usize, scaling: f32) -> Result<()> {
    use crate::runtime::{triton, TArg};
    // N 256, K 6, BLOCK_M 1, one warp per row; None pointers (bias_alt, input_ids, padding) and
    // unit strides are compile-time constants.
    // DeepSeek-V4-Pro (384 experts): SGLang launches this kernel without programmatic dependent
    // launch up to 8 rows on sm_103 and with it above (two images); Flash's bundles hold one.
    if crate::model::EXPERTS == 384 {
        let pdl = m > 8;
        return crate::runtime::triton_with(k, s, "_router_triton_kernel", [m as u32, 1, 1], 1, 32,
           &[TArg::Ptr(logits), TArg::Ptr(bias), TArg::Ptr(weights), TArg::Ptr(ids), TArg::Ptr(packed), TArg::I32(m as i32),
             TArg::F32(scaling), TArg::F32(0.0), TArg::I32(1), TArg::I32(0), TArg::I32(0), TArg::I32(384), TArg::I32(1),
             TArg::I32(6), TArg::I32(1), TArg::I32(6), TArg::I32(1), TArg::I32(0), TArg::I32(0)], &format!("pdl={}", pdl as i32),
           &if pdl { vec![attr_pdl()] } else { vec![] });
    }
    triton(k, s, "_router_triton_kernel", [m as u32, 1, 1], 1, 32,
           &[TArg::Ptr(logits), TArg::Ptr(bias), TArg::Ptr(weights), TArg::Ptr(ids), TArg::Ptr(packed), TArg::I32(m as i32),
             TArg::F32(scaling), TArg::F32(0.0), TArg::I32(1), TArg::I32(0), TArg::I32(0), TArg::I32(crate::model::EXPERTS as i32), TArg::I32(1),
             TArg::I32(6), TArg::I32(1), TArg::I32(6), TArg::I32(1), TArg::I32(0), TArg::I32(0)], "")
}

/// SGLang's fused input-logprob pass (srt/layers/logsumexp.py `row_logsumexp_topk`, k = 5): per
/// fp32 row of `cols`, (max, logsumexp - max) and the top-5 raw values / int64 indices. One
/// program per row, BLOCK_N 16384, K_PAD 8, 8 warps.
pub fn row_logsumexp_topk(k: &Kernels, s: Stream, x: u64, out_max: u64, out_log_sum: u64, top_vals: u64, top_idx: u64,
                          rows: usize, cols: usize) -> Result<()> {
    use crate::runtime::TArg;
    crate::runtime::triton(k, s, "_row_logsumexp_topk_kernel", [rows as u32, 1, 1], 8, 128,
        &[TArg::Ptr(x), TArg::Ptr(out_max), TArg::Ptr(out_log_sum), TArg::Ptr(top_vals), TArg::Ptr(top_idx),
          TArg::I32(cols as i32), TArg::I32(1), TArg::I32(cols as i32)], "")
}

/// `fused_q_indexer_rope_hadamard_quant<bf16, int32 positions, PDL>` (main_norm_rope.cuh): the C4
/// indexer's query [t, heads, 128] bf16 -> RoPE on the last 64 dims, Hadamard, FP8 per (token, head)
/// into `q_fp8`; `weights_out[t, h]` = weight[t, h] * weight_scale * q_scale[t, h] (f32).
#[allow(clippy::too_many_arguments)]
pub fn fused_q_indexer_rope_hadamard_quant(k: &Kernels, s: Stream, q_in: u64, q_fp8: u64, weight: u64, weights_out: u64, weight_scale: f32,
                                           rope_cache: u64, positions_i32: u64, weight_stride: usize, t: usize, heads: usize) -> Result<()> {
    if t == 0 { return Ok(()); }
    let mut p = Pod::default();
    p.u64(q_in).u64(q_fp8).u64(weight).u64(weights_out).f32(weight_scale).u64(rope_cache).u64(positions_i32)
     .i64(weight_stride as i64).u32(t as u32).u32(heads as u32);
    Launch { name: "_ZN6sglang35fused_q_indexer_rope_hadamard_quantI13__nv_bfloat16iLb1ELb0ELb1EEEvNS_36FusedQIndexerRopeHadamardQuantParamsE",
             grid: [(t * heads).div_ceil(8) as u32, 1, 1], block: [256, 1, 1], smem: 0, attrs: vec![attr_pdl()], params: vec![p.take()] }.run(k, s)
}

/// `_get_k_and_s_triton` (dsa_index_k_cache.py) for one request: gathers its first `*seq_len`
/// indexer keys (FP8 [n, 128]) and scales (f32 [n]) from the paged index-K cache (pages of 64
/// entries: 64 x 128 key bytes, then 64 x 4 scale bytes). `page_table` is the request's row of
/// `width` page ids; `max_seq_len` sizes the grid.
#[allow(clippy::too_many_arguments)]
pub fn get_k_and_s(k: &Kernels, s: Stream, cache: u64, page_table: u64, k_out: u64, s_out: u64, seq_len: u64, max_seq_len: usize, width: usize) -> Result<()> {
    use crate::runtime::{triton, TArg};
    triton(k, s, "_get_k_and_s_triton_kernel", [1, max_seq_len.div_ceil(256) as u32, 1], 4, 1024,
           &[TArg::Ptr(cache), TArg::Ptr(page_table), TArg::Ptr(k_out), TArg::Ptr(s_out), TArg::Ptr(seq_len), TArg::I32(width as i32)], "")
}

/// `_combine_topk_swa_indices_kernel` (sparse_prefill_kernels.py, one request, implicit SWA): per
/// query, [its top-k raw indices + compressed_base (-1 kept) | its window in the workspace | -1] and
/// the scanned length. `meta` holds int32 query_start_loc[2], seq_lens[1], gather_lens[1],
/// compressed_base[1], swa_base[1], each in its own 16-byte slot (separate tensors in SGLang, so
/// Triton's alignment specialization sees aligned pointers). `kind` (c4 / c128 / meta for the
/// window-only layers) and `width` (a constexpr) select the image.
#[allow(clippy::too_many_arguments)]
pub fn combine_topk_swa(k: &Kernels, s: Stream, out: u64, width: usize, out_lens: u64, topk: u64, topk_stride: usize, meta: u64, meta_slot: usize,
                        nreqs: usize, query_pos_i32: u64, t: usize, top_k: usize, kind: &str) -> Result<()> {
    use crate::runtime::{triton, TArg};
    // query_start_loc, seq_lens, gather_lens, compressed_base, swa_base: `meta_slot` bytes apart.
    let w = |i: u64| TArg::Ptr(meta + meta_slot as u64 * i);
    // SEARCH_STEPS (the request search's bisection steps) is a constexpr: one image per bit length.
    let steps = (usize::BITS - nreqs.max(1).leading_zeros()) as usize;
    triton(k, s, "_combine_topk_swa_indices_kernel", [t.div_ceil(2) as u32, 1, 1], 2, 0,
           &[TArg::Ptr(out), TArg::I32(width as i32), TArg::Ptr(out_lens), TArg::Ptr(topk), TArg::I32(topk_stride as i32), w(0),
             TArg::Ptr(query_pos_i32), w(1), w(2), w(3), w(4), TArg::I32(0), TArg::I32NoSpec(t as i32), TArg::I32NoSpec(nreqs as i32),
             TArg::I32NoSpec(top_k as i32)], &format!("kind={kind}&width={width}&steps={steps}"))
}
