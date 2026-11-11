//! The routed experts: flashinfer's trtllm-gen FP4 MoE (routing, two batched GEMMs, finalize),
//! driven by its own host code compiled into this binary (kernels/moe_host.cu).
use crate::cuda::Stream;
use crate::Result;
use std::ffi::{c_char, c_void, CString};

#[repr(C)]
struct Args {
    num_tokens: i32, num_experts: i32, top_k: i32, hidden_size: i32, intermediate_size: i32, local_expert_offset: i32,
    local_num_experts: i32, tile_tokens_dim: i32, enable_pdl: i32,
    gemm1_kernel: *const c_char, gemm2_kernel: *const c_char,
    topk_ids: u64, topk_weights: u64, hidden_states: u64, hidden_states_scale: u64,
    gemm1_weights: u64, gemm1_weights_scale: u64, gemm2_weights: u64, gemm2_weights_scale: u64,
    gemm1_clamp_limit: u64, output1_scales_scalar: u64, output1_scales_gate_scalar: u64, output2_scales_scalar: u64,
    output: u64, scratch: u64, scratch_bytes: usize, stream: *mut c_void, cubin_dir: *const c_char, scratch_used: *mut usize,
    /// 1: MXFP4 weights x MXFP8 activations (the MTP draft's experts).
    mx: i32,
}

extern "C" {
    fn dsv4_moe_fp4_routed(a: *const Args, err: *mut c_char, cap: usize) -> i32;
}

/// One layer's expert parameters (SGLang's loaded tensors, device addresses).
pub struct Experts { pub w13: u64, pub w13_scale: u64, pub w2: u64, pub w2_scale: u64, pub clamp_limit: u64,
                     pub g1_scale_c: u64, pub g1_alphas: u64, pub g2_alphas: u64 }

/// The MoE tactic (tile tokens, FC1 kernel, FC2 kernel) by token count: flashinfer's autotuner picks
/// these per server start, so the port takes the kernels the reference run whose golds it
/// reproduces (the census run, 20261108-132810) used for each token count of the gold prompts.
pub fn tactic(t: usize) -> (i32, &'static str, &'static str) {
    const FC1_T8: &str = "bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x8x512u2_s5_et128x8_m128x8x64_c1x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_ldgsts_ldgstsSf_rgTma_clmp_swiGlu_dynB_sm100f";
    const FC1_T16: &str = "bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x16x512u2_s5_et128x16_m128x16x64_c1x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_ldgsts_ldgstsSf_rgTma_clmp_swiGlu_dynB_sm100f";
    const FC2_T8_DECODE: &str = "bmm_Bfloat16_E2m1E2m1_Fp32_Ab16_Bb16_t128x8x512_s5_et128x8_m128x8x64_c1x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_rgTma_clmp_dynB_sm100f";
    const FC2_T8: &str = "bmm_Bfloat16_E2m1E2m1_Fp32_Ab16_Bb16tokFp32_t128x8x512u2_s5_et128x8_m128x8x64_c1x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_rgTma_clmp_dynB_sm100f";
    const FC2_T16: &str = "bmm_Bfloat16_E2m1E2m1_Fp32_Ab16_Bb16tokFp32_t128x16x512u2_s5_et128x16_m256x16x64_c2x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_rgTma_clmp_dynB_sm100f";
    match t {
        1 => (8, FC1_T8, FC2_T8_DECODE),
        2..=8 => (8, FC1_T8, FC2_T8),
        _ => (16, FC1_T16, FC2_T16),
    }
}

/// `trtllm_fp4_block_scale_routed_moe` for DeepSeek-V4-Flash (256 experts, top-6, hidden 4096,
/// intermediate 2048): hidden_states e2m1 [t, 2048] + linear e4m3 scales [t, 256], routing ids
/// int32 / weights float32 [t, 6] -> output bf16 [t, 4096].
#[allow(clippy::too_many_arguments)]
pub fn fp4_routed(k: &crate::runtime::Kernels, s: Stream, cubin_dir: &str, e: &Experts, topk_ids: u64, topk_weights: u64, hidden: u64,
                  hidden_scale: u64, output: u64, t: usize, scratch: u64, scratch_bytes: usize) -> Result<usize> {
    routed(k, s, cubin_dir, e, topk_ids, topk_weights, hidden, hidden_scale, output, t, scratch, scratch_bytes, false)
}

/// `trtllm_fp4_block_scale_routed_moe` with MXFP4 experts (Mxfp4FlashinferTrtllmMoEMethod, the MTP
/// draft's layer): `hidden` MXFP8 e4m3 [t, 4096] with linear E8M0 scales [t, 128]; `e.clamp_limit`
/// the per-expert swiglu limit, the output scales ones.
#[allow(clippy::too_many_arguments)]
pub fn mxfp4_routed(k: &crate::runtime::Kernels, s: Stream, cubin_dir: &str, e: &Experts, topk_ids: u64, topk_weights: u64, hidden: u64,
                    hidden_scale: u64, output: u64, t: usize, scratch: u64, scratch_bytes: usize) -> Result<usize> {
    routed(k, s, cubin_dir, e, topk_ids, topk_weights, hidden, hidden_scale, output, t, scratch, scratch_bytes, true)
}

#[allow(clippy::too_many_arguments)]
fn routed(k: &crate::runtime::Kernels, s: Stream, cubin_dir: &str, e: &Experts, topk_ids: u64, topk_weights: u64, hidden: u64,
          hidden_scale: u64, output: u64, t: usize, scratch: u64, scratch_bytes: usize, mx: bool) -> Result<usize> {
    // The bundle's census tactics for this token count, else the first census run's table.
    let (tile, fc1, fc2) = match (if mx { k.moe_tactic_mx(t) } else { k.moe_tactic(t) }, mx) {
        (Some(x), _) => x,
        (None, false) => { let (tile, a, b) = tactic(t); (tile, a.to_string(), b.to_string()) }
        (None, true) => return Err(format!("no MXFP4 MoE tactic for {t} tokens in the bundle")),
    };
    let (g1, g2, dir) = (CString::new(fc1).unwrap(), CString::new(fc2).unwrap(), CString::new(cubin_dir).unwrap());
    let mut used = 0usize;
    let a = Args {
        num_tokens: t as i32, num_experts: crate::model::EXPERTS as i32, top_k: 6, hidden_size: crate::model::HIDDEN as i32,
        intermediate_size: crate::model::INTER as i32, local_expert_offset: 0,
        local_num_experts: crate::model::EXPERTS as i32, tile_tokens_dim: tile, enable_pdl: (t <= 8192) as i32,
        gemm1_kernel: g1.as_ptr(), gemm2_kernel: g2.as_ptr(),
        topk_ids, topk_weights, hidden_states: hidden, hidden_states_scale: hidden_scale,
        gemm1_weights: e.w13, gemm1_weights_scale: e.w13_scale, gemm2_weights: e.w2, gemm2_weights_scale: e.w2_scale,
        gemm1_clamp_limit: e.clamp_limit, output1_scales_scalar: e.g1_scale_c, output1_scales_gate_scalar: e.g1_alphas,
        output2_scales_scalar: e.g2_alphas, output, scratch, scratch_bytes, stream: s.0, cubin_dir: dir.as_ptr(), scratch_used: &mut used,
        mx: mx as i32,
    };
    let mut err = vec![0u8; 8192];
    let rc = unsafe { dsv4_moe_fp4_routed(&a, err.as_mut_ptr().cast(), err.len()) };
    if rc != 0 {
        let n = err.iter().position(|&b| b == 0).unwrap_or(err.len());
        return Err(format!("trtllm MoE: {}", String::from_utf8_lossy(&err[..n])));
    }
    Ok(used)
}
