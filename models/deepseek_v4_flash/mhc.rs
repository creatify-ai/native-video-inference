//! Hyper-connection (mHC) kernels: SGLang's TileLang JIT kernels (kernels/ops/layernorm/mhc.py)
//! and its Triton broadcast/head kernels, launched from the reference's CUBINs.
//!
//! TileLang orders a kernel's buffer parameters by name, then its dynamic shape parameters;
//! scalar hyper-parameters (eps, sinkhorn iterations, ...) are compiled into the CUBIN.
use crate::cuda::Stream;
use crate::runtime::{attr_pdl, p_i32, p_u64, Kernels, Launch};
use crate::Result;

/// `_hc_broadcast_kernel`: x [T, H] bf16 -> y [T, HC, H] (every copy equal to x).
pub fn hc_broadcast(k: &Kernels, s: Stream, x: u64, y: u64, t: usize, hidden: usize) -> Result<()> {
    Launch {
        name: "_hc_broadcast_kernel", grid: [t as u32, hidden.div_ceil(1024) as u32, 1], block: [128, 1, 1], smem: 0,
        attrs: vec![],
        // x_ptr, y_ptr, H, x_stride_m, then Triton's global and profile scratch pointers.
        params: vec![p_u64(x), p_u64(y), p_i32(hidden as i32), p_i32(hidden as i32), p_u64(0), p_u64(0)],
    }.run(k, s)
}

pub struct MhcPreOut { pub post_mix: u64, pub comb_mix: u64, pub layer_input: u64 }

/// `mhc_pre_big_fuse_with_norm_tilelang` over `n_splits` GEMM partials (a compile-time constant of
/// the TileLang kernel: one CUBIN per split count).
#[allow(clippy::too_many_arguments)]
pub fn pre_big_fuse_with_norm(k: &Kernels, s: Stream, gemm_out_mul: u64, gemm_out_sqrsum: u64, hc_scale: u64,
                              hc_base: u64, residual: u64, out: &MhcPreOut, norm_weight: u64, t: usize, n_splits: usize) -> Result<()> {
    Launch {
        name: &format!("mhc_pre_big_fuse_with_norm_tilelang_kernel#n_splits={n_splits}"), grid: [t as u32, 1, 1], block: [96, 1, 1],
        // 37232 at hidden 4096, 43376 at 7168: a hidden-wide bf16 row plus fixed tiles.
        smem: (29040 + 2 * crate::model::HIDDEN) as u32,
        attrs: vec![attr_pdl()],
        // Sorted by name: comb_mix, gemm_out_mul, gemm_out_sqrsum, hc_base, hc_scale, layer_input,
        // norm_weight, post_mix, residual; then num_tokens.
        params: vec![p_u64(out.comb_mix), p_u64(gemm_out_mul), p_u64(gemm_out_sqrsum), p_u64(hc_base), p_u64(hc_scale),
                     p_u64(out.layer_input), p_u64(norm_weight), p_u64(out.post_mix), p_u64(residual), p_i32(t as i32)],
    }.run(k, s)
}

/// `mhc_fused_post_pre_fma_tilelang` (small batches, T <= 32): the previous sublayer's mHC post
/// (residual_cur = post * hidden_in + comb^T residual) fused with the next pre's GEMM partials and
/// RMS square sums. Mix-output tile 2 for T < 8 (else 3) is compiled in; split_k is a runtime
/// argument (8 for T < 8, else 4).
#[allow(clippy::too_many_arguments)]
pub fn fused_post_pre_fma(k: &Kernels, s: Stream, prev_comb_mix: u64, prev_residual: u64, prev_post_mix: u64, hidden_in: u64,
                          pre_fn: u64, mixes_partial_out: u64, sqrsum_partial_out: u64, cur_residual_out: u64, t: usize) -> Result<()> {
    let (tile, split_k) = fused_post_pre_shape(t);
    Launch {
        name: &format!("mhc_fused_post_pre_fma_tilelang_kernel#tile={tile}"), grid: [t as u32, 24u32.div_ceil(tile as u32), split_k as u32],
        block: [256, 1, 1], smem: if cfg!(feature = "pro") && tile == 3 { 128 } else { 96 }, attrs: vec![attr_pdl()],
        // Sorted by name: cur_residual_out, hidden_in, mixes_partial_out, pre_fn, prev_comb_mix,
        // prev_post_mix, prev_residual, sqrsum_partial_out; then num_tokens, split_k.
        params: vec![p_u64(cur_residual_out), p_u64(hidden_in), p_u64(mixes_partial_out), p_u64(pre_fn), p_u64(prev_comb_mix),
                     p_u64(prev_post_mix), p_u64(prev_residual), p_u64(sqrsum_partial_out), p_i32(t as i32), p_i32(split_k as i32)],
    }.run(k, s)
}

/// (mix-output tile, split count) of the fused post/pre path for `t` tokens (t <= 32).
pub fn fused_post_pre_shape(t: usize) -> (usize, usize) {
    assert!(t <= 32, "fused mHC post/pre covers at most 32 tokens");
    // DeepSeek-V4-Pro (hidden 7168) runs 4 splits at every size.
    if t < 8 { (2, if cfg!(feature = "pro") { 4 } else { 8 }) } else { (3, 4) }
}

/// `mhc_post_tilelang` (the last sublayer's mHC post when no pre follows): out [t, 4, h] =
/// post * x + comb^T residual. TileLang parameters a = comb, b = residual, c = post, d = x, x = out;
/// 4096 / gcd(4096, 1024) hidden blocks for t <= 256.
pub fn post(k: &Kernels, s: Stream, comb: u64, residual: u64, post_mix: u64, x: u64, out: u64, t: usize) -> Result<()> {
    // num_h_blocks: 4096 / gcd(4096, 1024) for at most 256 tokens (more parallelism), else 1 (a
    // compile-time constant: two TileLang kernels of one name).
    // (4 at hidden 4096, 7 at 7168.)
    let (hb, smem) = if t <= 256 { ((crate::model::HIDDEN / 1024) as u32, 20480) } else { (1u32, 28672) };
    Launch { name: &format!("mhc_post_tilelang_kernel#hb={hb}"), grid: [t as u32, hb, 1], block: [128, 1, 1], smem, attrs: vec![attr_pdl()],
             params: vec![p_u64(comb), p_u64(residual), p_u64(post_mix), p_u64(x), p_u64(out), p_i32(t as i32)] }.run(k, s)
}

/// `_hc_head_kernel` (Triton): collapse the 4 hyper-connection streams into the final hidden state,
/// y [t, h] = sum_m (sigmoid(rmsnorm-scaled mix_m) + eps) * x[m].
pub fn head(k: &Kernels, s: Stream, x: u64, hc_fn: u64, hc_scale: u64, hc_base: u64, y: u64, t: usize) -> Result<()> {
    use crate::runtime::{triton, TArg};
    triton(k, s, "_hc_head_kernel", [t as u32, 1, 1], 4, 2048,
           &[TArg::Ptr(x), TArg::Ptr(hc_fn), TArg::Ptr(hc_scale), TArg::Ptr(hc_base), TArg::Ptr(y)], "")
}
