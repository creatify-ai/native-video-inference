//! DeepGEMM host logic (sgl-deep-gemm 0.2.3.post2, csrc/jit_kernels/impls/*), ported: TMA
//! descriptors, launch configurations and the JIT kernels' template instantiations. The kernels
//! themselves are the reference's JIT CUBINs from the kernel bundle.
use crate::cuda::{encode_tiled, TensorMap};
use crate::runtime::{attr_pdl, p_map, p_u32, Kernels, Launch};
use crate::Result;

// CUtensorMapDataType
pub const DT_UINT8: u32 = 0;
pub const DT_INT32: u32 = 3;
pub const DT_FLOAT32: u32 = 7;
pub const DT_BFLOAT16: u32 = 9;
pub const DT_TFLOAT32: u32 = 11;

/// `get_swizzle_mode`: the largest of 128/64/32 bytes dividing the block's inner bytes, else 0.
pub fn swizzle_mode(block_inner: usize, elem: usize) -> usize {
    let bytes = block_inner * elem;
    for m in [128, 64, 32] { if bytes % m == 0 { return m; } }
    0
}

fn swizzle_enum(mode: usize) -> u32 {
    match mode { 0 | 16 => 0, 32 => 1, 64 => 2, 128 => 3, _ => panic!("swizzle {mode}") }
}

/// `make_tma_2d_desc` (L2 promotion 256B, no OOB fill, element strides 1).
pub fn tma_2d(dtype: u32, elem: usize, addr: u64, gmem_inner: usize, gmem_outer: usize, mut smem_inner: usize,
              smem_outer: usize, outer_stride_elems: usize, swizzle: usize) -> Result<TensorMap> {
    if swizzle != 0 { smem_inner = swizzle / elem; }
    encode_tiled(dtype, addr, &[gmem_inner as u64, gmem_outer as u64], &[(outer_stride_elems * elem) as u64],
                 &[smem_inner as u32, smem_outer as u32], &[1, 1], 0, swizzle_enum(swizzle), 3, 0)
}

/// `make_tma_3d_desc`.
#[allow(clippy::too_many_arguments)]
pub fn tma_3d(dtype: u32, elem: usize, addr: u64, g: [usize; 3], mut s: [usize; 3], stride0: usize, stride1: usize,
              swizzle: usize) -> Result<TensorMap> {
    if swizzle != 0 { s[0] = swizzle / elem; }
    encode_tiled(dtype, addr, &[g[0] as u64, g[1] as u64, g[2] as u64], &[(stride0 * elem) as u64, (stride1 * elem) as u64],
                 &[s[0] as u32, s[1] as u32, s[2] as u32], &[1, 1, 1], 0, swizzle_enum(swizzle), 3, 0)
}

/// `_compute_num_split_for_mhc_pre` (SGLang mhc.py) on a 148-SM B300.
pub fn mhc_pre_splits(num_tokens: usize, hc_hidden: usize, num_sms: usize) -> usize {
    let grid = num_tokens.div_ceil(64).max(1);
    let blocks_k = hc_hidden.div_ceil(64);
    (num_sms / grid).min(blocks_k / 4).max(1)
}

/// `sm100_tf32_hc_prenorm_gemm`: D[split, m, n] (fp32 partial sums over K splits) = A[m, k] (bf16)
/// x B[n, k]^T (fp32 as TF32), and sqr_sum[split, m] = sum over the split of A^2.
#[allow(clippy::too_many_arguments)]
pub fn tf32_hc_prenorm_gemm(kernels: &Kernels, stream: crate::cuda::Stream, a: u64, b: u64, d: u64, sqr_sum: u64,
                            m: usize, n: usize, k: usize, num_splits: usize) -> Result<()> {
    let (block_m, block_k) = (64usize, 64usize);
    let block_n = n.div_ceil(16) * 16;
    assert!(n <= 128 && n % 8 == 0 && k % block_k == 0);
    let swizzle_cd = swizzle_mode(block_n, 4);
    let map_a = tma_2d(DT_BFLOAT16, 2, a, k, m, block_k, block_m, k, swizzle_mode(block_k, 2))?;
    let map_b = tma_2d(DT_TFLOAT32, 4, b, k, n, block_k, block_n, k, swizzle_mode(block_k, 4))?;
    let map_d = if num_splits == 1 {
        tma_2d(DT_FLOAT32, 4, d, n, m, block_n, block_m, n, swizzle_cd)?
    } else {
        tma_3d(DT_FLOAT32, 4, d, [n, m, num_splits], [block_n, block_m, 1], n, m * n, swizzle_cd)?
    };
    let mut stages = 12usize;
    let mut smem = 0usize;
    while stages > 0 {
        smem = (block_m * block_k * 2 + block_n * block_k * 4) * stages + block_m * swizzle_cd + (stages * 4 + 1) * 8 + 4;
        if smem <= 232448 { break; }
        stages -= 1;
    }
    let name = format!(
        "_ZN9deep_gemm31sm100_tf32_hc_prenorm_gemm_implILj{n}ELj{k}ELj{block_m}ELj{block_n}ELj{block_k}ELj{num_splits}ELj{swizzle_cd}ELj{stages}ELj128ELj128EEEvj14CUtensorMap_stS1_S1_Pf");
    Launch {
        name: &name, grid: [(num_splits * m.div_ceil(block_m)) as u32, 1, 1], block: [256, 1, 1], smem: smem as u32,
        attrs: vec![attr_pdl()],
        params: vec![p_u32(m as u32), p_map(&map_a), p_map(&map_b), p_map(&map_d), sqr_sum.to_le_bytes().to_vec()],
    }.run(kernels, stream)
}

// ---------------------------------------------------------------------------------------------
// FP8 x FP8 -> BF16 1D1D GEMMs (sm100_fp8_fp4_gemm_1d1d.hpp) with the SM100 heuristics
// (heuristics/sm100.hpp, get_best_config). A and B are K-major FP8 with 1x128 / 128x128 UE8M0
// scales packed MN-major; D is BF16.
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Layout { pub swap_ab: bool, pub block_m: usize, pub block_n: usize, pub block_k: usize, pub cluster_m: usize, pub cluster_n: usize }

impl Layout { fn cluster(&self) -> usize { self.cluster_m * self.cluster_n } }

#[derive(Clone, Copy, Debug)]
pub struct GemmDesc { pub batched: bool, pub m: usize, pub n: usize, pub k: usize, pub groups: usize, pub num_sms: usize }

#[derive(Clone, Copy, Debug)]
pub struct Config {
    pub layout: Layout,
    pub load_block_m: usize, pub load_block_n: usize, pub store_block_m: usize, pub store_block_n: usize,
    pub swizzle_a: usize, pub swizzle_b: usize, pub swizzle_cd: usize,
    pub smem: usize, pub stages: usize, pub store_stages: usize,
}

fn align(x: usize, a: usize) -> usize { x.div_ceil(a) * a }

/// `get_swizzle_mode` of the heuristics (includes 16 = no swizzle, interleaved).
fn hswizzle(block: usize, elem: usize) -> usize {
    for m in [128, 64, 32, 16] { if (block * elem) % m == 0 { return m; } }
    unreachable!()
}

fn candidates(d: &GemmDesc) -> Vec<Layout> {
    // FP8 x FP8 (MXFP8FP4 MMA): block K 128, SF granularity 32, SF blocks UTCCP-aligned to 128.
    let block_k = 128;
    let mut out = Vec::new();
    for swap in [false, true] {
        let (bms, bns): (Vec<usize>, Vec<usize>) = if swap {
            ((16..=256).step_by(16).collect(), vec![128])
        } else {
            let bm = if d.m <= 32 { 32 } else if d.m <= 64 { 64 } else { 128 };
            let end = if d.k <= 256 { 128 } else { 256 };
            let mut bn = vec![16];
            bn.extend((32..=end).step_by(32));
            (vec![bm], bn)
        };
        for cm in 1..=2 {
            if swap && cm > 1 { continue; }
            for cn in 1..=2 {
                if cm * cn > 2 || (!swap && cn > 1) || d.num_sms % (cm * cn) != 0 { continue; }
                for &bm in &bms {
                    if (bm / cn) % 8 != 0 || d.m.div_ceil(bm) % cm != 0 { continue; }
                    for &bn in &bns {
                        if (bn / cm) % 8 != 0 || d.n.div_ceil(bn) % cn != 0 { continue; }
                        if swap && bn != 128 { continue; }
                        let tmem_sf_cols = (align(bm, 128) + align(bn, 128)) * (block_k / (32 * 4)) / 32;
                        let umma_n = if swap { bm } else { bn };
                        if umma_n + tmem_sf_cols > 512 { continue; }
                        out.push(Layout { swap_ab: swap, block_m: bm, block_n: bn, block_k, cluster_m: cm, cluster_n: cn });
                    }
                }
            }
        }
    }
    assert!(!out.is_empty());
    out
}

struct Info { waves: usize, last_util: usize, layout: Layout }

fn info(d: &GemmDesc, l: Layout) -> Info {
    let blocks = d.m.div_ceil(l.block_m) * d.n.div_ceil(l.block_n) * d.groups;
    let waves = blocks.div_ceil(d.num_sms);
    let last = blocks % d.num_sms;
    Info { waves, last_util: if last == 0 { d.num_sms } else { last }, layout: l }
}

/// SM100ArchSpec::compare: is `a` better than `b`?
fn better(a: &Info, b: &Info) -> bool {
    if (a.waves == 1 || b.waves == 1) && a.waves != b.waves { return a.waves < b.waves; }
    if a.layout.cluster() != b.layout.cluster() { return a.layout.cluster() > b.layout.cluster(); }
    if a.waves != b.waves { return a.waves < b.waves; }
    if a.last_util != b.last_util { return a.last_util > b.last_util; }
    let (sa, sb) = (a.layout.block_m + a.layout.block_n, b.layout.block_m + b.layout.block_n);
    if sa != sb { return sa < sb; }
    a.layout.block_m * a.layout.block_n < b.layout.block_m * b.layout.block_n
}

pub fn best_config(d: &GemmDesc) -> Config {
    let cands = candidates(d);
    let mut best = info(d, cands[0]);
    for &c in &cands[1..] {
        let i = info(d, c);
        if better(&i, &best) { best = i; }
    }
    config_for(best.layout)
}

/// The heuristic's configs for `d`, best first.
fn ranked_configs(d: &GemmDesc) -> Vec<Config> {
    let mut infos: Vec<Info> = candidates(d).into_iter().map(|c| info(d, c)).collect();
    // Insertion order on ties, as best_config's scan keeps the first of equals.
    let mut ranked: Vec<Info> = Vec::with_capacity(infos.len());
    for i in infos.drain(..) {
        let at = ranked.iter().position(|r| better(&i, r)).unwrap_or(ranked.len());
        ranked.insert(at, i);
    }
    ranked.into_iter().map(|i| config_for(i.layout)).collect()
}

/// The config SGLang's heuristic picks for `d` when the bundle has its kernel (the reference ran
/// that shape), else the best-ranked one whose kernel the bundle has. The block shapes tile M and
/// N only: every output element sums its K blocks in the same order whatever the tiling (the
/// 16k prefill's wo_a runs 128x256 natively, 240x128 swapped in SGLang, bit for bit equal), so a
/// shape the reference never compiled (a packed prefill of 16129 tokens) runs a compiled tiling.
fn available_config(kernels: &Kernels, d: &GemmDesc, name: impl Fn(&Config) -> String) -> Result<(Config, String)> {
    let best = best_config(d);
    let n = name(&best);
    if kernels.has(&n) { return Ok((best, n)); }
    for c in ranked_configs(d) {
        let n2 = name(&c);
        if kernels.has(&n2) { return Ok((c, n2)); }
    }
    Err(format!("kernel bundle has no kernel {n} (nor another tiling of the shape)"))
}

fn config_for(l: Layout) -> Config {
    let (load_m, load_n) = (l.block_m / l.cluster_n, l.block_n / l.cluster_m);
    let store_m = if l.swap_ab { 16 } else { l.block_m.min(128) };
    let store_n = l.block_n;
    let (swa, swb, swcd) = (hswizzle(l.block_k, 1), hswizzle(l.block_k, 1), hswizzle(store_n, 2));
    let store_stages = 2;
    let smem_cd = if l.swap_ab { store_m * store_n * 2 } else { store_m * swcd } * store_stages;
    let extra = smem_cd + (32 * 8 * 3 + 2 * 8 * 3 + 8) + 4;
    let per_stage = load_m * l.block_k + load_n * l.block_k + align(l.block_m, 128) * l.block_k / 32 + align(l.block_n, 128) * l.block_k / 32;
    let stages = ((232448 - extra) / per_stage).min(32);
    Config { layout: l, load_block_m: load_m, load_block_n: load_n, store_block_m: store_m, store_block_n: store_n,
             swizzle_a: swa, swizzle_b: swb, swizzle_cd: swcd, smem: extra + stages * per_stage, stages, store_stages }
}

fn gemm_name(c: &Config, d: &GemmDesc, compiled: (usize, usize, usize)) -> String {
    let l = &c.layout;
    let b = |v: bool| if v { 1 } else { 0 };
    format!(
        "_ZN9deep_gemm28sm100_fp8_fp4_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj128ELj128ELj128ELj{}ELj{}ELj{}ELj{}ELj{}ELj{}ELj{}ELj{}ELj{}ELj{}ELj{}ELj{}ELj128ELj128ELj{}ELb{}ELj{}ELb{}ELb1ELNS_8GemmTypeE{}ELb0EN7cutlass12float_e4m3_tES6_NS5_10bfloat16_tENS_8epilogue9transform16EpilogueIdentityELb1EEEvPijjjT28_14CUtensorMap_stSD_SD_SD_SD_",
        compiled.0, compiled.1, compiled.2, l.block_m, l.block_n, l.block_k, d.groups, c.swizzle_a, c.swizzle_b, c.swizzle_cd,
        c.stages, c.store_stages, l.cluster(), b(l.cluster_n > 1), d.num_sms, b(l.swap_ab), if d.batched { 4 } else { 0 })
}

/// EpilogueArgs for EpilogueIdentity: sfd null, strides/shape zero, alpha 1.
fn epilogue_identity() -> Vec<u8> {
    let mut v = vec![0u8; 20];
    v.extend(1.0f32.to_le_bytes());
    v
}

/// `make_tma_sf_desc`: packed UE8M0 int32 scales, MN-major, per-128 K sub-blocks.
fn tma_sf(addr: u64, mn: usize, k: usize, block_mn: usize, groups: usize, sf_block_k: usize) -> Result<TensorMap> {
    let shape_mn = align(mn, 4);
    tma_2d(DT_INT32, 4, addr, shape_mn, k.div_ceil(128 * 4) * groups, block_mn, sf_block_k, shape_mn, 0)
}

fn launch_gemm(kernels: &Kernels, stream: crate::cuda::Stream, c: &Config, d: &GemmDesc, name: &str, maps: [TensorMap; 5]) -> Result<()> {
    let mut attrs = Vec::new();
    if c.layout.cluster() > 1 { attrs.push(crate::runtime::attr_cluster(c.layout.cluster() as u32, 1, 1)); }
    attrs.push(attr_pdl());
    let mut params = vec![0u64.to_le_bytes().to_vec(), p_u32(d.m as u32), p_u32(d.n as u32), p_u32(d.k as u32), epilogue_identity()];
    params.extend(maps.iter().map(p_map));
    Launch { name, grid: [d.num_sms as u32, 1, 1], block: [256, 1, 1], smem: c.smem as u32, attrs, params }.run(kernels, stream)
}

/// `fp8_gemm_nt` (compiled_dims ""): D[m, n] bf16 = A[m, k] fp8 x B[n, k]^T fp8 with packed scales
/// `sfa` ([m] x k/512 int32, MN-major, M aligned to 4) and `sfb` (likewise for N).
#[allow(clippy::too_many_arguments)]
pub fn fp8_gemm_nt(kernels: &Kernels, stream: crate::cuda::Stream, a: u64, sfa: u64, b: u64, sfb: u64, d_ptr: u64,
                   m: usize, n: usize, k: usize, num_sms: usize) -> Result<()> {
    let d = GemmDesc { batched: false, m, n, k, groups: 1, num_sms };
    let (c, name) = available_config(kernels, &d, |c| gemm_name(c, &d, (0, 0, 0)))?;
    let l = c.layout;
    let maps = [
        tma_2d(DT_UINT8, 1, a, k, m, l.block_k, c.load_block_m, k, c.swizzle_a)?,
        tma_2d(DT_UINT8, 1, b, k, n, l.block_k, c.load_block_n, k, c.swizzle_b)?,
        tma_sf(sfa, m, k, align(l.block_m, 128), 1, l.block_k / 128)?,
        tma_sf(sfb, n, k, align(l.block_n, 128), 1, l.block_k / 128)?,
        tma_2d(DT_BFLOAT16, 2, d_ptr, n, m, c.store_block_n, c.store_block_m, n, c.swizzle_cd)?,
    ];
    launch_gemm(kernels, stream, &c, &d, &name, maps)
}

/// The batched FP8 GEMM behind `fp8_einsum` (compiled_dims "nk"): for each of `groups` batches,
/// D[g] (m x n) = A[g] (m x k) x B[g] (n x k)^T. A and D are token-major views: element (g, i, j)
/// of A sits at i * a_token_stride + g * k + j; of D at i * d_token_stride + g * n + j.
#[allow(clippy::too_many_arguments)]
pub fn fp8_bmm_token_major(kernels: &Kernels, stream: crate::cuda::Stream, a: u64, sfa: u64, b: u64, sfb: u64, d_ptr: u64,
                           groups: usize, m: usize, n: usize, k: usize, num_sms: usize) -> Result<()> {
    let d = GemmDesc { batched: true, m, n, k, groups, num_sms };
    let (c, name) = available_config(kernels, &d, |c| gemm_name(c, &d, (0, n, k)))?;
    let l = c.layout;
    let maps = [
        tma_3d(DT_UINT8, 1, a, [k, m, groups], [l.block_k, c.load_block_m, 1], groups * k, k, c.swizzle_a)?,
        tma_3d(DT_UINT8, 1, b, [k, n, groups], [l.block_k, c.load_block_n, 1], k, n * k, c.swizzle_b)?,
        tma_sf(sfa, m, k, align(l.block_m, 128), groups, l.block_k / 128)?,
        tma_sf(sfb, n, k, align(l.block_n, 128), groups, l.block_k / 128)?,
        tma_3d(DT_BFLOAT16, 2, d_ptr, [n, m, groups], [c.store_block_n, c.store_block_m, 1], groups * n, n, c.swizzle_cd)?,
    ];
    launch_gemm(kernels, stream, &c, &d, &name, maps)
}

// ---------------------------------------------------------------------------------------------
// FP8 MQA logits of the sparse indexer (sm100_mqa_logits.hpp, contiguous KV): logits[i, j] =
// sum over heads of relu(q[i, h] . kv[j]) * kv_scale[j] * weights[i, h], for j in [ks[i], ke[i]).
// ---------------------------------------------------------------------------------------------

/// `get_mqa_logits_smem_size` for FP8 Q/KV (no MX scales) and fp32 weights.
pub fn mqa_logits_smem(num_heads: usize, head_dim: usize, block_q: usize, split_kv: usize, q_stages: usize, kv_stages: usize, tmem_stages: usize) -> usize {
    let swizzle_alignment = 8 * head_dim;
    let umma_n = align(block_q * num_heads, 8);
    let mut smem = 0usize;
    let mut add = |bytes: usize, alignment: usize| { smem = align(smem, alignment) + bytes; };
    add(q_stages * umma_n * head_dim, swizzle_alignment);
    add(kv_stages * split_kv * head_dim, swizzle_alignment);
    add(q_stages * 4, 128);              // sf_q: one element without MX scales
    add(kv_stages * split_kv * 4, 128);
    let row = align(num_heads * 4, 16);
    add(q_stages * align(block_q * row, 128), 128);
    add(2 * (q_stages + kv_stages + tmem_stages) * 8, 8);
    add(4, 4);
    align(smem, swizzle_alignment)
}

/// The DeepGEMM `fp8_mqa_logits` row stride (fp32 elements) for compressed logits
/// (`max_seqlen_k` > 0, 256-token KV split): 1024-byte aligned.
pub fn mqa_logits_stride(max_seqlen_k: usize) -> usize { align(align(max_seqlen_k, 256), 256) }

/// `fp8_mqa_logits(q, (kv, kv_scales), weights, ks, ke, clean_logits=False, max_seqlen_k)` on SM100:
/// q [num_q, heads, 128] fp8, kv [num_kv, 128] fp8, kv_scales [num_kv] f32, weights [num_q, heads]
/// f32, ks / ke int32 [num_q]; logits [num_q, stride] f32 with stride `mqa_logits_stride`.
#[allow(clippy::too_many_arguments)]
pub fn mqa_logits(kernels: &Kernels, stream: crate::cuda::Stream, q: u64, kv: u64, kv_scales: u64, weights: u64, ks: u64, ke: u64,
                  logits: u64, num_q: usize, num_kv: usize, max_seqlen_k: usize, num_heads: usize, num_sms: usize) -> Result<()> {
    let head_dim = 128usize;
    let block_q = 128 / num_heads;
    let split_kv = 256usize;          // fp32 weights keep the 256-token split
    let (q_stages, kv_stages, kv_tma_tokens) = (3usize, 5usize, 256usize);
    let math_threads = split_kv / 128 * 128;
    let stride = mqa_logits_stride(max_seqlen_k);
    let swizzle = head_dim;           // FP8: one 128-byte row per token
    let map_q = tma_2d(DT_UINT8, 1, q, head_dim, num_q * num_heads, head_dim, block_q * num_heads, head_dim, swizzle)?;
    let map_kv = tma_2d(DT_UINT8, 1, kv, head_dim, num_kv, head_dim, kv_tma_tokens, head_dim, swizzle)?;
    let map_sf_kv = tma_2d(DT_FLOAT32, 4, kv_scales, align(num_kv, 4), 1, kv_tma_tokens, 1, 0, 0)?;
    let row_elems = align(num_heads * 4, 16) / 4;
    let map_w = tma_2d(DT_FLOAT32, 4, weights, num_heads, num_q, row_elems, block_q, num_heads, 0)?;
    let smem = mqa_logits_smem(num_heads, head_dim, block_q, split_kv, q_stages, kv_stages, 3);
    let name = format!(
        "_ZN9deep_gemm16sm100_mqa_logitsILj{num_heads}ELj{head_dim}ELb0ELb1ELb0ELb0ELj{block_q}ELj{split_kv}ELj{q_stages}ELj{kv_stages}ELj{num_sms}ELj128ELj{math_threads}EN7cutlass12float_e4m3_tEffLj2EEEvjjjPKjS4_S4_PT13_14CUtensorMap_stS7_S7_S7_S7_");
    Launch {
        name: &name, grid: [num_sms as u32, 1, 1], block: [(128 + math_threads) as u32, 1, 1], smem: smem as u32, attrs: vec![attr_pdl()],
        params: vec![p_u32(num_q as u32), p_u32(num_kv as u32), p_u32(stride as u32), ks.to_le_bytes().to_vec(), ke.to_le_bytes().to_vec(),
                     0u64.to_le_bytes().to_vec(), logits.to_le_bytes().to_vec(),
                     p_map(&map_q), p_map(&map_sf_kv), p_map(&map_kv), p_map(&map_sf_kv), p_map(&map_w)],
    }.run(kernels, stream)
}

/// The paged MQA-logits schedule (SGLang paged_mqa_metadata.cuh, `paged_mqa_metadata_*_kernel`,
/// which matches DeepGEMM's): per SM i in 0..=num_sms, the (row, 256-token split) where its work
/// starts, balancing ceil(len / 256) splits over the SMs. Row-major [num_sms + 1, 2] int32. The
/// schedule only distributes work; every logit is computed by one CTA either way.
pub fn paged_mqa_schedule(context_lens: &[i32], num_sms: usize) -> Vec<i32> {
    let bs = context_lens.len() as u32;
    let mut prefix = Vec::with_capacity(context_lens.len());
    let mut sum = 0u32;
    for &l in context_lens { sum += (l as u32).div_ceil(256); prefix.push(sum); }
    let n = num_sms as u32;
    let (avg, ret) = (sum / n, sum % n);
    let pivot = n - ret;
    let mut out = Vec::with_capacity(2 * (num_sms + 1));
    for i in 0..=n {
        let target = i * avg + if i > pivot { i - pivot } else { 0 };
        let q = prefix.partition_point(|&p| p <= target) as u32;
        if q >= bs { out.extend([bs as i32, 0]); } else {
            let prev = if q == 0 { 0 } else { prefix[q as usize - 1] };
            out.extend([q as i32, (target - prev) as i32]);
        }
    }
    out
}

/// `fp8_paged_mqa_logits(q, kv_cache, weights, context_lens, block_table, schedule_meta,
/// max_context_len, clean_logits=False)` on SM100 (sm100_paged_mqa_logits, one token per request):
/// q [rows, heads, 128] fp8; the index-K cache [num_blocks, 64 x (128 + 4)] bytes; weights
/// [rows, heads] f32; context_lens int32 [rows]; block_table int32 rows of `block_table_stride`;
/// logits [rows, stride] f32 with stride = `mqa_logits_stride(max_context_len)`.
#[allow(clippy::too_many_arguments)]
pub fn paged_mqa_logits(kernels: &Kernels, stream: crate::cuda::Stream, q: u64, kv_cache: u64, num_blocks: usize, weights: u64, context_lens: u64,
                        block_table: u64, block_table_stride: usize, schedule: u64, logits: u64, rows: usize, max_context_len: usize,
                        num_heads: usize, num_sms: usize) -> Result<()> {
    let (head_dim, page_kv, split_kv) = (128usize, 64usize, 256usize);
    let block_q = 128 / num_heads;
    let page_bytes = page_kv * (head_dim + 4);
    let stride = mqa_logits_stride(max_context_len);
    let map_q = tma_2d(DT_UINT8, 1, q, head_dim, rows * num_heads, head_dim, block_q * num_heads, head_dim, head_dim)?;
    let map_kv = tma_3d(DT_UINT8, 1, kv_cache, [head_dim, page_kv, num_blocks], [head_dim, page_kv, 1], head_dim, page_bytes, head_dim)?;
    let map_sf_kv = tma_2d(DT_FLOAT32, 4, kv_cache + (page_kv * head_dim) as u64, page_kv, num_blocks, page_kv, 1, page_bytes / 4, 0)?;
    let row_elems = align(num_heads * 4, 16) / 4;
    let map_w = tma_2d(DT_FLOAT32, 4, weights, num_heads, rows, row_elems, block_q, num_heads, 0)?;
    let smem = mqa_logits_smem(num_heads, head_dim, block_q, split_kv, 3, 5, 3);
    let name = format!(
        "_ZN9deep_gemm22sm100_paged_mqa_logitsILj1ELj{num_heads}ELj{head_dim}ELj{page_kv}ELb0ELb1ELb0ELj3ELj5ELj{split_kv}ELj16ELj128ELj256EN7cutlass12float_e4m3_tEffLj2EEEvjjjPKjPT13_S4_S4_S4_14CUtensorMap_stS7_S7_S7_S7_");
    Launch {
        name: &name, grid: [num_sms as u32, 1, 1], block: [384, 1, 1], smem: smem as u32, attrs: vec![attr_pdl()],
        params: vec![p_u32(rows as u32), p_u32(stride as u32), p_u32(block_table_stride as u32), context_lens.to_le_bytes().to_vec(),
                     logits.to_le_bytes().to_vec(), block_table.to_le_bytes().to_vec(), 0u64.to_le_bytes().to_vec(), schedule.to_le_bytes().to_vec(),
                     p_map(&map_q), p_map(&map_sf_kv), p_map(&map_kv), p_map(&map_sf_kv), p_map(&map_w)],
    }.run(kernels, stream)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn heuristics_pick_the_reference_configs() {
        // wqkv_a on 5 tokens: swap AB, 16x128, cluster 2, 12 stages, 230204 B (from the tape).
        let c = best_config(&GemmDesc { batched: false, m: 5, n: 1536, k: 4096, groups: 1, num_sms: 148 });
        assert_eq!((c.layout.swap_ab, c.layout.block_m, c.layout.block_n, c.layout.cluster_n, c.stages, c.smem), (true, 16, 128, 2, 12, 230204));
        // wq_b: 32x224, one wave of 147 blocks, 6 stages, 210748 B, CD swizzle 64.
        let c = best_config(&GemmDesc { batched: false, m: 5, n: 32768, k: 1024, groups: 1, num_sms: 148 });
        assert_eq!((c.layout.swap_ab, c.layout.block_m, c.layout.block_n, c.layout.cluster(), c.stages, c.smem, c.swizzle_cd), (false, 32, 224, 1, 6, 210748, 64));
    }

    #[test]
    fn mqa_logits_smem_matches_the_tape() {
        // C4 indexer at 16384 queries: 64 heads, block_q 2, 256-token split (launch smem 220160).
        assert_eq!(mqa_logits_smem(64, 128, 2, 256, 3, 5, 3), 220160);
        assert_eq!(mqa_logits_stride(4096), 4096);
        // Decode (max context 4097 pages of 64): the tape's logits stride.
        assert_eq!(mqa_logits_stride(4097 * 64), 262400);
    }

    #[test]
    fn paged_schedule_balances_splits() {
        // One row of 4097 splits over 148 SMs: 27 or 28 splits each, in order.
        let s = paged_mqa_schedule(&[4097 * 256], 148);
        assert_eq!(&s[..4], &[0, 0, 0, 27]);
        assert_eq!(&s[s.len() - 2..], &[1, 0]);
        // Rows of 1 split each: SM i starts at row i * avg (+ the remainder spread at the end).
        let s = paged_mqa_schedule(&[256; 296], 148);
        assert_eq!(&s[..6], &[0, 0, 2, 0, 4, 0]);
    }
}
