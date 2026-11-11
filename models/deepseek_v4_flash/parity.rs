//! Parity rungs against the reference's golds (reference/gold): run native pieces of the forward
//! from gold inputs and compare every produced tensor's bytes with the gold's SHA-256.
use crate::cuda::{download, Buffer, Device};
use crate::gold::{compare, Comparison, Gold};
use crate::runtime::{Arena, Kernels};
use crate::safetensors::Checkpoint;
use crate::weights::Source;
use crate::snapshots::{check_prefix, Snapshots};
use crate::{deepgemm, flashmla, mhc, moe, sgl, torch_ops, Result};
use std::collections::HashMap;
use std::path::Path;

/// Device copies of SGLang's loaded parameters, in the memory layout its kernels read: either
/// rebuilt from the checkpoint and uploaded one by one on first use (development), or all loaded
/// up front from a prepared package (prepare.rs, prepared.rs).
pub struct DeviceWeights<'a> { source: Option<Source<'a>>, pub tensors: HashMap<String, Buffer>, prepared: Option<crate::prepared::Weights>,
                                /// Parameters loaded beside the package (the MTP draft's), found first.
                                extra: HashMap<String, Buffer>,
                                /// A rank package read tensor by tensor on first use (development).
                                lazy: Option<LazyPackage> }

/// A rank package (written by the reference's gold hook: `rank<r>/manifest.json`, `params.json`,
/// `weights/`) read one tensor at a time, on first use, for rungs that touch a few layers. Each
/// tensor's bytes are checked against the SHA-256 the reference rank recorded.
pub struct LazyPackage { dir: std::path::PathBuf, tensors: HashMap<String, (String, u64, usize)>, files: HashMap<String, String>,
                         meta: HashMap<String, (Option<String>, usize, usize, Option<String>)>, dummy: Option<Buffer> }

impl LazyPackage {
    pub fn open(dir: &Path) -> Result<Self> {
        let read = |f: &str| -> Result<serde_json::Value> {
            serde_json::from_slice(&std::fs::read(dir.join(f)).map_err(|e| format!("{}/{f}: {e}", dir.display()))?).map_err(|e| format!("{f}: {e}"))
        };
        let manifest = read("manifest.json")?;
        let files = manifest["files"].as_object().ok_or("manifest has no files")?.iter()
            .map(|(k, v)| (k.clone(), v["path"].as_str().unwrap_or("").to_string())).collect();
        let tensors = manifest["tensors"].as_object().ok_or("manifest has no tensors")?.iter()
            .map(|(k, v)| (k.clone(), (v["file"].as_str().unwrap_or("").to_string(), v["offset"].as_u64().unwrap_or(0), v["bytes"].as_u64().unwrap_or(0) as usize))).collect();
        // name -> (device SHA-256, device bytes, pieces, alias target)
        let meta = read("params.json")?["params"].as_array().ok_or("params.json has no params")?.iter().map(|r| {
            (r["name"].as_str().unwrap_or("").to_string(),
             (r["device_sha256"].as_str().map(String::from), r["device_bytes"].as_u64().unwrap_or(0) as usize,
              r["pieces"].as_u64().unwrap_or(0) as usize, r["alias_of"].as_str().map(String::from)))
        }).collect();
        // Dry runs (DSV4_DRY=1) launch nothing from the bundle: every parameter is one scratch buffer.
        let dummy = if crate::runtime::dry() { Some(Buffer::alloc(1280 << 20)?) } else { None };
        Ok(Self { dir: dir.to_path_buf(), tensors, files, meta, dummy })
    }

    fn read(&self, name: &str) -> Result<Vec<u8>> {
        use std::io::{Read, Seek, SeekFrom};
        let (sha, bytes, pieces, _) = self.meta.get(name).ok_or_else(|| format!("parameter {name} is not in the rank package"))?;
        let names: Vec<String> = if *pieces > 0 { (0..*pieces).map(|k| crate::prepare::piece_name(name, k)).collect() } else { vec![name.to_string()] };
        let mut out = Vec::with_capacity(*bytes);
        for n in names {
            let (file, offset, len) = self.tensors.get(&n).ok_or_else(|| format!("{n} is not in the manifest"))?;
            let path = self.dir.join("weights").join(self.files.get(file).ok_or_else(|| format!("manifest file {file}"))?);
            let mut f = std::fs::File::open(&path).map_err(|e| format!("{}: {e}", path.display()))?;
            f.seek(SeekFrom::Start(*offset)).map_err(|e| e.to_string())?;
            let at = out.len();
            out.resize(at + len, 0);
            f.read_exact(&mut out[at..]).map_err(|e| format!("{}: {e}", path.display()))?;
        }
        if out.len() != *bytes || sha.as_deref() != Some(crate::gold::sha(&out).as_str()) {
            return Err(format!("{name}: package bytes differ from the reference rank's SHA-256"));
        }
        Ok(out)
    }
}

impl<'a> DeviceWeights<'a> {
    pub fn new(ckpt: &'a Checkpoint) -> Self { Self { source: Some(Source { ckpt }), tensors: HashMap::new(), prepared: None, extra: HashMap::new(), lazy: None } }

    pub fn from_prepared(weights: crate::prepared::Weights) -> Self { Self { source: None, tensors: HashMap::new(), prepared: Some(weights), extra: HashMap::new(), lazy: None } }

    /// No weights yet (a model assembled while its package is still loading).
    pub fn empty() -> Self { Self { source: None, tensors: HashMap::new(), prepared: None, extra: HashMap::new(), lazy: None } }

    pub fn from_lazy(package: LazyPackage) -> Self { Self { source: None, tensors: HashMap::new(), prepared: None, extra: HashMap::new(), lazy: Some(package) } }

    /// The streaming load's statistics, for a prepared package.
    pub fn load_stats(&self) -> Option<&native_core_infra::model_load::Stats> { self.prepared.as_ref().map(|p| &p.stats) }

    /// The parameter's device address (uploaded on first use when rebuilding from the checkpoint).
    /// Add a parameter beside the package (its buffer is kept for the weights' lifetime).
    pub fn insert(&mut self, name: &str, b: Buffer) { self.extra.insert(name.to_string(), b); }

    pub fn get(&mut self, name: &str) -> Result<u64> {
        if let Some(b) = self.extra.get(name) { return Ok(b.ptr); }
        if let Some(p) = &self.prepared { return p.get(name); }
        if let Some(b) = self.tensors.get(name) { return Ok(b.ptr); }
        if let Some(lazy) = &self.lazy {
            if let Some(d) = &lazy.dummy { return Ok(d.ptr); }
            // The same memory under another name in the reference (shared tables).
            if let Some((_, _, _, Some(target))) = lazy.meta.get(name) { let target = target.clone(); return self.get(&target); }
            let buffer = Buffer::from_host(&lazy.read(name)?)?;
            let ptr = buffer.ptr;
            self.tensors.insert(name.to_string(), buffer);
            return Ok(ptr);
        }
        let source = self.source.as_ref().ok_or("no weight source")?;
        let logical = source.prepare(name)?.ok_or_else(|| format!("parameter {name} is not ported"))?;
        let bytes = crate::weights::memory_layout(name, logical);
        let buffer = Buffer::from_host(&bytes)?;
        let ptr = buffer.ptr;
        self.tensors.insert(name.to_string(), buffer);
        Ok(ptr)
    }
}

fn check(gold: &Gold, name: &str, ptr: u64, bytes: usize, out: &mut Vec<Comparison>) -> Result<bool> {
    let c = compare(gold, name, &download(ptr, bytes)?)?;
    let ok = c.equal;
    println!("{}", serde_json::to_string(&c).unwrap());
    out.push(c);
    Ok(ok)
}

/// Rung: prefill layer 0 up to the attention input (embedding, mHC broadcast, prenorm GEMM, mHC pre).
pub fn prefill_layer0_pre_attention(ckpt_dir: &Path, bundle: &Path, gold_dir: &Path) -> Result<Vec<Comparison>> {
    let dev = Device::open(0)?;
    let s = dev.stream;
    let kernels = Kernels::open(bundle)?;
    let ckpt = Checkpoint::open(ckpt_dir)?;
    let mut w = DeviceWeights::new(&ckpt);
    let gold = Gold::open(gold_dir);
    let ids = gold.bytes("batch.input_ids")?;
    let t = ids.len() / 8;
    let (hidden, hc) = (4096usize, 4usize);
    let ids_dev = Buffer::from_host(&ids)?;
    let mut a = Arena::new(64 << 20)?;
    let mut out = Vec::new();

    let embed = a.alloc(t * hidden * 2)?;
    torch_ops::embedding_bf16(&kernels, s, embed, w.get("model.embed_tokens.weight")?, ids_dev.ptr, t, 129280, hidden)?;
    s.synchronize()?;
    check(&gold, "model.embed_tokens.out", embed, t * hidden * 2, &mut out)?;

    let residual = a.alloc(t * hc * hidden * 2)?;
    mhc::hc_broadcast(&kernels, s, embed, residual, t, hidden)?;
    s.synchronize()?;
    check(&gold, "model.layers.0.in.hidden_states", residual, t * hc * hidden * 2, &mut out)?;

    let splits = deepgemm::mhc_pre_splits(t, hc * hidden, dev.sm_count as usize);
    let (mix, hc_hidden) = (24usize, hc * hidden);
    let d = a.alloc(splits * t * mix * 4)?;
    let sq = a.alloc(splits * t * 4)?;
    deepgemm::tf32_hc_prenorm_gemm(&kernels, s, residual, w.get("model.layers.0.hc_attn_fn")?, d, sq, t, mix, hc_hidden, splits)?;
    let pre = mhc::MhcPreOut { post_mix: a.alloc(t * hc * 4)?, comb_mix: a.alloc(t * hc * hc * 4)?, layer_input: a.alloc(t * hidden * 2)? };
    mhc::pre_big_fuse_with_norm(&kernels, s, d, sq, w.get("model.layers.0.hc_attn_scale")?, w.get("model.layers.0.hc_attn_base")?,
                                residual, &pre, w.get("model.layers.0.input_layernorm.weight")?, t, splits)?;
    s.synchronize()?;
    check(&gold, "model.layers.0.self_attn.in.x", pre.layer_input, t * hidden * 2, &mut out)?;
    Ok(out)
}


/// Optional per-launch snapshot checks (DSV4_SNAPSHOTS=<reference tape dir>).
struct SnapCheck { snaps: Option<Snapshots>, label: String }

impl SnapCheck {
    fn new(label: &str) -> Result<Self> {
        let snaps = match std::env::var("DSV4_SNAPSHOTS") { Ok(d) if !d.is_empty() => Some(Snapshots::open(Path::new(&d))?), _ => None };
        Ok(Self { snaps, label: label.into() })
    }
    /// Compare `bytes` at native `ptr` with what the reference's `nth` launch named `fragment` left
    /// behind its parameter word (param, word).
    fn out(&self, s: crate::cuda::Stream, what: &str, fragment: &str, nth: usize, param: usize, word: usize, ptr: u64, bytes: usize) -> Result<()> {
        let Some(snaps) = &self.snaps else { return Ok(()) };
        s.synchronize()?;
        let launch = match snaps.find(&self.label, fragment, nth) { Ok(l) => l, Err(e) => { println!("{}", serde_json::json!({"snapshot": what, "error": e})); return Ok(()) } };
        match snaps.read(&launch, param, word, "out") {
            Ok(snap) => { check_prefix(what, &download(ptr, bytes)?, &snap); }
            Err(e) => println!("{}", serde_json::json!({"snapshot": what, "error": e})),
        }
        Ok(())
    }
}

/// Little-endian bytes of a slice of plain values.
fn le_i32(v: &[i32]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }

/// The SWA-only (compress ratio 0) sparse-prefill indices SGLang builds for one fresh request of
/// `t` tokens (`_combine_topk_swa_indices_kernel`, topk 0): row i lists workspace rows
/// [i + 1 - min(i + 1, 128), i] then -1 up to the 128-wide row; lengths min(i + 1, 128).
pub fn swa_prefill_indices(t: usize, window: usize) -> (Vec<i32>, Vec<i32>) {
    let width = window.div_ceil(128) * 128;
    let mut idx = vec![-1i32; t * width];
    let mut lens = vec![0i32; t];
    for i in 0..t {
        let len = (i + 1).min(window);
        for j in 0..len { idx[i * width + j] = (i + 1 - len + j) as i32; }
        lens[i] = len as i32;
    }
    (idx, lens)
}

/// SWA KV cache page: 256 tokens x (448 fp8 nope + 64 bf16 rope) then 256 x 8 scale bytes,
/// padded to a multiple of 576 (deepseek_v4::kv_page_bytes<V4>(256)).
pub const SWA_PAGE_TOKENS: usize = 256;
pub const SWA_PAGE_BYTES: usize = (584 * SWA_PAGE_TOKENS).div_ceil(576) * 576;

/// Rung: prefill layer-0 attention (SWA only) from the gold attention input to `self_attn.out`.
pub fn prefill_layer0_attention(ckpt_dir: &Path, bundle: &Path, gold_dir: &Path, tables_dir: &Path) -> Result<Vec<Comparison>> {
    let dev = Device::open(0)?;
    let s = dev.stream;
    let kernels = Kernels::open(bundle)?;
    let ckpt = Checkpoint::open(ckpt_dir)?;
    let mut w = DeviceWeights::new(&ckpt);
    let gold = Gold::open(gold_dir);
    let tables = Gold::open(tables_dir);
    let snap = SnapCheck::new("fwd-001|model.layers.0.self_attn")?;
    let l = "model.layers.0.self_attn";
    let x_host = gold.bytes(&format!("{l}.in.x"))?;
    let pos_host = gold.bytes(&format!("{l}.in.positions"))?;
    let (hidden, heads, head_dim, q_lora, rope) = (4096usize, 64usize, 512usize, 1024usize, 64usize);
    let t = x_host.len() / (hidden * 2);
    let eps = 1e-6f32;
    let sms = dev.sm_count as usize;
    let freqs = Buffer::from_host(&tables.bytes(&format!("{l}.freqs_cis"))?)?;
    let x = Buffer::from_host(&x_host)?;
    let positions = Buffer::from_host(&pos_host)?;
    let mut a = Arena::new(256 << 20)?;
    let mut out = Vec::new();

    // q / kv projections.
    let xq = a.alloc(t * hidden)?;
    let xs = a.alloc(sgl::act_scale_bytes(t, hidden))?;
    sgl::quant_fp8(&kernels, s, x.ptr, hidden, xq, xs, t, hidden)?;
    snap.out(s, "quant(x)", "per_token_group_quant", 0, 0, 3, xq, t * hidden)?;
    let qkv_n = q_lora + head_dim;
    let qkv_a = a.alloc(t * qkv_n * 2)?;
    deepgemm::fp8_gemm_nt(&kernels, s, xq, xs, w.get(&format!("{l}.wqkv_a.weight"))?, w.get(&format!("{l}.wqkv_a.weight_scale_inv"))?,
                          qkv_a, t, qkv_n, hidden, sms)?;
    snap.out(s, "wqkv_a", "fp8_fp4_gemm", 0, 9, 0, qkv_a, t * qkv_n * 2)?;
    let q_lora_n = a.alloc(t * q_lora * 2)?;
    sgl::rmsnorm_1024(&kernels, s, qkv_a, qkv_n, w.get(&format!("{l}.q_norm.weight"))?, q_lora_n, t, eps)?;
    snap.out(s, "q_norm", "RMSNorm", 0, 2, 0, q_lora_n, t * q_lora * 2)?;
    let lq = a.alloc(t * q_lora)?;
    let ls = a.alloc(sgl::act_scale_bytes(t, q_lora))?;
    sgl::quant_fp8(&kernels, s, q_lora_n, q_lora, lq, ls, t, q_lora)?;
    snap.out(s, "quant(q_lora)", "per_token_group_quant", 1, 0, 3, lq, t * q_lora)?;
    let qn = heads * head_dim;
    let q = a.alloc(t * qn * 2)?;
    deepgemm::fp8_gemm_nt(&kernels, s, lq, ls, w.get(&format!("{l}.wq_b.weight"))?, w.get(&format!("{l}.wq_b.weight_scale_inv"))?,
                          q, t, qn, q_lora, sms)?;
    snap.out(s, "wq_b", "fp8_fp4_gemm", 1, 9, 0, q, t * qn * 2)?;
    let q_rot = a.alloc(t * qn * 2)?;
    sgl::fused_q_norm_rope(&kernels, s, q, q_rot, freqs.ptr, positions.ptr, qn, qn, t, heads, eps)?;
    snap.out(s, "q_norm_rope", "fused_q_norm_rope", 0, 0, 1, q_rot, t * qn * 2)?;

    // KV: norm + rope + FP8 store into the SWA cache (request slots from page 1, as SGLang's
    // allocator hands out), then the dequantized window the sparse prefill reads.
    let kvcache = a.alloc(2 * SWA_PAGE_BYTES)?;
    crate::cuda::memset_async(kvcache, 0, 2 * SWA_PAGE_BYTES, s)?;
    let slots: Vec<i32> = (0..t as i32).map(|i| SWA_PAGE_TOKENS as i32 + i).collect();
    let out_loc = Buffer::from_host(&le_i32(&slots))?;
    sgl::fused_k_norm_rope_flashmla(&kernels, s, qkv_a + (q_lora * 2) as u64, w.get(&format!("{l}.kv_norm.weight"))?, freqs.ptr,
                                    positions.ptr, out_loc.ptr, kvcache, qkv_n, t, eps)?;
    if snap.snaps.is_some() {
        // Compare the request's page (page 1) of the cache.
        snap.out(s, "kv_store(page0+1)", "fused_k_norm_rope_flashmla", 0, 0, 5, kvcache, 2 * SWA_PAGE_BYTES)?;
    }
    let workspace = a.alloc(t * head_dim * 2)?;
    sgl::dequant_k_cache(&kernels, s, workspace, kvcache, out_loc.ptr, head_dim, t, "swa")?;
    snap.out(s, "dequant_k", "_dequantize_k_cache_paged_kernel", 0, 0, 0, workspace, t * head_dim * 2)?;

    // Sparse prefill over each token's causal SWA window.
    let (idx, lens) = swa_prefill_indices(t, 128);
    let indices = Buffer::from_host(&le_i32(&idx))?;
    let topk_length = Buffer::from_host(&le_i32(&lens))?;
    let o = a.alloc(t * qn * 2)?;
    let max_logits = a.alloc(t * heads * 4)?;
    let lse = a.alloc(t * heads * 4)?;
    let args = flashmla::PrefillArgs {
        s_q: t as i32, s_kv: t as i32, h_q: heads as i32, h_kv: 1, d_qk: head_dim as i32, d_v: head_dim as i32, topk: 128,
        sm_scale: f32::from_bits(0x3d3504f3), // head_dim ** -0.5
        q: q_rot, kv: workspace, indices: indices.ptr, attn_sink: w.get(&format!("{l}.attn_sink"))?, topk_length: topk_length.ptr,
        stride_q_s_q: qn as i32, stride_q_h_q: head_dim as i32, stride_kv_s_kv: head_dim as i32, stride_kv_h_kv: head_dim as i32,
        stride_indices_s_q: 128, stride_indices_h_kv: 128,
        out: o, max_logits, lse, num_sm: sms as i32, stats_in_log2: 1,
    };
    flashmla::sparse_prefill(&kernels, s, &args)?;
    snap.out(s, "flashmla_prefill", "sparse_attn_fwd_kernel", 0, 0, 13, o, t * qn * 2)?;
    sgl::inverse_rope_q(&kernels, s, o + ((head_dim - rope) * 2) as u64, freqs.ptr, positions.ptr, qn * 2, head_dim * 2, heads, t)?;
    snap.out(s, "inverse_rope", "deepseek_rope_kernel", 0, 0, 0, o, t * qn * 2)?;

    // Output projections: grouped wo_a (8 groups of 8 heads), then wo_b.
    let (groups, o_lora) = (8usize, 1024usize);
    let gd = qn / groups;
    let oq = a.alloc(t * qn)?;
    let os = a.alloc(groups * gd.div_ceil(512) * t.div_ceil(4) * 4 * 4)?;
    sgl::wo_a_group_quant(&kernels, s, o, oq, os, t, groups, gd, qn)?;
    snap.out(s, "wo_a_quant", "fp8_wo_a_group_major_quant", 0, 1, 0, oq, t * qn)?;
    let wo_a = a.alloc(t * groups * o_lora * 2)?;
    deepgemm::fp8_bmm_token_major(&kernels, s, oq, os, w.get(&format!("{l}.wo_a.weight"))?, w.get(&format!("{l}.wo_a.weight_scale_inv"))?,
                                  wo_a, groups, t, o_lora, gd, sms)?;
    snap.out(s, "wo_a", "fp8_fp4_gemm", 2, 9, 0, wo_a, t * groups * o_lora * 2)?;
    let bk = groups * o_lora;
    let bq = a.alloc(t * bk)?;
    let bs = a.alloc(sgl::act_scale_bytes(t, bk))?;
    sgl::quant_fp8(&kernels, s, wo_a, bk, bq, bs, t, bk)?;
    snap.out(s, "quant(wo_a)", "per_token_group_quant", 2, 0, 3, bq, t * bk)?;
    let y = a.alloc(t * hidden * 2)?;
    deepgemm::fp8_gemm_nt(&kernels, s, bq, bs, w.get(&format!("{l}.wo_b.weight"))?, w.get(&format!("{l}.wo_b.weight_scale_inv"))?,
                          y, t, hidden, bk, sms)?;
    s.synchronize()?;
    check(&gold, &format!("{l}.out"), y, t * hidden * 2, &mut out)?;
    Ok(out)
}

/// Rung: prefill layer-0 FFN half from the gold attention output: the attention's mHC post fused
/// with the FFN's mHC pre, the hash-routed NVFP4 MoE (trtllm-gen) and the FP8 shared expert, to the
/// layer output (routed + shared; the FFN's mHC post runs in the next layer's fused kernel).
pub fn prefill_layer0_ffn(ckpt_dir: &Path, bundle: &Path, gold_dir: &Path) -> Result<Vec<Comparison>> {
    let dev = Device::open(0)?;
    let s = dev.stream;
    let kernels: &'static Kernels = Box::leak(Box::new(Kernels::open(bundle)?));
    crate::runtime::install_redirect(kernels);
    let ckpt = Checkpoint::open(ckpt_dir)?;
    let mut w = DeviceWeights::new(&ckpt);
    let gold = Gold::open(gold_dir);
    let snap_layer = SnapCheck::new("fwd-001|model.layers.0")?;
    let snap_mlp = SnapCheck::new("fwd-001|model.layers.0.mlp")?;
    let (hidden, hc, mix) = (4096usize, 4usize, 24usize);
    let sms = dev.sm_count as usize;
    let residual_host = gold.bytes("model.layers.0.in.hidden_states")?;
    let t = residual_host.len() / (hc * hidden * 2);
    let residual = Buffer::from_host(&residual_host)?;
    let attn_out = Buffer::from_host(&gold.bytes("model.layers.0.self_attn.out")?)?;
    let input_ids = Buffer::from_host(&gold.bytes("model.layers.0.mlp.in.input_ids")?)?;
    let mut a = Arena::new(512 << 20)?;
    let mut out = Vec::new();

    // The attention's mHC pre (its post/comb mixes feed the fused post below).
    let splits = deepgemm::mhc_pre_splits(t, hc * hidden, sms);
    let d = a.alloc(splits * t * mix * 4)?;
    let sq = a.alloc(splits * t * 4)?;
    deepgemm::tf32_hc_prenorm_gemm(kernels, s, residual.ptr, w.get("model.layers.0.hc_attn_fn")?, d, sq, t, mix, hc * hidden, splits)?;
    let pre = mhc::MhcPreOut { post_mix: a.alloc(t * hc * 4)?, comb_mix: a.alloc(t * hc * hc * 4)?, layer_input: a.alloc(t * hidden * 2)? };
    mhc::pre_big_fuse_with_norm(kernels, s, d, sq, w.get("model.layers.0.hc_attn_scale")?, w.get("model.layers.0.hc_attn_base")?,
                                residual.ptr, &pre, w.get("model.layers.0.input_layernorm.weight")?, t, splits)?;

    // Attention mHC post fused with the FFN mHC pre.
    let (_, split_k) = mhc::fused_post_pre_shape(t);
    let mixes = a.alloc(split_k * t * mix * 4)?;
    let sqrsum = a.alloc(split_k * t * 4)?;
    let residual_cur = a.alloc(t * hc * hidden * 2)?;
    mhc::fused_post_pre_fma(kernels, s, pre.comb_mix, residual.ptr, pre.post_mix, attn_out.ptr, w.get("model.layers.0.hc_ffn_fn")?,
                            mixes, sqrsum, residual_cur, t)?;
    snap_layer.out(s, "fused_post_pre.residual", "mhc_fused_post_pre_fma", 0, 0, 0, residual_cur, t * hc * hidden * 2)?;
    let ffn = mhc::MhcPreOut { post_mix: a.alloc(t * hc * 4)?, comb_mix: a.alloc(t * hc * hc * 4)?, layer_input: a.alloc(t * hidden * 2)? };
    mhc::pre_big_fuse_with_norm(kernels, s, mixes, sqrsum, w.get("model.layers.0.hc_ffn_scale")?, w.get("model.layers.0.hc_ffn_base")?,
                                residual_cur, &ffn, w.get("model.layers.0.post_attention_layernorm.weight")?, t, split_k)?;
    s.synchronize()?;
    check(&gold, "model.layers.0.mlp.in.0", ffn.layer_input, t * hidden * 2, &mut out)?;
    let x = ffn.layer_input;

    // Hash routing.
    let m = "model.layers.0.mlp";
    let logits = a.alloc(t * 256 * 4)?;
    sgl::router_tiny_gemm(kernels, s, logits, x, w.get(&format!("{m}.gate.weight"))?, t, hidden)?;
    snap_mlp.out(s, "router_logits", "tiny_n_gemm", 0, 0, 0, logits, t * 256 * 4)?;
    let topk_ids = a.alloc(t * 6 * 4)?;
    let topk_w = a.alloc(t * 6 * 4)?;
    sgl::hash_topk(kernels, s, logits, input_ids.ptr, w.get(&format!("{m}.topk.tid2eid"))?, topk_ids, topk_w, t, 6, 256, 1.5)?;
    snap_mlp.out(s, "topk_ids", "moe_hash_topk", 0, 0, 3, topk_ids, t * 6 * 4)?;
    snap_mlp.out(s, "topk_weights", "moe_hash_topk", 0, 0, 4, topk_w, t * 6 * 4)?;
    let topk_ws = a.alloc(t * 6 * 4)?;
    torch_ops::mul_scalar_f32(kernels, s, topk_ws, topk_w, t * 6, 1.5)?;

    // Routed experts.
    let xq = a.alloc(t * hidden / 2)?;
    let xs = a.alloc(t * hidden / 16)?;
    sgl::nvfp4_quantize_4096(kernels, s, x, xq, xs, t, w.get(&format!("{m}.experts.w13_input_scale_quant"))?, sms)?;
    snap_mlp.out(s, "nvfp4(x)", "nvfp4_quantize", 0, 1, 0, xq, t * hidden / 2)?;
    snap_mlp.out(s, "nvfp4(x).scales", "nvfp4_quantize", 0, 2, 0, xs, t * hidden / 16)?;
    let experts = moe::Experts {
        w13: w.get(&format!("{m}.experts.w13_weight"))?, w13_scale: w.get(&format!("{m}.experts.w13_weight_scale"))?,
        w2: w.get(&format!("{m}.experts.w2_weight"))?, w2_scale: w.get(&format!("{m}.experts.w2_weight_scale"))?,
        clamp_limit: w.get(&format!("{m}.experts.gemm1_clamp_limit"))?, g1_scale_c: w.get(&format!("{m}.experts.g1_scale_c"))?,
        g1_alphas: w.get(&format!("{m}.experts.g1_alphas"))?, g2_alphas: w.get(&format!("{m}.experts.g2_alphas"))?,
    };
    let routed = a.alloc(t * hidden * 2)?;
    let scratch_bytes = 64 << 20;
    let scratch = a.alloc(scratch_bytes)?;
    let cubins = std::env::var("DSV4_TRTLLM_CUBINS").unwrap_or_else(|_| "/dsv4/kernels/trtllm_bmm".into());
    moe::fp4_routed(kernels, s, &cubins, &experts, topk_ids, topk_ws, xq, xs, routed, t, scratch, scratch_bytes)?;
    s.synchronize()?;
    check(&gold, &format!("{m}.out.0"), routed, t * hidden * 2, &mut out)?;

    // Shared expert (FP8 block-scaled, UE8M0).
    let sh = format!("{m}.shared_experts");
    let q1 = a.alloc(t * hidden)?;
    let s1 = a.alloc(sgl::act_scale_bytes(t, hidden))?;
    sgl::quant_fp8(kernels, s, x, hidden, q1, s1, t, hidden)?;
    let gate_up = a.alloc(t * 4096 * 2)?;
    deepgemm::fp8_gemm_nt(kernels, s, q1, s1, w.get(&format!("{sh}.gate_up_proj.weight"))?, w.get(&format!("{sh}.gate_up_proj.weight_scale_inv"))?,
                          gate_up, t, 4096, hidden, sms)?;
    let act = a.alloc(t * 2048 * 2)?;
    sgl::silu_mul_clamp(kernels, s, gate_up, act, t, 2048, 10.0)?;
    let q2 = a.alloc(t * 2048)?;
    let s2 = a.alloc(sgl::act_scale_bytes(t, 2048))?;
    sgl::quant_fp8(kernels, s, act, 2048, q2, s2, t, 2048)?;
    let shared = a.alloc(t * hidden * 2)?;
    deepgemm::fp8_gemm_nt(kernels, s, q2, s2, w.get(&format!("{sh}.down_proj.weight"))?, w.get(&format!("{sh}.down_proj.weight_scale_inv"))?,
                          shared, t, hidden, 2048, sms)?;
    s.synchronize()?;
    check(&gold, &format!("{m}.out.1"), shared, t * hidden * 2, &mut out)?;
    torch_ops::add_bf16(kernels, s, routed, routed, shared, t * hidden)?;
    s.synchronize()?;
    check(&gold, "model.layers.0.out", routed, t * hidden * 2, &mut out)?;
    Ok(out)
}

/// Rung: the native prefill driver (model.rs) from the token ids through the first `layers` layers,
/// checking every layer's output (routed + shared, before its deferred mHC post).
pub fn prefill_layers(ckpt_dir: &Path, bundle: &Path, gold_dir: &Path, tables_dir: &Path, layers: usize) -> Result<Vec<Comparison>> {
    use crate::model::{Batch, KvCache, Model, Tables, HIDDEN};
    let dev = Device::open(0)?;
    let kernels: &'static Kernels = Box::leak(Box::new(Kernels::open(bundle)?));
    crate::runtime::install_redirect(kernels);
    let ckpt = Checkpoint::open(ckpt_dir)?;
    let gold = Gold::open(gold_dir);
    let ids: Vec<i64> = gold.bytes("batch.input_ids")?.chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().unwrap())).collect();
    let t = ids.len();
    let dev_sms = dev.sm_count;
    let mut m = Model { dev, k: kernels, w: DeviceWeights::new(&ckpt), tables: Tables::load(tables_dir)?, cache: KvCache::new(4096)?,
                        trtllm_cubins: std::env::var("DSV4_TRTLLM_CUBINS").unwrap_or_else(|_| "/dsv4/kernels/trtllm_bmm".into()),
                        lt: crate::cublaslt::Lt::open(&std::env::var("DSV4_CUBLASLT").unwrap_or_else(|_| "/opt/deps/cublas/nvidia/cu13/lib/libcublasLt.so.13".into()), dev_sms as i32)?,
                        probe: None, probes: vec![], snaps: None, snap_label: None, c128_sched: None, decode_in: None, last_logits_only: false, prune_rows: vec![], hc_dest: None, reduce: None, blas_ws: None, streams: crate::model::Streams::default(), chunk: crate::model::ChunkMeta::default(), serving: None, verify_in: None, verify_ready: false, dl: crate::model::DecodeLayout::default(), dl_indexer: false, decode_w: 0, mtp_serving: None, rank: 0 };
    let s = m.dev.stream;
    let b = Batch::new(&ids, 0)?;
    let mut persist = Arena::new(256 << 20)?;
    let mut scratch = Arena::new(1 << 30)?;
    let mut out = Vec::new();
    m.chunk = m.chunk_meta(&b, &mut persist)?;
    let (mut hc, mut x) = m.embed(&b, &mut persist, &mut scratch)?;
    let mut layer_out = 0u64;
    for layer in 0..layers {
        scratch.reset();
        if layer > 0 { x = m.post_pre(layer, "attn", &mut hc, layer_out, t, &mut persist, &mut scratch)?; }
        let y = m.attention(layer, x, &b, &mut scratch)?;
        let xf = m.post_pre(layer, "ffn", &mut hc, y, t, &mut persist, &mut scratch)?;
        let routed = m.ffn(layer, xf, &b, &mut scratch)?;
        layer_out = persist.alloc(t * HIDDEN * 2)?;
        crate::cuda::copy_async(layer_out, routed, t * HIDDEN * 2, s)?;
        s.synchronize()?;
        let ok = check(&gold, &format!("model.layers.{layer}.out"), layer_out, t * HIDDEN * 2, &mut out)?;
        if !ok { return Ok(out); }
    }
    if layers == crate::model::LAYERS {
        scratch.reset();
        let (y, normed, logits) = m.head(&hc, layer_out, t, &mut scratch)?;
        s.synchronize()?;
        check(&gold, "model.norm.in.0", y, t * HIDDEN * 2, &mut out)?;
        check(&gold, "model.norm.out", normed, t * HIDDEN * 2, &mut out)?;
        // next_token_logits: the last row, widened to fp32 (exact).
        let v = crate::model::VOCAB;
        let last = download(logits + ((t - 1) * v * 2) as u64, v * 2)?;
        let f32s: Vec<u8> = last.chunks_exact(2).flat_map(|c| crate::dense_prepare::bf16_to_f32(u16::from_le_bytes([c[0], c[1]])).to_le_bytes()).collect();
        let c = compare(&gold, "logits_processor.out.next_token_logits", &f32s)?;
        println!("{}", serde_json::to_string(&c).unwrap());
        out.push(c);
        // Greedy sampling: torch.argmax, the first maximal index.
        let vals: Vec<f32> = f32s.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        let mut best = 0usize;
        for (i, &x) in vals.iter().enumerate() { if x > vals[best] { best = i; } }
        let c = compare(&gold, "sample.out", &(best as i64).to_le_bytes())?;
        println!("{}", serde_json::json!({"next_token": best}));
        println!("{}", serde_json::to_string(&c).unwrap());
        out.push(c);
    }
    Ok(out)
}

/// Rung for one TP rank of DeepSeek-V4-Pro (feature `pro`): the prefill of the reference rank's
/// captured forward `gold_dir` (`.../gold/rank<r>/fwd-<n>`), on this rank's package, through
/// `layers` layers. Every sum over the ranks takes the reference's result (Reduce::Gold), so the
/// rung checks this rank alone: each partial against the reference rank's all-reduce input, and
/// with all layers the logits of this rank's vocabulary shard. `DSV4_DRY=1` records the launch
/// sequence without launching (DSV4_TRACE).
pub fn pro_prefill(rank_dir: &Path, bundle: &Path, gold_dir: &Path, layers: usize) -> Result<Vec<Comparison>> {
    use crate::model::{Batch, Model, Reduce, Tables, HIDDEN, VOCAB_LOCAL};
    let dev = Device::open(0)?;
    let kernels: &'static Kernels = Box::leak(Box::new(Kernels::open(bundle)?));
    let gold = Gold::open(gold_dir);
    let ids: Vec<i64> = gold.bytes("batch.input_ids")?.chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().unwrap())).collect();
    let t = ids.len();
    let rank = serde_json::from_slice::<serde_json::Value>(&std::fs::read(rank_dir.join("params.json")).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?["rank"].as_u64().unwrap_or(0) as usize;
    let mut w = DeviceWeights::from_lazy(LazyPackage::open(rank_dir)?);
    let tables = Tables::single(w.get("model.layers.0.self_attn.freqs_cis")?);
    let mut m = Model::assemble(dev, kernels, w, tables, 4096)?;
    m.rank = rank;
    // Every reference rank's golds of this forward (.../gold/rank<r>/<fwd>).
    let all: Vec<Gold> = match (gold_dir.parent().and_then(|p| p.parent()), gold_dir.file_name()) {
        (Some(root), Some(fwd)) => (0..crate::model::TP).map(|r| Gold::open(root.join(format!("rank{r}")).join(fwd))).collect(),
        _ => vec![],
    };
    m.reduce = Some(Reduce::Gold { gold: Gold::open(gold_dir), all, next: 0, results: vec![] });
    let s = m.dev.stream;
    let b = Batch::new(&ids, 0)?;
    let mut persist = Arena::new(512 << 20)?;
    let mut scratch = Arena::new(2 << 30)?;
    let mut out = Vec::new();
    println!("{}", serde_json::json!({"pro_prefill": {"rank": rank, "tokens": t, "layers": layers}}));
    m.chunk = m.chunk_meta(&b, &mut persist)?;
    let (mut hc, mut x) = m.embed(&b, &mut persist, &mut scratch)?;
    let mut layer_out = 0u64;
    for layer in 0..layers {
        scratch.reset();
        if layer > 0 { x = m.post_pre(layer, "attn", &mut hc, layer_out, t, &mut persist, &mut scratch)?; }
        if !crate::runtime::dry() { s.synchronize()?; check(&gold, &format!("model.layers.{layer}.self_attn.in.x"), x, t * HIDDEN * 2, &mut out)?; }
        let y = m.attention(layer, x, &b, &mut scratch)?;
        m.reduce(y, t * HIDDEN * 2)?;
        let xf = m.post_pre(layer, "ffn", &mut hc, y, t, &mut persist, &mut scratch)?;
        if !crate::runtime::dry() { s.synchronize()?; check(&gold, &format!("model.layers.{layer}.mlp.in.0"), xf, t * HIDDEN * 2, &mut out)?; }
        m.probe = Some(layer);
        m.probes.clear();
        let routed = m.ffn(layer, xf, &b, &mut scratch)?;
        m.probe = None;
        if !crate::runtime::dry() {
            // The routed experts' and the shared expert's outputs, before they are added.
            s.synchronize()?;
            for (name, ptr, bytes) in std::mem::take(&mut m.probes) {
                if !crate::gold::exists(gold_dir, &name) { continue; }
                // Whichever reference rank held this package's slice of the layer.
                let native = crate::gold::sha(&download(ptr, bytes)?);
                let rank = (0..crate::model::TP).find(|r| gold_dir.parent().and_then(|p| p.parent()).zip(gold_dir.file_name())
                    .and_then(|(root, fwd)| Gold::open(root.join(format!("rank{r}")).join(fwd)).meta(&name).ok()).map(|m| m.sha256 == native).unwrap_or(false));
                println!("{}", serde_json::json!({"name": name, "equal": rank.is_some(), "reference_rank": rank}));
                out.push(Comparison { name: name.clone(), equal: rank.is_some(), native_sha256: native, gold_sha256: String::new(), first_diff_byte: None });
            }
        }
        m.reduce(routed, t * HIDDEN * 2)?;
        layer_out = persist.alloc(t * HIDDEN * 2)?;
        crate::cuda::copy_async(layer_out, routed, t * HIDDEN * 2, s)?;
        s.synchronize()?;
        println!("{}", serde_json::json!({"layer": layer, "done": true}));
    }
    if layers == crate::model::LAYERS {
        scratch.reset();
        let (_, _, logits) = m.head(&hc, layer_out, t, &mut scratch)?;
        s.synchronize()?;
        if !crate::runtime::dry() {
            // The reference's next-token logits (fp32 [1, vocab], the ranks' shards in rank order):
            // this rank's shard of the last row, widened (exact).
            let last = download(logits + ((t - 1) * VOCAB_LOCAL * 2) as u64, VOCAB_LOCAL * 2)?;
            let f32s: Vec<u8> = last.chunks_exact(2).flat_map(|c| crate::dense_prepare::bf16_to_f32(u16::from_le_bytes([c[0], c[1]])).to_le_bytes()).collect();
            let full = gold.bytes("output.next_token_logits").or_else(|_| gold.bytes("logits_processor.out.next_token_logits"))?;
            let rows = full.len() / (crate::model::VOCAB * 4);
            let at = (rows - 1) * crate::model::VOCAB * 4 + rank * VOCAB_LOCAL * 4;
            let want = &full[at..at + VOCAB_LOCAL * 4];
            let equal = want == &f32s[..];
            let vals: Vec<f32> = f32s.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
            let best = (0..vals.len()).fold(0, |b, i| if vals[i] > vals[b] { i } else { b });
            println!("{}", serde_json::json!({"logits_shard_equal": equal, "shard_argmax_token": rank * VOCAB_LOCAL + best, "shard_max": vals[best]}));
            out.push(Comparison { name: "logits shard".into(), equal, native_sha256: crate::gold::sha(&f32s), gold_sha256: crate::gold::sha(want), first_diff_byte: None });
        }
    }
    if let Some(Reduce::Gold { results, .. }) = m.reduce.take() { out.extend(results); }
    Ok(out)
}

/// One TP rank of DeepSeek-V4-Pro as its own process (feature `pro`), one per GPU: loads the
/// rank's package (the streaming loader, or tensor by tensor with DSV4_LAZY=1), then prefills the
/// prompt `ids` with the other rank processes: every sum over the ranks and the logits gather go
/// through `shm` (Reduce::Shm). Rank 0 prints the next token (greedy: the first maximum of the
/// gathered logits, the ranks' vocabulary shards in rank order). With `gold` (the reference's
/// captured forward of the same prompt), every sum and the logits are compared with the
/// reference's. Prints one JSON line with the timeline (seconds since process start).
pub fn pro_rank(rank_dir: &Path, bundle: &Path, ids: &[i64], shm: &Path, gold_dir: Option<&Path>, t0: std::time::Instant) -> Result<()> {
    use crate::model::{Batch, Model, Reduce, Tables, HIDDEN, LAYERS, TP, VOCAB, VOCAB_LOCAL};
    let rank = serde_json::from_slice::<serde_json::Value>(&std::fs::read(rank_dir.join("params.json")).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?["rank"].as_u64().unwrap_or(0) as usize;
    let fail = |e: String| -> String { let _ = std::fs::write(shm.join("abort"), format!("rank {rank}: {e}")); e };
    let mut timeline = serde_json::Map::new();
    let mut mark = |name: &str| { timeline.insert(name.to_string(), serde_json::json!((t0.elapsed().as_secs_f64() * 1000.0).round() / 1000.0)); };
    let lazy = std::env::var("DSV4_LAZY").as_deref() == Ok("1");
    // Readers before CUDA; the CUDA context on its own thread (in turn with the other ranks:
    // DSV4_CTX_LOCK), the kernel bundle's manifest meanwhile.
    let load = if lazy { None } else {
        // Measured with eight ranks on one warm host (2026-11-11): 32 readers per rank reach the
        // first token in 26.5-26.6 s, 16 in 27.0-27.3 s; 64 MiB chunks with 16 pinned slots 29.1-33.6 s.
        for (key, value) in [("NATIVE_LOAD_FILE_IO", "direct"), ("NATIVE_LOAD_WORKERS", "32"), ("NATIVE_LOAD_CHUNK_MIB", "16")] {
            if std::env::var_os(key).is_none() { std::env::set_var(key, value); }
        }
        let sha = std::env::var("DSV4_MANIFEST_SHA256").map_err(|_| "a rank package needs DSV4_MANIFEST_SHA256 (its manifest's pin)")?;
        Some(crate::prepared::start(rank_dir, &sha).map_err(&fail)?)
    };
    mark("readers_started");
    let device = std::thread::spawn(|| -> Result<Device> {
        let lock = match std::env::var("DSV4_CTX_LOCK") {
            Ok(path) if !path.is_empty() => {
                let f = std::fs::OpenOptions::new().create(true).append(true).open(&path).map_err(|e| format!("{path}: {e}"))?;
                f.lock().map_err(|e| format!("{path}: {e}"))?;
                Some(f)
            }
            _ => None,
        };
        let dev = Device::open(0)?;
        drop(lock);
        Ok(dev)
    });
    let kernels: &'static Kernels = Box::leak(Box::new(Kernels::open(bundle).map_err(&fail)?));
    let dev = device.join().map_err(|_| "CUDA context thread panicked".to_string()).and_then(|d| d).map_err(&fail)?;
    crate::cuda::set_current(dev.ctx)?;
    mark("cuda_context");
    // While the weights stream in (uploads on their own thread and stream): the bundle's kernel
    // modules, cuBLASLt with its GEMM shapes warmed for prefills of up to 32 tokens, and the
    // model's caches and arenas.
    let (ctx, sms) = (dev.ctx as usize, dev.sm_count as i32);
    let uploads = load.map(|load| std::thread::spawn(move || -> Result<crate::prepared::Weights> {
        crate::cuda::set_current(ctx as crate::cuda::CUcontext)?;
        let stream = crate::cuda::Stream::new()?;
        let w = load.finish(stream, ctx as crate::cuda::CUcontext)?;
        stream.synchronize()?;
        Ok(w)
    }));
    let modules = std::thread::spawn(move || -> Result<usize> {
        crate::cuda::set_current(ctx as crate::cuda::CUcontext)?;
        kernels.load_all(kernels.read_all()?)
    });
    let lt_path = std::env::var("DSV4_CUBLASLT").unwrap_or_else(|_| "/opt/deps/cublas/nvidia/cu13/lib/libcublasLt.so.13".into());
    let warm = std::thread::spawn(move || -> Result<crate::cublaslt::Lt> {
        crate::cuda::set_current(ctx as crate::cuda::CUcontext)?;
        let lt = crate::cublaslt::Lt::init(&lt_path, sms)?;
        let ms: Vec<usize> = (1..=32).collect();
        // Compressor wkv_gate (C4, C128, the indexer's; fp32 out) and this rank's lm_head shard (bf16 out).
        for (n, f) in [(4 * crate::model::HEAD_DIM, true), (2 * crate::model::HEAD_DIM, true), (4 * crate::model::INDEX_DIM, true), (VOCAB_LOCAL, false)] {
            lt.warm(n, HIDDEN, f, &ms)?;
        }
        Ok(lt)
    });
    let mut m = Model::assemble(dev, kernels, DeviceWeights::empty(), Tables::single(0), 4096).map_err(&fail)?;
    mark("model_assembled");
    m.lt = warm.join().map_err(|_| "cuBLASLt warm-up panicked".to_string()).and_then(|r| r).map_err(&fail)?;
    mark("cublaslt_warm");
    modules.join().map_err(|_| "module loader panicked".to_string()).and_then(|r| r).map_err(&fail)?;
    mark("kernel_modules");
    let mut w = match uploads {
        Some(h) => DeviceWeights::from_prepared(h.join().map_err(|_| "weight loader panicked".to_string()).and_then(|r| r).map_err(&fail)?),
        None => DeviceWeights::from_lazy(LazyPackage::open(rank_dir).map_err(&fail)?),
    };
    mark("weights_on_device");
    m.tables = Tables::single(w.get("model.layers.0.self_attn.freqs_cis").map_err(&fail)?);
    m.w = w;
    m.rank = rank;
    m.reduce = Some(Reduce::Shm { dir: shm.to_path_buf(), rank, ranks: TP, next: 0, gold: gold_dir.map(Gold::open), differ: vec![], seconds: 0.0 });
    let s = m.dev.stream;
    let t = ids.len();
    let b = Batch::new(ids, 0)?;
    let mut persist = Arena::new(512 << 20)?;
    let mut scratch = Arena::new(2 << 30)?;
    mark("model_ready");
    // Where the forward's time goes (seconds since process start, device synchronized at each mark).
    let stages = std::cell::RefCell::new(Vec::<(String, f64)>::new());
    let stage = |name: String| -> Result<()> { s.synchronize()?; stages.borrow_mut().push((name, (t0.elapsed().as_secs_f64() * 1000.0).round() / 1000.0)); Ok(()) };
    let run = |m: &mut Model, persist: &mut Arena, scratch: &mut Arena| -> Result<u64> {
        m.chunk = m.chunk_meta(&b, persist)?;
        let (mut hc, mut x) = m.embed(&b, persist, scratch)?;
        stage("embed".into())?;
        let mut layer_out = 0u64;
        for layer in 0..LAYERS {
            scratch.reset();
            if layer > 0 { x = m.post_pre(layer, "attn", &mut hc, layer_out, t, persist, scratch)?; }
            let y = m.attention(layer, x, &b, scratch)?;
            if layer < 4 { stage(format!("l{layer}.attention"))?; }
            m.reduce(y, t * HIDDEN * 2)?;
            let xf = m.post_pre(layer, "ffn", &mut hc, y, t, persist, scratch)?;
            layer_out = hc.layer_out();
            m.ffn_into(layer, xf, &b, scratch, layer_out)?;
            if layer < 4 { stage(format!("l{layer}.ffn"))?; }
            m.reduce(layer_out, t * HIDDEN * 2)?;
            if layer < 4 || layer % 20 == 0 { stage(format!("l{layer}.done"))?; }
        }
        scratch.reset();
        let (_, _, logits) = m.head(&hc, layer_out, t, scratch)?;
        s.synchronize()?;
        Ok(logits)
    };
    let logits = run(&mut m, &mut persist, &mut scratch).map_err(&fail)?;
    mark("forward_done");
    // The last position's logits: this rank's shard; gathered in rank order.
    let shard = download(logits + ((t - 1) * VOCAB_LOCAL * 2) as u64, VOCAB_LOCAL * 2)?;
    let full = crate::model::shm_exchange(shm, "logits", rank, TP, &shard, |parts| parts.concat()).map_err(&fail)?;
    mark("first_token");
    let (differ, reduce_s) = match &m.reduce { Some(Reduce::Shm { differ, seconds, .. }) => (differ.clone(), *seconds), _ => (vec![], 0.0) };
    let mut report = serde_json::json!({"rank": rank, "tokens": t, "timeline_s": timeline, "reduce_s": reduce_s, "stages_s": stages.into_inner(),
                                        "sums_differing_from_reference": differ.len(), "first_differing_sum": differ.first()});
    if rank == 0 {
        let token = argmax_bf16(&full[..VOCAB * 2]);
        report["next_token"] = serde_json::json!(token);
        if let Some(g) = gold_dir.map(Gold::open) {
            // The reference's fp32 next-token logits [rows, vocab]: its last row against the widened bf16 row.
            let f32s: Vec<u8> = full[..VOCAB * 2].chunks_exact(2).flat_map(|c| crate::dense_prepare::bf16_to_f32(u16::from_le_bytes([c[0], c[1]])).to_le_bytes()).collect();
            if let Ok(want) = g.bytes("logits_processor.out.next_token_logits").or_else(|_| g.bytes("output.next_token_logits")) {
                let last = &want[want.len() - VOCAB * 4..];
                report["logits_equal_reference"] = serde_json::json!(last == &f32s[..]);
                let vals: Vec<f32> = last.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
                report["reference_next_token"] = serde_json::json!((0..vals.len()).fold(0, |b, i| if vals[i] > vals[b] { i } else { b }));
                let max_diff = vals.iter().zip(f32s.chunks_exact(4)).map(|(a, c)| (a - f32::from_le_bytes([c[0], c[1], c[2], c[3]])).abs()).fold(0f32, f32::max);
                report["logits_max_abs_diff"] = serde_json::json!(max_diff);
            }
        }
    }
    println!("{}", serde_json::json!({"pro_rank": report, "epoch_first_token": std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs_f64()).unwrap_or(0.0)}));
    Ok(())
}

/// Greedy next token from bf16 logits [vocab] (torch.argmax over the fp32 widening: the first maximum).
/// Greedy sampling of a bf16 logits row: the first index of the maximum (torch.argmax). Compares
/// order-preserving integer keys of the bf16 bit patterns (no float conversion; NaN-free logits).
pub fn argmax_bf16(row: &[u8]) -> usize {
    // -0 compares equal to +0.
    let key = |b: u16| -> u16 { let b = if b == 0x8000 { 0 } else { b }; if b & 0x8000 != 0 { !b } else { b | 0x8000 } };
    let mut best = 0usize;
    let mut best_k = 0u16;
    for (i, c) in row.chunks_exact(2).enumerate() {
        let k = key(u16::from_le_bytes([c[0], c[1]]));
        if k > best_k || i == 0 { best_k = k; best = i; }
    }
    best
}

/// Rung: generate `steps` tokens greedily for gold prompt `prompt` (its prefill forward is
/// fwd-(128 * prompt + 1) of the census golds), checking each forward's next-token logits and
/// sampled token, and on the first mismatch every layer output of that forward.
pub fn generate(ckpt_dir: &Path, bundle: &Path, gold_root: &Path, tables_dir: &Path, prompts: &[usize], steps: usize) -> Result<Vec<Comparison>> {
    use crate::model::{Batch, KvCache, Model, Tables, HIDDEN, VOCAB};
    let dev = Device::open(0)?;
    let kernels: &'static Kernels = Box::leak(Box::new(Kernels::open(bundle)?));
    crate::runtime::install_redirect(kernels);
    let ckpt = Checkpoint::open(ckpt_dir)?;
    let sms = dev.sm_count as i32;
    let mut m = Model { dev, k: kernels, w: DeviceWeights::new(&ckpt), tables: Tables::load(tables_dir)?, cache: KvCache::new(4096)?,
                        trtllm_cubins: std::env::var("DSV4_TRTLLM_CUBINS").unwrap_or_else(|_| "/dsv4/kernels/trtllm_bmm".into()),
                        lt: crate::cublaslt::Lt::open(&std::env::var("DSV4_CUBLASLT").unwrap_or_else(|_| "/opt/deps/cublas/nvidia/cu13/lib/libcublasLt.so.13".into()), sms)?,
                        probe: std::env::var("DSV4_PROBE_LAYER").ok().and_then(|v| v.parse().ok()), probes: vec![],
                        snaps: match std::env::var("DSV4_SNAPSHOTS") { Ok(d) if !d.is_empty() => Some(crate::snapshots::Snapshots::open(Path::new(&d))?), _ => None },
                        snap_label: None, c128_sched: None, decode_in: None, last_logits_only: false, prune_rows: vec![], hc_dest: None, reduce: None, blas_ws: None, streams: crate::model::Streams::default(), chunk: crate::model::ChunkMeta::default(), serving: None, verify_in: None, verify_ready: false, dl: crate::model::DecodeLayout::default(), dl_indexer: false, decode_w: 0, mtp_serving: None, rank: 0 };
    let probe_step: Option<usize> = std::env::var("DSV4_PROBE_STEP").ok().and_then(|v| v.parse().ok());
    let s = m.dev.stream;
    let gold_root = std::env::var("DSV4_GOLD_ROOT").map(std::path::PathBuf::from).unwrap_or_else(|_| gold_root.to_path_buf());
    // The reference server's /generate responses (tokens and logprobs, identical across its runs).
    let outputs_path = std::env::var("DSV4_OUTPUTS").unwrap_or_else(|_| "/dsv4/runs/20261108-101648-gold/outputs-a.json".into());
    let outputs: serde_json::Value = serde_json::from_slice(&std::fs::read(&outputs_path).map_err(|e| format!("{outputs_path}: {e}"))?)
        .map_err(|e| e.to_string())?;
    let tok = crate::tokenize::Tokenizer::open(ckpt_dir)?;
    let mut out = Vec::new();
    for &prompt in prompts {
        let want = &outputs[prompt];
        let mut lp = LogprobTally::default();
        m.cache = KvCache::new(4096)?;
        let fwd = |i: usize| Gold::open(gold_root.join(format!("fwd-{:03}", 128 * prompt + 1 + i)));
        // The prompt text, tokenized natively; the reference's prefill must have run the same ids.
        let ids: Vec<i64> = tok.encode(want["prompt"].as_str().ok_or("reference output has no prompt")?)?.iter().map(|&i| i as i64).collect();
        let gold_ids: Vec<i64> = fwd(0).bytes("batch.input_ids")?.chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().unwrap())).collect();
        if ids != gold_ids { return Err(format!("prompt {prompt}: native ids {ids:?} != reference {gold_ids:?}")); }
        let mut persist = Arena::new(1 << 30)?;
        let mut scratch = Arena::new(1 << 30)?;
        let mut tokens = Vec::new();
        let mut pos = 0usize;
        let mut next: Vec<i64> = ids.clone();
        for step in 0..=steps {
            let gold = fwd(step);
            let b = if step == 0 { Batch::new(&next, 0)? } else { m.decode_batch(next[0], pos)? };
            if step > 0 {
                let want: Vec<i64> = gold.bytes("batch.input_ids")?.chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().unwrap())).collect();
                if want != next { return Err(format!("step {step}: gold input {want:?} != native {next:?}")); }
            }
            persist.reset();
            m.snap_label = if probe_step == Some(step) { Some(format!("fwd-{:03}", 128 * prompt + 1 + step)) } else { None };
            let mut layer_outs = Vec::new();
            let t0 = std::time::Instant::now();
            let logits = m.forward(&b, &mut persist, &mut scratch, Some(&mut layer_outs))?;
            s.synchronize()?;
            let ms = t0.elapsed().as_millis();
            let last = download(logits + ((b.t - 1) * VOCAB * 2) as u64, VOCAB * 2)?;
            let f32s: Vec<u8> = last.chunks_exact(2).flat_map(|c| crate::dense_prepare::bf16_to_f32(u16::from_le_bytes([c[0], c[1]])).to_le_bytes()).collect();
            let c = compare(&gold, "logits_processor.out.next_token_logits", &f32s)?;
            let token = argmax_bf16(&last);
            let cs = compare(&gold, "sample.out", &(token as i64).to_le_bytes())?;
            scratch.reset();
            let got = m.logprobs(logits, b.t, if step == 0 { Some(&ids) } else { None }, token, &mut scratch)?;
            let lp_ok = lp.check(want, step, token, &got, if step == 0 { Some(&ids) } else { None });
            println!("{}", serde_json::json!({"step": step, "pos": pos, "token": token, "logits_equal": c.equal, "sample_equal": cs.equal,
                                              "logprobs_equal": lp_ok, "ms": ms}));
            let ok = c.equal && cs.equal;
            out.push(c);
            out.push(cs);
            if !ok {
                for (name, ptr, bytes) in m.probes.clone() {
                    if crate::gold::exists(&gold.root, &name) {
                        println!("{}", serde_json::to_string(&compare(&gold, &name, &download(ptr, bytes)?)?).unwrap());
                    } else {
                        println!("{}", serde_json::json!({"probe": name, "note": "no gold"}));
                    }
                }
                for (layer, &ptr) in layer_outs.iter().enumerate() {
                    let name = format!("model.layers.{layer}.out");
                    if crate::gold::exists(&gold.root, &name) {
                        let c = compare(&gold, &name, &download(ptr, b.t * HIDDEN * 2)?)?;
                        let eq = c.equal;
                        println!("{}", serde_json::to_string(&c).unwrap());
                        if !eq { break; }
                    }
                }
                break;
            }
            tokens.push(token);
            pos += b.t;
            next = vec![token as i64];
        }
        let want_tokens: Vec<i64> = want["output_token_logprobs"].as_array().map(|a| a.iter().map(|e| e[1].as_i64().unwrap_or(-1)).collect()).unwrap_or_default();
        let tokens_equal = tokens.iter().map(|&t| t as i64).collect::<Vec<_>>() == want_tokens;
        let text = tok.decode(&tokens.iter().map(|&t| t as u32).collect::<Vec<_>>())?;
        let text_equal = text == want["text"].as_str().unwrap_or("");
        println!("{}", serde_json::json!({"prompt": prompt, "tokens": tokens, "tokens_equal_reference_output": tokens_equal,
                                          "text_equal_reference_output": text_equal, "logprobs": lp.summary()}));
    }
    Ok(out)
}

/// Native logprobs against one reference response (`/generate` meta_info): entries are
/// [logprob, token_id, text]; values are fp32 widened to Python floats, so they compare as fp32 bits.
#[derive(Default)]
struct LogprobTally { equal: [usize; 3], total: [usize; 3], first_diff: Vec<serde_json::Value> }

impl LogprobTally {
    fn same(v: &serde_json::Value, x: f32) -> bool { v.as_f64().map(|f| (f as f32).to_bits() == x.to_bits()).unwrap_or(false) }

    fn note(&mut self, kind: usize, ok: bool, detail: serde_json::Value) -> bool {
        self.total[kind] += 1;
        if ok { self.equal[kind] += 1; } else if self.first_diff.len() < 8 { self.first_diff.push(detail); }
        ok
    }

    /// Check one forward: its sampled token's logprob and top-5, and (prefill) the input tokens'.
    fn check(&mut self, want: &serde_json::Value, step: usize, token: usize, got: &crate::logprobs::Logprobs, input_ids: Option<&[i64]>) -> bool {
        let mut all = true;
        let e = &want["output_token_logprobs"][step];
        let ok = e[1].as_i64() == Some(token as i64) && Self::same(&e[0], got.token);
        all &= self.note(0, ok, serde_json::json!({"step": step, "kind": "token", "want": e, "got": [got.token, token]}));
        let top = &want["output_top_logprobs"][step];
        let ok = top.as_array().map(|t| t.len() == got.top.len() && t.iter().zip(&got.top)
            .all(|(w, g)| w[1].as_i64() == Some(g.1 as i64) && Self::same(&w[0], g.0))).unwrap_or(false);
        all &= self.note(1, ok, serde_json::json!({"step": step, "kind": "top", "want": top, "got": got.top}));
        if let Some(ids) = input_ids {
            let w = &want["input_token_logprobs"];
            let n = w.as_array().map(|a| a.len()).unwrap_or(0);
            let ok0 = n == ids.len() && w[0][0].is_null();
            all &= self.note(2, ok0, serde_json::json!({"kind": "input", "index": 0, "want": w[0], "len": n}));
            for i in 1..ids.len().min(n) {
                let ok = w[i][1].as_i64() == Some(ids[i]) && Self::same(&w[i][0], got.input[i - 1]);
                all &= self.note(2, ok, serde_json::json!({"kind": "input", "index": i, "want": w[i], "got": [got.input[i - 1], ids[i]]}));
            }
        }
        all
    }

    fn summary(&self) -> serde_json::Value {
        serde_json::json!({"output_token": [self.equal[0], self.total[0]], "output_top5": [self.equal[1], self.total[1]],
                           "input_token": [self.equal[2], self.total[2]], "first_diffs": self.first_diff})
    }
}

/// Rung: run the prefill of gold prompt `prompt` `repeats` times (fresh KV cache each time) and report
/// the first layer whose output differs between repeats (native determinism).
pub fn determinism(ckpt_dir: &Path, bundle: &Path, gold_root: &Path, tables_dir: &Path, prompt: usize, repeats: usize) -> Result<Vec<Comparison>> {
    use crate::model::{Batch, KvCache, Model, Tables, HIDDEN};
    let dev = Device::open(0)?;
    let kernels: &'static Kernels = Box::leak(Box::new(Kernels::open(bundle)?));
    crate::runtime::install_redirect(kernels);
    let ckpt = Checkpoint::open(ckpt_dir)?;
    let sms = dev.sm_count as i32;
    let mut m = Model { dev, k: kernels, w: DeviceWeights::new(&ckpt), tables: Tables::load(tables_dir)?, cache: KvCache::new(4096)?,
                        trtllm_cubins: std::env::var("DSV4_TRTLLM_CUBINS").unwrap_or_else(|_| "/dsv4/kernels/trtllm_bmm".into()),
                        lt: crate::cublaslt::Lt::open(&std::env::var("DSV4_CUBLASLT").unwrap_or_else(|_| "/opt/deps/cublas/nvidia/cu13/lib/libcublasLt.so.13".into()), sms)?,
                        probe: None, probes: vec![], snaps: None, snap_label: None, c128_sched: None, decode_in: None, last_logits_only: false, prune_rows: vec![], hc_dest: None, reduce: None, blas_ws: None, streams: crate::model::Streams::default(), chunk: crate::model::ChunkMeta::default(), serving: None, verify_in: None, verify_ready: false, dl: crate::model::DecodeLayout::default(), dl_indexer: false, decode_w: 0, mtp_serving: None, rank: 0 };
    let gold = Gold::open(gold_root.join(format!("fwd-{:03}", 128 * prompt + 1)));
    let ids: Vec<i64> = gold.bytes("batch.input_ids")?.chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().unwrap())).collect();
    let t = ids.len();
    let mut runs: Vec<Vec<String>> = Vec::new();
    let mut out = Vec::new();
    for r in 0..repeats {
        m.cache = KvCache::new(4096)?;
        let mut persist = Arena::new(1 << 30)?;
        let mut scratch = Arena::new(1 << 30)?;
        let b = Batch::new(&ids, 0)?;
        let mut outs = Vec::new();
        m.forward(&b, &mut persist, &mut scratch, Some(&mut outs))?;
        m.dev.stream.synchronize()?;
        let shas: Vec<String> = outs.iter().map(|&p| download(p, t * HIDDEN * 2).map(|d| crate::gold::sha(&d))).collect::<Result<_>>()?;
        let mut first_bad = None;
        for (l, sha) in shas.iter().enumerate() {
            let name = format!("model.layers.{l}.out");
            let c = compare(&gold, &name, &download(outs[l], t * HIDDEN * 2)?)?;
            if !c.equal && first_bad.is_none() { first_bad = Some(l); }
            let _ = sha;
            if l + 1 == shas.len() { out.push(c); }
        }
        let first_diff_prev = runs.last().and_then(|p| p.iter().zip(&shas).position(|(a, b)| a != b));
        println!("{}", serde_json::json!({"repeat": r, "first_layer_off_gold": first_bad, "first_layer_differs_from_previous": first_diff_prev}));
        runs.push(shas);
    }
    Ok(out)
}

/// Rung: long-context prefill of prompt `prompt` from a long reference capture (sglang_b300.py
/// --long-gold-): every 16384-token chunk layer by layer against the capture's per-forward golds
/// (layer outputs, the final chunk's next-token logits and sample), stopping at the first mismatch.
/// The model loads from a prepared package (DSV4_MANIFEST_SHA256 pins its manifest).
pub fn long_prefill(prepared: &Path, bundle: &Path, run: &Path, prompt: usize, max_layers: usize) -> Result<Vec<Comparison>> {
    use crate::model::{Batch, KvCache, Model, HIDDEN, LAYERS, VOCAB};
    const CHUNK: usize = 16384;
    const DECODE: usize = 16;
    let sha = std::env::var("DSV4_MANIFEST_SHA256").map_err(|_| "DSV4_MANIFEST_SHA256 pins the prepared package")?;
    let t0 = std::time::Instant::now();
    let mut timeline = Vec::new();
    let mut m = Model::start_prepared(prepared, &sha, bundle)?.finish(t0, &mut timeline)?;
    m.last_logits_only = true;
    m.enable_streams()?;
    let prompts: Vec<Vec<i64>> = serde_json::from_slice(&std::fs::read(run.join("prompts.json")).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    let ids = &prompts[prompt];
    let base = 1 + prompts[..prompt].iter().map(|p| p.len().div_ceil(CHUNK) + DECODE).sum::<usize>();
    m.cache = KvCache::new(ids.len() + 64)?;
    let s = m.dev.stream;
    let mut persist = Arena::new(24usize << 30)?;
    let mut scratch = Arena::new(24usize << 30)?;
    let mut out = Vec::new();
    let chunks = ids.len().div_ceil(CHUNK);
    for c in 0..chunks {
        let start = c * CHUNK;
        let chunk = &ids[start..(start + CHUNK).min(ids.len())];
        let t = chunk.len();
        let gold = Gold::open(run.join("gold").join(format!("fwd-{:03}", base + c)));
        let want: Vec<i64> = gold.bytes("batch.input_ids")?.chunks_exact(8).map(|v| i64::from_le_bytes(v.try_into().unwrap())).collect();
        if want != chunk { return Err(format!("chunk {c}: the gold forward's input ids differ from the prompt's chunk")); }
        let b = Batch::new(chunk, start)?;
        persist.reset();
        scratch.reset();
        m.c128_sched = Some((persist.alloc(148 * 32)?, persist.alloc(64)?, false));
        let tc = std::time::Instant::now();
        m.chunk = m.chunk_meta(&b, &mut persist)?;
        let (mut hc, mut x) = m.embed(&b, &mut persist, &mut scratch)?;
        let mut layer_out = 0u64;
        let layers = max_layers.min(LAYERS);
        for layer in 0..layers {
            scratch.reset();
            if layer > 0 { x = m.post_pre(layer, "attn", &mut hc, layer_out, t, &mut persist, &mut scratch)?; }
            let y = m.attention(layer, x, &b, &mut scratch)?;
            if crate::gold::exists(&gold.root, &format!("model.layers.{layer}.self_attn.out")) {
                s.synchronize()?;
                let ok = check(&gold, &format!("model.layers.{layer}.self_attn.out"), y, t * HIDDEN * 2, &mut out)?;
                if !ok { println!("{}", serde_json::json!({"chunk": c, "first_mismatch": format!("layer {layer} self_attn")})); return Ok(out); }
            }
            let xf = m.post_pre(layer, "ffn", &mut hc, y, t, &mut persist, &mut scratch)?;
            let routed = m.ffn(layer, xf, &b, &mut scratch)?;
            layer_out = hc.layer_out();
            crate::cuda::copy_async(layer_out, routed, t * HIDDEN * 2, s)?;
            s.synchronize()?;
            let ok = check(&gold, &format!("model.layers.{layer}.out"), layer_out, t * HIDDEN * 2, &mut out)?;
            if !ok { println!("{}", serde_json::json!({"chunk": c, "first_mismatch": format!("layer {layer} out")})); return Ok(out); }
        }
        println!("{}", serde_json::json!({"chunk": c, "tokens": t, "layers_equal": layers, "ms": tc.elapsed().as_millis()}));
        if layers == LAYERS && c + 1 == chunks {
            scratch.reset();
            let (_, _, logits) = m.head(&hc, layer_out, t, &mut scratch)?;
            s.synchronize()?;
            let last = download(logits + ((t - 1) * VOCAB * 2) as u64, VOCAB * 2)?;
            let f32s: Vec<u8> = last.chunks_exact(2).flat_map(|v| crate::dense_prepare::bf16_to_f32(u16::from_le_bytes([v[0], v[1]])).to_le_bytes()).collect();
            let cl = compare(&gold, "logits_processor.out.next_token_logits", &f32s)?;
            println!("{}", serde_json::to_string(&cl).unwrap());
            out.push(cl);
            let token = argmax_bf16(&last);
            let cs = compare(&gold, "sample.out", &(token as i64).to_le_bytes())?;
            println!("{}", serde_json::json!({"next_token": token, "sample_equal": cs.equal}));
            out.push(cs);
            // Decode steps (gold forwards after the prompt's chunks), eager or with the decode graph
            // regime logic of `generate` left aside: each step compares its logits and sample.
            let steps = std::env::var("DSV4_LONG_DECODE").ok().and_then(|v| v.parse().ok()).unwrap_or(DECODE);
            let mut token = token;
            for j in 0..steps {
                let pos = ids.len() + j;
                let gold = Gold::open(run.join("gold").join(format!("fwd-{:03}", base + chunks + j)));
                if crate::gold::exists(&gold.root, "batch.input_ids") {
                    let want: Vec<i64> = gold.bytes("batch.input_ids")?.chunks_exact(8).map(|v| i64::from_le_bytes(v.try_into().unwrap())).collect();
                    if want != [token as i64] { return Err(format!("decode step {j}: gold input {want:?}, native token {token}")); }
                }
                let b = m.decode_batch(token as i64, pos)?;
                persist.reset();
                scratch.reset();
                let td = std::time::Instant::now();
                let logits = m.forward(&b, &mut persist, &mut scratch, None)?;
                s.synchronize()?;
                let ms = td.elapsed().as_secs_f64() * 1e3;
                let last = download(logits, VOCAB * 2)?;
                let f32s: Vec<u8> = last.chunks_exact(2).flat_map(|v| crate::dense_prepare::bf16_to_f32(u16::from_le_bytes([v[0], v[1]])).to_le_bytes()).collect();
                let cl = compare(&gold, "logits_processor.out.next_token_logits", &f32s)?;
                token = argmax_bf16(&last);
                let cs = compare(&gold, "sample.out", &(token as i64).to_le_bytes())?;
                println!("{}", serde_json::json!({"decode_step": j, "position": pos, "logits_equal": cl.equal, "next_token": token, "sample_equal": cs.equal, "ms": ms}));
                let stop = !cl.equal || !cs.equal;
                out.push(cl);
                out.push(cs);
                if stop { break; }
            }
        }
    }
    Ok(out)
}
