//! The native DeepSeek-V4-Flash forward: SGLang's launch sequence (dsv4 attention backend,
//! flashinfer_trtllm_routed MoE, decode CUDA graph off), one reference CUBIN per launch.
//!
//! Per layer: [mHC pre] attention [mHC post fused with the FFN's mHC pre] MoE + shared expert; the
//! FFN's mHC post is fused into the next layer's first kernel (or the head's mhc_post).
use crate::cuda::{Buffer, Device, Stream};
use crate::gold::Gold;
use crate::parity::DeviceWeights;
use crate::runtime::{Arena, Kernels};
use crate::{compress, deepgemm, flashmla, mhc, moe, sgl, torch_ops, Result};
use crate::cublaslt::Lt;

/// Model dimensions as one process sees them. Default: DeepSeek-V4-Flash on one GPU. Feature
/// `pro`: one rank of DeepSeek-V4-Pro at tensor parallelism 8 (SGLang `--tp 8`): 16 of the 128
/// heads, 2 of the 16 wo_a groups, an eighth of each expert's intermediate dimension and of the
/// vocabulary's embedding / lm_head rows; everything else replicated.
#[cfg(not(feature = "pro"))]
mod dims {
    pub const HIDDEN: usize = 4096;
    pub const HEADS: usize = 64;
    pub const Q_LORA: usize = 1024;
    pub const LAYERS: usize = 43;
    pub const TP: usize = 1;
    pub const EXPERTS: usize = 256;
    pub const INTER: usize = 2048;
    pub const O_GROUPS: usize = 8;
    pub const ROUTED_SCALE: f32 = 1.5;
    pub const INDEX_TOPK: usize = 512;
}
#[cfg(feature = "pro")]
mod dims {
    pub const HIDDEN: usize = 7168;
    pub const HEADS: usize = 16;
    pub const Q_LORA: usize = 1536;
    pub const LAYERS: usize = 61;
    pub const TP: usize = 8;
    pub const EXPERTS: usize = 384;
    pub const INTER: usize = 384;
    pub const O_GROUPS: usize = 2;
    pub const ROUTED_SCALE: f32 = 2.5;
    pub const INDEX_TOPK: usize = 1024;
}
pub use dims::*;
/// Head slots of the attention kernels: FlashMLA's sparse prefill runs 64 (or 128) heads, so a
/// DeepSeek-V4-Pro rank pads its 16 heads to 64 (SGLang's q and output tensors are [t, 64, 512],
/// rows past the rank's heads zero and unused).
pub const HEAD_SLOTS: usize = 64;
pub const HC: usize = 4;
pub const MIX: usize = 24;
pub const HEAD_DIM: usize = 512;
pub const ROPE: usize = 64;
pub const WINDOW: usize = 128;
pub const EPS: f32 = 1e-6;
pub const VOCAB: usize = 129280;
/// This rank's embedding / lm_head rows (SGLang's vocab-parallel shard).
pub const VOCAB_LOCAL: usize = VOCAB / TP;
/// The wo_a low rank per group.
pub const O_LORA: usize = 1024;

/// SWA KV cache page: 256 tokens x (448 fp8 nope + 64 bf16 rope) then 256 x 8 scale bytes,
/// padded to a multiple of 576 (deepseek_v4::kv_page_bytes<V4>(256)).
pub const SWA_PAGE_TOKENS: usize = 256;
pub const SWA_PAGE_BYTES: usize = (584 * SWA_PAGE_TOKENS).div_ceil(576) * 576;

/// Layer kinds by compress ratio (config.compress_ratios): 0 = sliding window only, 4 = C4 with the
/// sparse indexer, 128 = C128.
#[cfg(not(feature = "pro"))]
pub fn compress_ratio(layer: usize) -> usize {
    match layer { 0 | 1 | DRAFT_LAYER => 0, l if l % 2 == 0 => 4, _ => 128 }
}
/// DeepSeek-V4-Pro: compress_ratios [128, 128, 4, 128, 4, ...] (no sliding-window-only layer).
#[cfg(feature = "pro")]
pub fn compress_ratio(layer: usize) -> usize {
    match layer { DRAFT_LAYER => 0, 0 | 1 => 128, l if l % 2 == 0 => 4, _ => 128 }
}

/// The MTP draft's decoder layer (SGLang's DeepseekV4ModelNextN.decoder, compress ratio 0), run as
/// layer 43 after the target's 43: its weights load as `model.layers.43.*`, its SWA cache is the
/// pool's 44th layer (SGLang's draft pool shares the target's slots).
pub const DRAFT_LAYER: usize = LAYERS;

/// SGLang's draft parameters (DeepseekV4ForCausalLMNextN) the port loads; embedding and LM head are
/// the target's.
const DRAFT_PARAMS: &[&str] = &[
    "model.hc_head_fn", "model.hc_head_base", "model.hc_head_scale", "model.enorm.weight", "model.hnorm.weight",
    "model.e_proj.weight", "model.e_proj.weight_scale_inv", "model.h_proj.weight", "model.h_proj.weight_scale_inv",
    "model.shared_head.norm.weight",
    "model.decoder.hc_attn_fn", "model.decoder.hc_ffn_fn", "model.decoder.hc_attn_base", "model.decoder.hc_ffn_base",
    "model.decoder.hc_attn_scale", "model.decoder.hc_ffn_scale", "model.decoder.self_attn.attn_sink",
    "model.decoder.self_attn.wqkv_a.weight", "model.decoder.self_attn.wqkv_a.weight_scale_inv", "model.decoder.self_attn.q_norm.weight",
    "model.decoder.self_attn.wq_b.weight", "model.decoder.self_attn.wq_b.weight_scale_inv", "model.decoder.self_attn.kv_norm.weight",
    "model.decoder.self_attn.wo_a.weight", "model.decoder.self_attn.wo_a.weight_scale_inv", "model.decoder.self_attn.wo_b.weight",
    "model.decoder.self_attn.wo_b.weight_scale_inv", "model.decoder.self_attn.attn_mqa.k_scale", "model.decoder.self_attn.attn_mqa.v_scale",
    "model.decoder.mlp.gate.weight", "model.decoder.mlp.gate.e_score_correction_bias",
    "model.decoder.mlp.experts.w13_weight", "model.decoder.mlp.experts.w2_weight",
    "model.decoder.mlp.experts.w13_weight_scale_inv", "model.decoder.mlp.experts.w2_weight_scale_inv",
    "model.decoder.mlp.experts.output1_scale_scalar", "model.decoder.mlp.experts.output1_scale_gate_scalar",
    "model.decoder.mlp.experts.output2_scale_scalar",
    "model.decoder.mlp.shared_experts.gate_up_proj.weight", "model.decoder.mlp.shared_experts.gate_up_proj.weight_scale_inv",
    "model.decoder.mlp.shared_experts.down_proj.weight", "model.decoder.mlp.shared_experts.down_proj.weight_scale_inv",
    "model.decoder.input_layernorm.weight", "model.decoder.post_attention_layernorm.weight",
];

fn le_i32(v: &[i32]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }

/// C4 compressed cache page: 64 compressed tokens in the V4 FlashMLA layout.
pub const C4_PAGE_TOKENS: usize = 64;
pub const C4_PAGE_BYTES: usize = (584 * C4_PAGE_TOKENS).div_ceil(576) * 576;
/// One compressor state row: kv | overlap kv | score | overlap score (512 floats each).
pub const C4_STATE_ROW_BYTES: usize = 4 * HEAD_DIM * 4;
/// The C4 sparse indexer: 64 heads of 128 dims, top-INDEX_TOPK (512; Pro 1024).
pub const INDEX_HEADS: usize = 64;
pub const INDEX_DIM: usize = 128;
/// Index-K cache page (shares the C4 page ids): 64 keys x 128 FP8 bytes, then 64 f32 scales.
pub const INDEX_PAGE_BYTES: usize = C4_PAGE_TOKENS * (INDEX_DIM + 4);
/// The indexer compressor's state row (head dim 128).
pub const INDEX_STATE_ROW_BYTES: usize = 4 * INDEX_DIM * 4;
/// Query count from which SGLang scores a prefill chunk with the non-paged indexer
/// (SGLANG_OPT_DSV4_NONPAGED_INDEXER_MIN_QUERY_TOKENS).
pub const NONPAGED_MIN_QUERIES: usize = 8192;
/// C128 compressed cache page: 2 compressed tokens (one SWA page of 256 tokens), V4 layout.
pub const C128_PAGE_TOKENS: usize = 2;
pub const C128_PAGE_BYTES: usize = (584 * C128_PAGE_TOKENS).div_ceil(576) * 576;
/// One C128 state row: kv | score (512 floats each); the request ring holds 128 rows.
pub const C128_STATE_ROW_BYTES: usize = 2 * HEAD_DIM * 4;
/// Width of SGLang's per-token C128 page-index row in decode (the tape's extra_topk): FlashMLA's
/// split-KV scheduling sees it, so it is kept as is.
pub const C128_DECODE_WIDTH: usize = 8256;
/// The request's row in SGLang's request pool (keys the C128 state ring).
pub const RID: i64 = 1;

/// Where a request lives in the KV pools: its first page id (its pages are consecutive; page 0 is
/// the allocator's padding page) and its request-pool row (the C128 state ring). Cache contents are
/// per token, so placement never changes values.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Req { pub page0: usize, pub rid: i64 }

impl Req {
    /// The single-request placement (SGLang's first request on a fresh server: page 1, row 1).
    pub const SINGLE: Req = Req { page0: 1, rid: RID };
    /// SWA cache slot of position `pos`.
    pub fn swa_slot(&self, pos: usize) -> i32 { (self.page0 * SWA_PAGE_TOKENS + pos) as i32 }
    /// Slot of the `e`-th C4 compressed token (and index-K entry).
    pub fn c4_slot(&self, e: usize) -> i32 { (self.page0 * C4_PAGE_TOKENS + e) as i32 }
    /// Slot of the `e`-th C128 compressed token.
    pub fn c128_slot(&self, e: usize) -> i32 { (self.page0 * C128_PAGE_TOKENS + e) as i32 }
}

/// Per-request KV state. The request's slots start at page 1 of every pool (SGLang's allocator
/// keeps page 0 as the padding page); cache contents are per token, so slot numbering never
/// affects values.
///
/// The SWA cache (and the C4 compressor states, which SGLang keys by SWA page) is either one page
/// per full-pool page (a request's SWA page p is its page page0 + p), or a separate, smaller pool
/// (`with_pools`), as SGLang's hybrid SWA allocator keeps it: each request maps its logical SWA
/// pages to pool pages, holds them while it writes them, and gives back the pages that fell out of
/// its window (`swa_hold` / `swa_evict` / `swa_release`).
pub struct KvCache { swa: Buffer, pub pages: usize, pub swa_pages: usize, c4: Buffer, pub c4_pages: usize, c4_state: Buffer, state_rows: usize,
                     c128: Buffer, pub c128_pages: usize, c128_state: Buffer, index_k: Buffer, index_state: Buffer, pub rows: usize,
                     /// The request's page table row: ids 1..pages (page p of the request is page id p + 1).
                     pub page_row: Buffer,
                     /// Separate SWA pool: per request-pool row, logical SWA page -> pool page (-1: not held).
                     swa_table: Option<Vec<Vec<i32>>>, swa_free: Vec<i32> }

impl KvCache {
    pub fn new(max_tokens: usize) -> Result<Self> { Self::with_requests(max_tokens, RID as usize + 1) }

    /// Pools for `max_tokens` tokens in all (page 0 reserved) and request-pool rows 0..rows.
    pub fn with_requests(max_tokens: usize, rows: usize) -> Result<Self> { Self::with_pools(max_tokens, rows, None) }

    /// As `with_requests`, with a separate SWA pool of `swa_tokens` tokens when given.
    pub fn with_pools(max_tokens: usize, rows: usize, swa_tokens: Option<usize>) -> Result<Self> {
        let pages = 1 + max_tokens.div_ceil(SWA_PAGE_TOKENS);
        let swa_pages = swa_tokens.map(|t| 1 + t.div_ceil(SWA_PAGE_TOKENS)).unwrap_or(pages);
        let c4_pages = 1 + (max_tokens / 4).div_ceil(C4_PAGE_TOKENS);
        let state_rows = swa_pages * compress::ring_size(4) as usize;
        // With speculative decoding the MTP draft's layer (DRAFT_LAYER) has an SWA cache too.
        let swa_layers = if compress::speculative() { LAYERS + 1 } else { LAYERS };
        let swa = Buffer::alloc(swa_layers * swa_pages * SWA_PAGE_BYTES)?;
        let c4 = Buffer::alloc(LAYERS * c4_pages * C4_PAGE_BYTES)?;
        let c4_state = Buffer::alloc(LAYERS * state_rows * C4_STATE_ROW_BYTES)?;
        let c128_pages = 1 + (max_tokens / 128).div_ceil(C128_PAGE_TOKENS);
        let c128 = Buffer::alloc(LAYERS * c128_pages * C128_PAGE_BYTES)?;
        let c128_state = Buffer::alloc(LAYERS * rows * compress::ring_size(128) as usize * C128_STATE_ROW_BYTES)?;
        let index_k = Buffer::alloc(LAYERS * c4_pages * INDEX_PAGE_BYTES)?;
        let index_state = Buffer::alloc(LAYERS * state_rows * INDEX_STATE_ROW_BYTES)?;
        for b in [&swa, &c4, &c4_state, &c128, &c128_state, &index_k, &index_state] { b.zero()?; }
        let page_row = Buffer::from_host(&le_i32(&(1..pages as i32).collect::<Vec<i32>>()))?;
        let mut c = Self { swa, pages, swa_pages, c4, c4_pages, c4_state, state_rows, c128, c128_pages, c128_state, index_k, index_state, rows, page_row,
                           swa_table: swa_tokens.map(|_| vec![]), swa_free: vec![] };
        c.reset_swa_pool();
        Ok(c)
    }
    fn reset_swa_pool(&mut self) {
        if let Some(t) = self.swa_table.as_mut() {
            *t = vec![vec![]; self.rows];
            self.swa_free = (1..self.swa_pages as i32).rev().collect();
        }
    }
    /// Zero every pool on `s` (a new request in a reused cache starts from the state a fresh one has).
    pub fn clear(&mut self, s: Stream) -> Result<()> {
        for b in [&self.swa, &self.c4, &self.c4_state, &self.c128, &self.c128_state, &self.index_k, &self.index_state] {
            crate::cuda::memset_async(b.ptr, 0, b.bytes, s)?;
        }
        self.reset_swa_pool();
        Ok(())
    }
    /// SWA cache slot of position `pos` of request `r` (request row 0 is the padding row: page 0).
    pub fn swa_slot(&self, r: &Req, pos: usize) -> i32 {
        match &self.swa_table {
            None => r.swa_slot(pos),
            Some(_) if r.rid == 0 => (pos % SWA_PAGE_TOKENS) as i32,
            Some(t) => {
                let page = t[r.rid as usize].get(pos / SWA_PAGE_TOKENS).copied().unwrap_or(-1);
                assert!(page > 0, "SWA page of position {pos} of request {} not held", r.rid);
                page * SWA_PAGE_TOKENS as i32 + (pos % SWA_PAGE_TOKENS) as i32
            }
        }
    }
    /// Hold pool pages for a forward of request `r` writing positions start..end: the pages it
    /// writes are taken from the pool; the ones it reads before `start` (its window and compressor
    /// state) must still be held.
    pub fn swa_hold(&mut self, r: &Req, start: usize, end: usize) -> Result<()> {
        let Some(t) = self.swa_table.as_mut() else { return Ok(()) };
        let row = &mut t[r.rid as usize];
        let last = end.div_ceil(SWA_PAGE_TOKENS);
        if row.len() < last { row.resize(last, -1); }
        let read_from = start.saturating_sub(WINDOW + 8) / SWA_PAGE_TOKENS;
        let write_from = start / SWA_PAGE_TOKENS;
        if row[read_from..write_from].iter().any(|&p| p <= 0) { return Err(format!("request {}: SWA window evicted before position {start}", r.rid)); }
        for p in row[write_from..last].iter_mut() {
            if *p <= 0 { *p = self.swa_free.pop().ok_or("SWA pool exhausted")?; }
        }
        Ok(())
    }
    /// Give back request `r`'s pool pages that no later forward reads, once it holds `seq_len`
    /// tokens: the next token attends to the last WINDOW - 1 positions and its compressor state
    /// reads the last 8 (SGLang frees out-of-window SWA pages the same way, page by page).
    pub fn swa_evict(&mut self, r: &Req, seq_len: usize) {
        let Some(t) = self.swa_table.as_mut() else { return };
        let keep_from = seq_len.saturating_sub(WINDOW + 8) / SWA_PAGE_TOKENS;
        for p in t[r.rid as usize].iter_mut().take(keep_from) {
            if *p > 0 { self.swa_free.push(*p); *p = -1; }
        }
    }
    /// Give back every pool page of request `r` (it finished).
    pub fn swa_release(&mut self, r: &Req) {
        let Some(t) = self.swa_table.as_mut() else { return };
        for p in t[r.rid as usize].drain(..) { if p > 0 { self.swa_free.push(p); } }
    }
    /// Free pool pages (separate SWA pool only).
    pub fn swa_free_pages(&self) -> usize { self.swa_free.len() }
    fn swa_layer(&self, layer: usize) -> u64 { self.swa.ptr + (layer * self.swa_pages * SWA_PAGE_BYTES) as u64 }
    fn c4_layer(&self, layer: usize) -> u64 { self.c4.ptr + (layer * self.c4_pages * C4_PAGE_BYTES) as u64 }
    fn c4_state_layer(&self, layer: usize) -> u64 { self.c4_state.ptr + (layer * self.state_rows * C4_STATE_ROW_BYTES) as u64 }
    fn c128_layer(&self, layer: usize) -> u64 { self.c128.ptr + (layer * self.c128_pages * C128_PAGE_BYTES) as u64 }
    /// After a speculative verify of `drafts` rows per request at positions seq_len.., of which
    /// the first `accept_len` were kept: reset the C128 compressor ring rows of the rejected
    /// positions (kv 0.0, score -inf), as SGLang's clear_unaccepted_c128_draft_states (a
    /// boundary's compression can read a rejected draft slot; C4 rings overwrite theirs before
    /// reading them). One kernel for every request and C128 layer: per request (r, seq_len,
    /// accept_len); `args` is device scratch for the index upload (4 * (LAYERS + 3 * rows) bytes).
    pub fn clear_c128_rejected_batch(&self, rows: &[(Req, usize, usize)], drafts: usize, args: u64, s: Stream) -> Result<()> {
        let layers: Vec<i32> = (0..LAYERS).filter(|&l| compress_ratio(l) == 128).map(|l| l as i32).collect();
        let mut v = layers.clone();
        v.extend(rows.iter().map(|r| r.0.rid as i32));
        v.extend(rows.iter().map(|r| r.1 as i32));
        v.extend(rows.iter().map(|r| r.2 as i32));
        crate::cuda::upload_async(args, &le_i32(&v), s)?;
        let ring = compress::ring_size(128) as usize;
        crate::exact::clear_c128_rejected(s, self.c128_state_layer(0), self.rows * ring * C128_STATE_ROW_BYTES / 4, args, layers.len(), rows.len(), ring, drafts, HEAD_DIM)
    }
    fn c128_state_layer(&self, layer: usize) -> u64 { self.c128_state.ptr + (layer * self.rows * compress::ring_size(128) as usize * C128_STATE_ROW_BYTES) as u64 }
    /// The device page-table row of a request (its consecutive page ids).
    pub fn req_page_row(&self, r: &Req) -> u64 { self.page_row.ptr + (4 * (r.page0 - 1)) as u64 }
    fn index_k_layer(&self, layer: usize) -> u64 { self.index_k.ptr + (layer * self.c4_pages * INDEX_PAGE_BYTES) as u64 }
    fn index_state_layer(&self, layer: usize) -> u64 { self.index_state.ptr + (layer * self.state_rows * INDEX_STATE_ROW_BYTES) as u64 }
    /// Width of the request's page table for `seq_len` tokens (page p holds tokens 256p..256p+255,
    /// C4 entries 64p..64p+63).
    fn page_width(seq_len: usize) -> usize { seq_len.div_ceil(SWA_PAGE_TOKENS) }
    /// The page table width of the cache's capacity (the decode indexer's fixed row).
    pub fn capacity_width(&self) -> usize { self.pages - 1 }

}

/// RoPE tables (freqs_cis, complex64 [1048576, 32]) dumped from the reference process: one for the
/// sliding-window layers, one for the compressed layers.
/// The RoPE tables (freqs_cis) SGLang computes at load, as exported from the reference process.
pub struct Tables { pub swa: u64, pub compressed: u64, _owned: Vec<Buffer> }

impl Tables {
    pub fn load(dir: &std::path::Path) -> Result<Self> {
        let g = Gold::open(dir);
        let swa = Buffer::from_host(&g.bytes("model.layers.0.self_attn.freqs_cis")?)?;
        let compressed = Buffer::from_host(&g.bytes("model.layers.2.self_attn.freqs_cis")?)?;
        Ok(Self { swa: swa.ptr, compressed: compressed.ptr, _owned: vec![swa, compressed] })
    }
    /// One table for every layer (DeepSeek-V4-Pro has no sliding-window-only layer: the reference
    /// holds a single freqs_cis).
    pub fn single(ptr: u64) -> Self { Self { swa: ptr, compressed: ptr, _owned: vec![] } }
    /// The tables stored in a prepared package (already on the device).
    pub fn prepared(w: &crate::prepared::Weights) -> Result<Self> {
        Ok(Self { swa: w.get(crate::prepare::TABLE_SWA)?, compressed: w.get(crate::prepare::TABLE_COMPRESSED)?, _owned: vec![] })
    }
    pub fn freqs(&self, layer: usize) -> u64 { if compress_ratio(layer) == 0 { self.swa } else { self.compressed } }
}

/// mHC state carried between sublayers: the residual stream and the mixes of the last mHC pre.
/// The hyper-connection state between sublayers: the current residual streams and mixes. Two
/// buffer sets (and two layer-output buffers) are allocated once per forward and alternate, so a
/// forward's memory does not grow with its depth (a 16k-token residual is 512 MiB).
/// Tensor parallelism: completing a sum over the TP ranks. SGLang all-reduces the embedding, each
/// layer's attention output and each layer's MoE output (bf16 partials; fp32 accumulation over the
/// ranks in order, one rounding back to bf16).
pub enum Reduce {
    /// Development on one rank: each partial is compared with the reference rank's all-reduce input
    /// (`ar-<k>.in`) and replaced by the reference's sum (`ar-<k>.out`), in call order.
    /// `all`: every reference rank's golds of the forward. SGLang's ranks do not hold the same
    /// slice at every layer (rank r's shard of one layer's attention or MoE weights may be another
    /// slice than its shard of the next layer's), so a partial matches whichever rank held this
    /// package's slice there.
    Gold { gold: Gold, all: Vec<Gold>, next: usize, results: Vec<crate::gold::Comparison> },
    /// The rank processes of one model, one per GPU on one host: every rank writes its partial
    /// to `dir` (a tmpfs directory), waits for all `ranks` partials of the same call, and takes
    /// their sum: fp32 accumulation from zero in rank order, one round-to-nearest-even back to
    /// bf16 (the arithmetic of the reference's all-reduce kernel). Slow and simple: a partial is
    /// one bf16 row per token. `gold`: the reference's sums (`ar-<k>.out`) to compare with;
    /// `differ` collects the calls whose sum is not the reference's.
    Shm { dir: std::path::PathBuf, rank: usize, ranks: usize, next: usize, gold: Option<Gold>, differ: Vec<usize>, seconds: f64 },
}

/// One collective among the rank processes through `dir`: publish this rank's `mine` as
/// `<tag>.r<rank>` (written under a temporary name, then renamed), wait for every rank's file and
/// return `combine` of them in rank order.
pub fn shm_exchange(dir: &std::path::Path, tag: &str, rank: usize, ranks: usize, mine: &[u8],
                    combine: impl FnOnce(&[Vec<u8>]) -> Vec<u8>) -> Result<Vec<u8>> {
    let path = |r: usize| dir.join(format!("{tag}.r{r}"));
    let tmp = dir.join(format!("{tag}.r{rank}.tmp"));
    std::fs::write(&tmp, mine).map_err(|e| format!("{}: {e}", tmp.display()))?;
    std::fs::rename(&tmp, path(rank)).map_err(|e| format!("{}: {e}", tmp.display()))?;
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(600);
    let mut parts: Vec<Option<Vec<u8>>> = vec![None; ranks];
    parts[rank] = Some(mine.to_vec());
    loop {
        for r in 0..ranks {
            if parts[r].is_none() {
                if let Ok(data) = std::fs::read(path(r)) { if data.len() == mine.len() { parts[r] = Some(data); } }
            }
        }
        if parts.iter().all(Option::is_some) { break; }
        if dir.join("abort").exists() { return Err("another rank failed".into()); }
        if std::time::Instant::now() > deadline { return Err(format!("{tag}: a rank's partial did not arrive")); }
        std::thread::sleep(std::time::Duration::from_micros(200));
    }
    Ok(combine(&parts.into_iter().map(Option::unwrap).collect::<Vec<_>>()))
}

pub struct Hc { pub residual: u64, pub post_mix: u64, pub comb_mix: u64, sets: [HcSet; 2], cur: usize, outs: [u64; 2], out_cur: usize }

#[derive(Clone, Copy)]
struct HcSet { residual: u64, post_mix: u64, comb_mix: u64, layer_input: u64 }

impl Hc {
    fn new(t: usize, persist: &mut Arena) -> Result<Self> {
        let mut set = || -> Result<HcSet> {
            Ok(HcSet { residual: persist.alloc(t * HC * HIDDEN * 2)?, post_mix: persist.alloc(t * HC * 4)?,
                       comb_mix: persist.alloc(t * HC * HC * 4)?, layer_input: persist.alloc(t * HIDDEN * 2)? })
        };
        let sets = [set()?, set()?];
        let outs = [persist.alloc(t * HIDDEN * 2)?, persist.alloc(t * HIDDEN * 2)?];
        Ok(Self { residual: sets[0].residual, post_mix: sets[0].post_mix, comb_mix: sets[0].comb_mix, sets, cur: 0, outs, out_cur: 0 })
    }
    /// The buffer set the next mHC pre writes (not the current state's).
    fn next(&self) -> HcSet { self.sets[1 - self.cur] }
    fn advance(&mut self) {
        self.cur = 1 - self.cur;
        let s = self.sets[self.cur];
        self.residual = s.residual;
        self.post_mix = s.post_mix;
        self.comb_mix = s.comb_mix;
    }
    /// A buffer for this layer's output (alternating, so the previous layer's stays readable).
    pub fn layer_out(&mut self) -> u64 {
        self.out_cur = 1 - self.out_cur;
        self.outs[self.out_cur]
    }
}

pub struct Model<'a> {
    pub dev: Device,
    pub k: &'static Kernels,
    pub w: DeviceWeights<'a>,
    pub tables: Tables,
    pub cache: KvCache,
    pub trtllm_cubins: String,
    pub lt: Lt,
    /// Debug: keep this layer's sublayer inputs/outputs (named like the reference's module golds).
    pub probe: Option<usize>,
    pub probes: Vec<(String, u64, usize)>,
    /// Debug: the reference's per-launch snapshots and the label of the forward/module to check.
    pub snaps: Option<crate::snapshots::Snapshots>,
    pub snap_label: Option<String>,
    /// The forward's FlashMLA split-KV schedule for C128 decode: (sched meta, num_splits, ready).
    pub c128_sched: Option<(u64, u64, bool)>,
    /// The decode inputs region (decode_inputs), allocated on first decode.
    pub decode_in: Option<Buffer>,
    /// Compute logits for the last position only (requests without input logprobs).
    pub last_logits_only: bool,
    /// Rows the head keeps for a packed prefill batch (each request's last token).
    pub prune_rows: Vec<usize>,
    /// Where the head copies the hyper-connection state before hc_head ([t, 4, hidden] bf16: SGLang's
    /// `pre_hc_head`, the hidden states its MTP draft reads), when set.
    pub hc_dest: Option<u64>,
    /// This process's TP rank (0 on one GPU): its slice of the per-head parameters.
    pub rank: usize,
    /// Tensor parallelism: how this rank's partial sums become the sum over the ranks (None: one GPU).
    pub reduce: Option<Reduce>,
    /// torch's cuBLAS workspace (the draft's fp32 GEMM), allocated on first use.
    pub blas_ws: Option<Buffer>,
    /// Side streams for SGLang's overlapped work (see `Streams`).
    pub streams: Streams,
    /// The current prefill chunk's metadata (set by `forward`).
    pub chunk: ChunkMeta,
    /// Serving state reused across requests (`generate_overlapped`).
    pub serving: Option<Serving>,
    /// The verify metadata region (`verify_meta`) and whether it holds the next verify's.
    pub verify_in: Option<Buffer>,
    pub verify_ready: bool,
    /// The current decode step's input layout, and whether its C4 layers run the sparse indexer
    /// (some request holds more than 512 compressed entries).
    pub dl: DecodeLayout,
    pub dl_indexer: bool,
    /// Page-table width per request in decode (0: the cache's whole capacity, one request).
    pub decode_w: usize,
    /// `serve_mtp`'s graphs and buffers, kept between calls.
    pub mtp_serving: Option<MtpServing>,
}

/// The side streams SGLang's forward overlaps work on (alt streams): in decode, the KV store, the
/// compressor and the C4 indexer run beside the query path; in every forward the shared expert runs
/// beside the routed experts. Each kernel sees the same inputs either way, so values do not change.
/// Empty (everything on the main stream) until `Model::enable_streams`.
#[derive(Default)]
pub struct Streams { cur: Option<Stream>, side: Vec<Stream>, events: Vec<crate::cuda::Event>, next: std::cell::Cell<usize> }

/// SGLang's chunked_prefill_size on this deployment: prompts are prefilled in chunks of this many tokens.
pub const PREFILL_CHUNK: usize = 16384;

/// (persist, scratch) arena bytes for forwards of up to `t` tokens: the hyper-connection state,
/// batch and logits rows live in persist; one layer's activations (including the indexer's MQA
/// logits at 128k contexts, 2 GiB) in scratch.
/// The routed MoE's workspace: its expert tiles padded per touched expert (up to all 256)
/// dominate at small token counts; a verify or draft extend of 1,024 rows (256 requests x 4)
/// needs more than 256 MiB. Up to 1 GiB when the arena has it, never less than 256 MiB or 96 KiB
/// per token.
pub fn moe_scratch_bytes(t: usize, available: usize) -> usize {
    (256usize << 20).max(t * (96 << 10)).max(available.min(1 << 30))
}

pub fn arena_bytes(t: usize) -> (usize, usize) {
    ((1usize << 30).max(t * (384 << 10)), (1usize << 30).max(t * (1 << 20)))
}

/// One request's tokens in a prefill forward: positions start..start + len of request `req`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Seg { pub req: Req, pub start: usize, pub len: usize }

/// One prefill forward: `t` new tokens, the segments of one or more requests in order (SGLang
/// packs requests into one extend batch), or a decode step. `ids` and `positions` are device
/// addresses (int64); a decode step's live in the model's decode inputs. `pos32` holds the
/// positions as int32 (the indexer and the sparse-prefill metadata read those). `start` and
/// `req` are the first segment's.
pub struct Batch { pub t: usize, pub start: usize, pub ids: u64, pub positions: u64, pub pos32: u64, pub decode: bool, pub req: Req, pub segs: Vec<Seg>,
                   /// A speculative verify step (decode rows, several per request; compressors run as an extend of
                   /// each request's `segs`).
                   pub verify: bool, _own: Vec<Buffer> }

impl Batch {
    pub fn new(ids: &[i64], start: usize) -> Result<Self> {
        let pos: Vec<i64> = (0..ids.len()).map(|i| (start + i) as i64).collect();
        let ids_b = Buffer::from_host(&ids.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>())?;
        let pos_b = Buffer::from_host(&pos.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>())?;
        let pos32 = Buffer::from_host(&pos.iter().flat_map(|v| (*v as i32).to_le_bytes()).collect::<Vec<u8>>())?;
        Ok(Self { t: ids.len(), start, decode: false, ids: ids_b.ptr, positions: pos_b.ptr, pos32: pos32.ptr, req: Req::SINGLE,
                  segs: vec![Seg { req: Req::SINGLE, start, len: ids.len() }], verify: false, _own: vec![ids_b, pos_b, pos32] })
    }

    /// The same chunk in `a`, uploaded asynchronously on `s` (stream-ordered after earlier work).
    pub fn upload(ids: &[i64], start: usize, a: &mut Arena, s: Stream) -> Result<Self> { Self::upload_req(ids, start, Req::SINGLE, a, s) }

    /// A chunk of request `req`.
    pub fn upload_req(ids: &[i64], start: usize, req: Req, a: &mut Arena, s: Stream) -> Result<Self> {
        Self::upload_segs(ids, &[Seg { req, start, len: ids.len() }], a, s)
    }

    /// The segments `segs` (their tokens concatenated in `ids`).
    pub fn upload_segs(ids: &[i64], segs: &[Seg], a: &mut Arena, s: Stream) -> Result<Self> {
        let t = ids.len();
        if segs.iter().map(|g| g.len).sum::<usize>() != t || segs.is_empty() { return Err("prefill segments do not cover the batch".into()); }
        let pos: Vec<usize> = segs.iter().flat_map(|g| g.start..g.start + g.len).collect();
        let (d_ids, d_pos, d_pos32) = (a.alloc(t * 8)?, a.alloc(t * 8)?, a.alloc(t * 4)?);
        crate::cuda::upload_async(d_ids, &ids.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>(), s)?;
        crate::cuda::upload_async(d_pos, &pos.iter().flat_map(|&p| (p as i64).to_le_bytes()).collect::<Vec<u8>>(), s)?;
        crate::cuda::upload_async(d_pos32, &pos.iter().flat_map(|&p| (p as i32).to_le_bytes()).collect::<Vec<u8>>(), s)?;
        Ok(Self { t, start: segs[0].start, decode: false, ids: d_ids, positions: d_pos, pos32: d_pos32, req: segs[0].req, segs: segs.to_vec(), verify: false, _own: vec![] })
    }
}

/// State a serving process keeps across requests so its decode graphs stay valid (SGLang captures
/// its graphs once at startup): the KV cache at a capacity, the arenas, the sampled-token slot and
/// the captured graphs (keyed by whether the C4 indexer runs).
pub struct Serving { capacity: usize, persist: Arena, scratch: Arena, token: Buffer, graphs: Vec<(bool, crate::cuda::Graph)>, eager: Vec<bool>,
                     batch_graphs: Vec<((usize, bool), crate::cuda::Graph)>, batch_eager: Vec<(usize, bool)> }

/// The decode batch SGLang's graph server runs for `b` requests: its CUDA graphs exist for 1, 2, 4,
/// 8, 12, 16 and multiples of 8 up to 256 requests; a batch in between replays the next one, the
/// extra rows padding (sequence length 1 at the padding page, request row 0).
pub fn graph_batch(b: usize) -> usize {
    match b { 0..=2 => b, 3..=4 => 4, 5..=8 => 8, 9..=12 => 12, 13..=16 => 16, _ => b.div_ceil(8) * 8 }
}

/// One forward of a serving schedule, as SGLang's scheduler decides them (`Model::serve` runs them).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Step {
    /// One prefill forward over chunks of one or more requests (SGLang packs them into one extend
    /// batch). A request's sequence is its prompt then the tokens it holds; a chunk ending it
    /// samples the next token.
    Prefill { chunks: Vec<Chunk> },
    /// One decode step of requests `rows`, in batch row order.
    Decode { rows: Vec<usize> },
    /// SGLang retracted request `req` (its cache freed) keeping its first `keep` sampled tokens;
    /// it is prefilled again (prompt then those tokens) by later steps. No forward.
    Retract { req: usize, keep: usize },
}

/// Positions start..start + len of request `req`'s sequence, in a prefill forward.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Chunk { pub req: usize, pub start: usize, pub len: usize }

impl Step {
    /// SGLang's schedule for requests arriving together that all fit in the SWA pool: every prompt
    /// prefilled in chunks, in order, then all decoded as one batch to `max_new_tokens` tokens.
    pub fn prefill_then_decode(lens: &[usize], max_new_tokens: usize) -> Vec<Step> {
        let mut v = vec![];
        for (req, &len) in lens.iter().enumerate() {
            for start in (0..len).step_by(PREFILL_CHUNK) { v.push(Step::Prefill { chunks: vec![Chunk { req, start, len: PREFILL_CHUNK.min(len - start) }] }); }
        }
        for _ in 1..max_new_tokens { v.push(Step::Decode { rows: (0..lens.len()).collect() }); }
        v
    }
}

/// Integer tensor of a gold (int32 or int64) as i64.
pub fn gold_i64(g: &Gold, name: &str) -> Result<Vec<i64>> {
    let (meta, raw) = (g.meta(name)?, g.bytes(name)?);
    Ok(match meta.dtype.trim_start_matches("torch.") {
        "int64" => raw.chunks_exact(8).map(|v| i64::from_le_bytes(v.try_into().unwrap())).collect(),
        "int32" => raw.chunks_exact(4).map(|v| i32::from_le_bytes(v.try_into().unwrap()) as i64).collect(),
        d => return Err(format!("gold {name}: dtype {d}")),
    })
}

/// One step of a concurrent MTP schedule: a prefill of chunks (several requests packed, a chunk
/// covering part of a request's prompt plus held tokens), a decode round over requests in batch
/// order, or a retraction (the request keeps `keep` tokens and is prefilled again).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MtpStep { Prefill(Vec<Chunk>), Round(Vec<usize>), Retract { req: usize, keep: usize } }

/// A forward replayed as a CUDA graph: run eagerly on its first uses (the first loads every
/// function it calls), captured on use `DSV4_GRAPH_AT` (default 3: shapes a batch's tail sees
/// once or twice stay eager, cheaper than a capture), replayed after.
#[derive(Default)]
pub struct GraphSlot { uses: usize, graph: Option<crate::cuda::Graph> }

impl GraphSlot {
    /// The use on which a slot captures (DSV4_GRAPH_AT, default 3, at least 2: a first use may
    /// initialize lazily, which a capture cannot).
    pub fn graph_at() -> usize {
        static AT: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
        *AT.get_or_init(|| std::env::var("DSV4_GRAPH_AT").ok().and_then(|v| v.parse().ok()).unwrap_or(3).max(2))
    }

    /// What the next `run` does: "replay", "capture" or "eager".
    pub fn next(&self, enabled: bool) -> &'static str {
        if self.graph.is_some() { "replay" } else if enabled && self.uses + 1 >= Self::graph_at() { "capture" } else { "eager" }
    }

    /// Eager until use `graph_at()`, which captures; replays after. A capture costs more than an
    /// eager run (95 vs 55 ms for a 256-request verify), so a shape seen once or twice stays eager.
    pub fn run<M>(&mut self, m: &mut M, s: Stream, enabled: bool, mut f: impl FnMut(&mut M) -> Result<()>) -> Result<()> {
        if let Some(g) = &self.graph { return g.launch(s); }
        let at = Self::graph_at();
        self.uses += 1;
        if enabled && self.uses >= at {
            let (g, ()) = crate::cuda::Graph::capture(s, || f(m))?;
            g.launch(s)?;
            self.graph = Some(g);
            return Ok(());
        }
        f(m)
    }
}

/// What `serve_mtp` keeps from one call to the next of the same shape, as a server keeps them for
/// its lifetime: the arenas and buffers its graphs read, and the graphs. The KV cache stays too
/// (zeroed at each call, `KvCache::clear`).
pub struct MtpServing {
    /// Requests, page-table width, SWA pool tokens, largest round, largest forward's rows.
    key: (usize, usize, usize, usize, usize),
    /// The decode inputs, the verify metadata and the SWA pool the graphs were captured on.
    ptrs: (u64, u64, u64),
    persist: Arena, scratch: Arena,
    hc_t: Buffer, hc_d: Buffer, hc_src: Buffer, hc_1: Buffer, hc_2: Buffer,
    d1v: Buffer, hcv: Buffer, xv: Buffer, wide: Buffer, probs: Buffer,
    res: Buffer, ext: Buffer, tok: Buffer, xb: Buffer, idx: Buffer, cargs: Buffer, pre: Buffer,
    g_d1: Vec<(usize, GraphSlot)>, g_d2: Vec<(usize, GraphSlot)>, g_ext: Vec<(usize, GraphSlot)>,
    g_verify: Vec<((usize, VerifyKey, bool), GraphSlot)>,
}

/// A verify graph's shape beyond its rows: the compress / write plan counts of C4 and C128.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct VerifyKey { pub c4: (usize, usize), pub c128: (usize, usize) }

/// One request's MTP generation: tokens, each round's accepted draft count, prefill and decode wall time.
#[derive(Debug)]
pub struct MtpRun { pub tokens: Vec<usize>, pub accepted: Vec<usize>, pub prefill_s: f64, pub decode_s: f64 }

/// The outcome of `Model::generate_batch`: each request's tokens and the phase timings.
#[derive(Debug)]
pub struct BatchRun { pub tokens: Vec<Vec<usize>>, pub prefill_s: f64, pub decode_s: f64, pub decode_steps: usize, pub wall_s: f64,
                      /// Tokens produced after the last prefill (in `decode_s`; MTP only, else 0).
                      pub decode_tokens: usize }

/// Debug check of one forward of a batched run against the reference's gold forward `fwd`: the
/// fp32 next-token logits [rows, vocab] and the sampled ids [rows] (SHA-256 against hash-only golds).
fn check_forward(dir: &std::path::Path, fwd: usize, logits_f32: u64, rows: usize, tokens: u64) -> Result<()> {
    crate::cuda::ctx_sync()?;
    let gold = Gold::open(dir.join(format!("fwd-{fwd:03}")));
    let l = crate::gold::compare(&gold, "logits_processor.out.next_token_logits", &crate::cuda::download(logits_f32, rows * VOCAB * 4)?)?;
    let t = crate::gold::compare(&gold, "sample.out", &crate::cuda::download(tokens, rows * 8)?)?;
    println!("{}", serde_json::json!({"check_fwd": fwd, "rows": rows, "logits_equal": l.equal, "sample_equal": t.equal}));
    if !l.equal || !t.equal { return Err(format!("forward {fwd} differs from the reference")); }
    Ok(())
}

/// One compressed layer kind's part of `ChunkMeta`: compress / write plans, the output slot of each
/// token's compressed row (int64, 0 when none), the request's compressed slots (dequant gather), the
/// combine metadata (`sgl::combine_topk_swa`).
#[derive(Clone, Copy, Debug, Default)]
pub struct RatioMeta { pub plan_c: u64, pub num_c: usize, pub plan_w: u64, pub num_w: usize, pub out_loc: u64, pub slots: u64, pub n: usize, pub combine: u64,
                       /// C128: the compressed rows of each request's region (every query takes all it may see).
                       pub per_req: usize }

/// A prefill chunk's metadata shared by every layer (SGLang's forward metadata): built on the host
/// once per chunk and uploaded in one copy, so layers launch without host round trips.
#[derive(Clone, Copy, Debug, Default)]
pub struct ChunkMeta {
    /// int32 [t]: each token's SWA cache slot.
    pub swa_out: u64,
    /// int32 [gather]: SWA slots of the window union of the chunk's queries.
    pub swa_ids: u64,
    pub gather: usize,
    /// int32 [0, 1, 2, ...]: a top-k row taking every compressed token in order (read with stride 0).
    pub ident: u64,
    /// int32 [t] zeros: the one-entry top-k rows of C128 while it holds one slot (SGLang's [t, 1]
    /// rows have stride 1, which Triton specializes; every row reads its own 0).
    pub zeros: u64,
    /// int32 [t]: each query's C4 compressed length (the indexer's key range).
    pub ke: u64,
    pub c0_combine: u64,
    pub c4: RatioMeta,
    pub c128: RatioMeta,
    /// Requests in the forward; bytes per combine-metadata array (`sgl::combine_topk_swa`).
    pub nreqs: usize,
    pub meta_slot: usize,
    /// Several requests: int32 [t, table_width], each query's request page ids (the paged
    /// indexer's block table and the top-k page table, one row per query as SGLang's).
    pub table: u64,
    pub table_width: usize,
}

/// Everything a decode step of `b` requests reads that depends on their positions, in one device
/// region filled by one upload before the step (so the step itself can be a replayed CUDA graph).
/// Per-request arrays are row-major [b, ...], as SGLang's batched decode metadata.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DecodeLayout {
    pub b: usize,
    /// Page-table width per request (the indexer's block table rows).
    pub w: usize,
    pub ids: usize,       // i64 [b]
    pub pos: usize,       // i64 [b]
    pub pos32: usize,     // i32 [b]
    pub swa_slot: usize,  // i32 [b]: each new token's SWA slot
    pub swa_len: usize,   // i32 [b]: window lengths
    pub swa_idx: usize,   // i32 [b, 128]: window slots, newest first
    pub c4_plan: usize,   // 16 B [b]: C4 decode plans
    pub c4_loc: usize,    // i64 [b]: compressed slot written this step (0 if none)
    pub c4_len: usize,    // i32 [b]: entries FlashMLA reads (at most 512)
    pub c4_ids: usize,    // i32 [b, 512]: every entry while at most 512
    pub c4_ctx: usize,    // i32 [b]: entries the indexer scores
    pub c128_plan: usize,
    pub c128_loc: usize,
    pub c128_len: usize,  // i32 [b], clamped to 1
    pub c128_ids: usize,  // i32 [b, 8256]
    pub sched: usize,     // i32 [(sms + 1) x 2]: paged MQA-logits schedule
    pub fill: usize,      // i32 [b, 5]: c4 base, c4 count, c128 base, c128 count, page0 (`fill_decode_ids`)
    pub table: usize,     // i32 [b, w]: each request's page ids
    /// Where the arrays `exact::fill_decode_ids` can write on the device begin (c4_ids, c128_ids
    /// and table come last).
    pub dev_start: usize,
    pub bytes: usize,
}

impl DecodeLayout {
    pub fn new(b: usize, w: usize, sms: usize) -> Self {
        let mut off = 0usize;
        let mut take = |bytes: usize| { let o = off; off = (off + bytes).div_ceil(256) * 256; o };
        let mut l = DecodeLayout { b, w, ..Default::default() };
        l.ids = take(8 * b); l.pos = take(8 * b); l.pos32 = take(4 * b); l.swa_slot = take(4 * b); l.swa_len = take(4 * b);
        l.swa_idx = take(4 * b * WINDOW);
        l.c4_plan = take(16 * b); l.c4_loc = take(8 * b); l.c4_len = take(4 * b); l.c4_ctx = take(4 * b);
        l.c128_plan = take(16 * b); l.c128_loc = take(8 * b); l.c128_len = take(4 * b);
        l.sched = take(8 * (sms + 1)); l.fill = take(20 * b);
        l.dev_start = take(0);
        l.c4_ids = take(4 * b * INDEX_TOPK); l.c128_ids = take(4 * b * C128_DECODE_WIDTH); l.table = take(4 * b * w.max(1));
        l.bytes = off;
        l
    }
}

/// A request in a decode batch: where it lives, the position of the token it feeds, that token.
#[derive(Clone, Copy, Debug)]
pub struct DecodeRow { pub req: Req, pub pos: usize, pub id: i64 }

/// A cold start in flight (Model::start_prepared).
pub struct Startup {
    load: crate::prepared::Load,
    device: std::thread::JoinHandle<Result<Device>>,
    kernels: &'static Kernels,
    images: std::thread::JoinHandle<Result<Vec<(String, Vec<u8>)>>>,
    lt_path: String,
}

impl Startup {
    /// Finish the cold start: weights stream onto the device on this thread while a helper loads
    /// the kernel modules, opens cuBLASLt and loads its kernels for the model's GEMMs. Returns once
    /// everything the first forward needs is resident; `timeline` gets the phases (s since `t0`).
    pub fn finish(self, t0: std::time::Instant, timeline: &mut Vec<(&'static str, f64)>) -> Result<Model<'static>> {
        let dev = self.device.join().map_err(|_| "CUDA context thread panicked")??;
        crate::cuda::set_current(dev.ctx)?;
        timeline.push(("cuda_context", t0.elapsed().as_secs_f64()));
        let (ctx, sms, kernels, images, lt_path) = (dev.ctx as usize, dev.sm_count as i32, self.kernels, self.images, self.lt_path);
        // Helpers in the context while this thread uploads the weights: cuBLASLt first (its lazy
        // kernel loads are the longest), one thread per GEMM shape; the bundle's modules on another.
        let modules = std::thread::spawn(move || -> Result<f64> {
            crate::cuda::set_current(ctx as crate::cuda::CUcontext)?;
            kernels.load_all(images.join().map_err(|_| "kernel reader panicked")??)?;
            Ok(t0.elapsed().as_secs_f64())
        });
        let helper = std::thread::spawn(move || -> Result<(Lt, f64)> {
            crate::cuda::set_current(ctx as crate::cuda::CUcontext)?;
            let lt = Lt::init(&lt_path, sms)?;
            // The model's cuBLASLt GEMMs: C4 / C128 compressor wkv_gate (fp32 out) and lm_head (bf16
            // out), for every prefill chunk size (<= 32 tokens) and decode (1).
            let ms: Vec<usize> = (1..=32).collect();
            let shapes = [(4 * HEAD_DIM, HIDDEN, true), (2 * HEAD_DIM, HIDDEN, true), (VOCAB, HIDDEN, false)];
            let calls = std::thread::scope(|scope| -> Result<Vec<(usize, Vec<(usize, f64, i32)>)>> {
                let handles: Vec<_> = shapes.iter().map(|&(n, k, f)| { let (lt, ms) = (&lt, &ms); scope.spawn(move || -> Result<_> {
                    crate::cuda::set_current(ctx as crate::cuda::CUcontext)?;
                    Ok((n, lt.warm(n, k, f, ms)?))
                }) }).collect();
                handles.into_iter().map(|h| h.join().map_err(|_| "cuBLASLt warm-up panicked")?).collect()
            })?;
            if std::env::var_os("DSV4_STAGE_TIMES").is_some() {
                let slow: Vec<_> = calls.iter().flat_map(|(n, c)| c.iter().filter(|c| c.1 > 20.0).map(move |c| serde_json::json!([n, c.0, (c.1 * 10.0).round() / 10.0, c.2]))).collect();
                println!("{}", serde_json::json!({"cublaslt_warm_over_20ms_n_m_ms_algo": slow}));
            }
            Ok((lt, t0.elapsed().as_secs_f64()))
        });
        let weights = self.load.finish(dev.stream, dev.ctx)?;
        timeline.push(("weights_on_device", t0.elapsed().as_secs_f64()));
        let modules = modules.join().map_err(|_| "module loader panicked")??;
        let (lt, warmed) = helper.join().map_err(|_| "startup helper panicked")??;
        timeline.push(("kernel_modules", modules));
        timeline.push(("cublaslt_kernels_warm", warmed));
        crate::runtime::install_redirect(kernels);
        Ok(Model { dev, k: kernels, tables: Tables::prepared(&weights)?, w: DeviceWeights::from_prepared(weights), cache: KvCache::new(SWA_PAGE_TOKENS)?,
                   trtllm_cubins: std::env::var("DSV4_TRTLLM_CUBINS").unwrap_or_else(|_| "/dsv4/kernels/trtllm_bmm".into()),
                   lt, probe: None, probes: vec![], snaps: None, snap_label: None, c128_sched: None, decode_in: None, last_logits_only: false, prune_rows: vec![], hc_dest: None, reduce: None, blas_ws: None, streams: Streams::default(), chunk: ChunkMeta::default(), serving: None, verify_in: None, verify_ready: false, dl: DecodeLayout::default(), dl_indexer: false, decode_w: 0, mtp_serving: None, rank: 0 })
    }
}

/// One request's greedy generation: the sampled ids and each forward's logprobs.
pub struct Generation { pub tokens: Vec<usize>, pub logprobs: Vec<crate::logprobs::Logprobs>, pub ms: Vec<u128>,
                       /// Per step: forward + greedy token (no logprobs), the per-token latency SGLang's bench sees.
                       pub token_ms: Vec<f64>,
                       /// Per step: inputs + launch + device sync (the forward alone).
                       pub forward_ms: Vec<f64> }

impl<'a> Model<'a> {
    /// The model on device 0: the kernel bundle's CUBINs (with the launch redirect installed), the
    /// checkpoint's parameters (uploaded on first use), the reference's RoPE tables, no KV cache yet.
    /// The model over already chosen weights and tables (rungs and the Pro rank process).
    pub fn assemble(dev: Device, kernels: &'static Kernels, w: DeviceWeights<'a>, tables: Tables, cache_tokens: usize) -> Result<Self> {
        crate::runtime::install_redirect(kernels);
        let sms = dev.sm_count as i32;
        Ok(Model { dev, k: kernels, w, tables, cache: KvCache::new(cache_tokens)?,
                   trtllm_cubins: std::env::var("DSV4_TRTLLM_CUBINS").unwrap_or_else(|_| "/dsv4/kernels/trtllm_bmm".into()),
                   lt: Lt::open(&std::env::var("DSV4_CUBLASLT").unwrap_or_else(|_| "/opt/deps/cublas/nvidia/cu13/lib/libcublasLt.so.13".into()), sms)?,
                   probe: None, probes: vec![], snaps: None, snap_label: None, c128_sched: None, decode_in: None, last_logits_only: false, prune_rows: vec![], hc_dest: None, reduce: None, blas_ws: None, streams: Streams::default(), chunk: ChunkMeta::default(), serving: None, verify_in: None, verify_ready: false, dl: DecodeLayout::default(), dl_indexer: false, decode_w: 0, mtp_serving: None, rank: 0 })
    }

    pub fn open(ckpt: &'a crate::safetensors::Checkpoint, bundle: &std::path::Path, tables: &std::path::Path) -> Result<Self> {
        let dev = Device::open(0)?;
        let kernels: &'static Kernels = Box::leak(Box::new(Kernels::open(bundle)?));
        crate::runtime::install_redirect(kernels);
        let sms = dev.sm_count as i32;
        Ok(Model { dev, k: kernels, w: DeviceWeights::new(ckpt), tables: Tables::load(tables)?, cache: KvCache::new(SWA_PAGE_TOKENS)?,
                   trtllm_cubins: std::env::var("DSV4_TRTLLM_CUBINS").unwrap_or_else(|_| "/dsv4/kernels/trtllm_bmm".into()),
                   lt: Lt::open(&std::env::var("DSV4_CUBLASLT").unwrap_or_else(|_| "/opt/deps/cublas/nvidia/cu13/lib/libcublasLt.so.13".into()), sms)?,
                   probe: None, probes: vec![], snaps: None, snap_label: None, c128_sched: None, decode_in: None, last_logits_only: false, prune_rows: vec![], hc_dest: None, reduce: None, blas_ws: None, streams: Streams::default(), chunk: ChunkMeta::default(), serving: None, verify_in: None, verify_ready: false, dl: DecodeLayout::default(), dl_indexer: false, decode_w: 0, mtp_serving: None, rank: 0 })
    }

    /// Begin a cold start from a prepared package (prepare.rs): the CUDA context opens on one
    /// thread while this one loads the pinned manifest and starts the weight readers, and the
    /// kernel images and cuBLASLt are read on another. Nothing here waits for the GPU.
    pub fn start_prepared(dir: &std::path::Path, manifest_sha256: &str, bundle: &std::path::Path) -> Result<Startup> {
        let device = std::thread::spawn(|| Device::open(0));
        let load = crate::prepared::start(dir, manifest_sha256)?;
        let kernels: &'static Kernels = Box::leak(Box::new(Kernels::open(bundle)?));
        let lt_path = std::env::var("DSV4_CUBLASLT").unwrap_or_else(|_| "/opt/deps/cublas/nvidia/cu13/lib/libcublasLt.so.13".into());
        let images = { let lt_path = lt_path.clone(); std::thread::spawn(move || -> Result<Vec<(String, Vec<u8>)>> {
            let images = kernels.read_all()?;
            Lt::verify(&lt_path)?;
            Ok(images)
        }) };
        Ok(Startup { load, device, kernels, images, lt_path })
    }

    /// Greedy generation for one request (SGLang temperature 0): prefill `ids`, then decode one token
    /// per forward until `max_new_tokens` or a token in `eos`, with the reference's logprobs.
    /// Prefill `ids` in SGLang's chunks (stream-ordered, no host wait); returns the last chunk's batch
    /// and logits. A prompt longer than one chunk computes the last position's logits only (no input
    /// logprobs, as SGLang serves such requests here).
    fn prefill(&mut self, ids: &[i64], persist: &mut Arena, scratch: &mut Arena) -> Result<(Batch, u64)> {
        let s = self.s();
        let keep = self.last_logits_only;
        self.last_logits_only |= ids.len() > PREFILL_CHUNK;
        let mut last = None;
        for start in (0..ids.len()).step_by(PREFILL_CHUNK) {
            persist.reset();
            scratch.reset();
            let b = Batch::upload(&ids[start..(start + PREFILL_CHUNK).min(ids.len())], start, persist, s)?;
            let logits = self.forward(&b, persist, scratch, None)?;
            last = Some((b, logits));
        }
        self.last_logits_only = keep;
        last.ok_or_else(|| "empty prompt".to_string())
    }

    /// Greedy decoding as SGLang's overlap scheduler runs it, for serving without logprobs or stop
    /// tokens (ignore_eos): each step samples on the device (fp32 logits, ATen argmax) and writes the
    /// token straight into the next step's inputs, so the host enqueues step after step without
    /// waiting for any (the next step's positions are uploaded while the current one runs). Decode
    /// steps replay the CUDA graph as `generate` does. Returns the tokens, the time to the first
    /// token (ms, the prefill's token read on the host) and the mean inter-token latency over the
    /// decode steps (ms, wall time from the first token to the last).
    pub fn generate_overlapped(&mut self, ids: &[i64], max_new_tokens: usize) -> Result<(Vec<usize>, f64, f64)> {
        let s = self.s();
        // Serving state at a capacity covering this request (grown, graphs dropped, when it does not).
        let need = (ids.len() + max_new_tokens).max(4096);
        let reuse = self.serving.as_ref().map(|v| v.capacity >= need).unwrap_or(false);
        if reuse { self.cache.clear(s)?; } else {
            self.serving = None;
            self.cache = KvCache::new(need)?;
            let (pb, sb) = arena_bytes(PREFILL_CHUNK);
            self.serving = Some(Serving { capacity: need, persist: Arena::new(pb)?, scratch: Arena::new(sb)?, token: Buffer::alloc(8)?, graphs: vec![], eager: vec![], batch_graphs: vec![], batch_eager: vec![] });
        }
        self.enable_streams()?;
        let mut sv = self.serving.take().unwrap();
        let r = self.overlapped_steps(ids, max_new_tokens, &mut sv);
        self.serving = Some(sv);
        r
    }

    fn overlapped_steps(&mut self, ids: &[i64], max_new_tokens: usize, sv: &mut Serving) -> Result<(Vec<usize>, f64, f64)> {
        let s = self.s();
        let all_tokens = Buffer::alloc(max_new_tokens.max(1) * 8)?;
        s.synchronize()?;
        let t0 = std::time::Instant::now();
        let (b, logits) = self.prefill(ids, &mut sv.persist, &mut sv.scratch)?;
        s.synchronize()?;
        let first = crate::parity::argmax_bf16(&crate::cuda::download(logits.wrapping_add(((b.t - 1) * VOCAB * 2) as u64), VOCAB * 2)?);
        let ttft = t0.elapsed().as_secs_f64() * 1e3;
        let t1 = std::time::Instant::now();
        let mut pos = ids.len();
        let token = sv.token.ptr;
        for step in 1..max_new_tokens {
            let b = if step == 1 { self.decode_batch(first as i64, pos)? } else {
                // Everything but the token id, which the previous step wrote on the device.
                let h = self.decode_host_inputs(&[(Req::SINGLE, pos)])?;
                let from = self.dl.pos;
                crate::cuda::upload_async(self.decode_in.as_ref().unwrap().ptr + from as u64, &h[from..], s)?;
                self.decode_view(Req::SINGLE, pos)
            };
            let ids_in = self.decode_in.as_ref().unwrap().ptr + self.dl.ids as u64;
            let regime = self.dl_indexer;
            let step_fn = |m: &mut Self, persist: &mut Arena, scratch: &mut Arena| -> Result<()> {
                persist.reset();
                let logits = m.forward(&b, persist, scratch, None)?;
                let wide = scratch.alloc(VOCAB * 4)?;
                torch_ops::copy_bf16_to_f32(m.k, m.s(), wide, logits, VOCAB)?;
                torch_ops::argmax_f32_row(m.k, m.s(), wide, VOCAB, token)?;
                crate::cuda::copy_async(ids_in, token, 8, m.s())
            };
            if let Some((_, gr)) = sv.graphs.iter().find(|g| g.0 == regime) {
                gr.launch(s)?;
            } else if sv.eager.contains(&regime) {
                // Second step of a new regime: capture (every function it uses is loaded by now).
                let (gr, ()) = crate::cuda::Graph::capture(s, || step_fn(self, &mut sv.persist, &mut sv.scratch))?;
                gr.launch(s)?;
                sv.graphs.push((regime, gr));
            } else {
                step_fn(self, &mut sv.persist, &mut sv.scratch)?;
                sv.eager.push(regime);
            }
            crate::cuda::copy_async(all_tokens.ptr + (step * 8) as u64, token, 8, s)?;
            pos += 1;
        }
        s.synchronize()?;
        let itl = t1.elapsed().as_secs_f64() * 1e3 / (max_new_tokens.max(2) - 1) as f64;
        let mut tokens = vec![first];
        if max_new_tokens > 1 {
            let raw = crate::cuda::download(all_tokens.ptr, max_new_tokens * 8)?;
            tokens.extend(raw.chunks_exact(8).skip(1).map(|c| i64::from_le_bytes(c.try_into().unwrap()) as usize));
        }
        Ok((tokens, ttft, itl))
    }

    /// Serve `prompts` concurrently as SGLang schedules requests that arrive together and all fit:
    /// every prompt prefilled (in SGLang's chunks, one request at a time, its first token sampled on
    /// the device), then all decoded as one batch for `max_new_tokens - 1` steps (ignore_eos).
    pub fn generate_batch(&mut self, prompts: &[Vec<i64>], max_new_tokens: usize) -> Result<BatchRun> {
        let lens: Vec<usize> = prompts.iter().map(|p| p.len()).collect();
        let schedule = Step::prefill_then_decode(&lens, max_new_tokens);
        self.serve(prompts, max_new_tokens, &schedule)
    }

    /// Serve `prompts` (each to `max_new_tokens` tokens, ignore_eos) by running `schedule`, the
    /// forwards SGLang's scheduler chose: prefill chunks and decode steps of changing batches.
    /// Each request owns a range of full-pool pages and a request-pool row; its SWA pages come
    /// from a separate pool and go back once out of its window. Every token is sampled on the
    /// device and feeds the next step without a host wait. Returns each request's tokens and the
    /// timings.
    pub fn serve(&mut self, prompts: &[Vec<i64>], max_new_tokens: usize, schedule: &[Step]) -> Result<BatchRun> {
        let n = prompts.len();
        let s = self.s();
        let longest = prompts.iter().map(|p| p.len()).max().unwrap_or(0);
        // Pages per request, a multiple of 4 so every request's page-table row (4-byte ids) starts
        // 16-byte aligned, as SGLang's per-request tables do (Triton specializes on alignment).
        let w = ((longest + max_new_tokens).div_ceil(SWA_PAGE_TOKENS) + 1).div_ceil(4) * 4;
        // SWA pool: a prefill chunk's pages and its window, plus at most 3 pages per request (its
        // window, its compressor states, the page its next token starts).
        let swa_tokens = (3 * n + PREFILL_CHUNK.div_ceil(SWA_PAGE_TOKENS) + 3) * SWA_PAGE_TOKENS;
        let max_b = schedule.iter().map(|st| match st { Step::Decode { rows } => graph_batch(rows.len()), Step::Prefill { chunks } => chunks.len(), Step::Retract { .. } => 1 }).max().unwrap_or(1);
        let need = (n * w * SWA_PAGE_TOKENS, n + 1);
        let reuse = self.serving.as_ref().map(|v| v.capacity >= need.0 && self.cache.rows >= need.1 && self.decode_w == w
                                                  && self.cache.swa_pages >= 1 + swa_tokens / SWA_PAGE_TOKENS && v.token.bytes >= 8 * max_b).unwrap_or(false);
        if reuse { self.cache.clear(s)?; } else {
            self.serving = None;
            self.cache = KvCache::with_pools(need.0, need.1, Some(swa_tokens))?;
            self.decode_w = w;
            let (pb, sb) = arena_bytes(PREFILL_CHUNK);
            self.serving = Some(Serving { capacity: need.0, persist: Arena::new(pb)?, scratch: Arena::new(sb)?, token: Buffer::alloc(8 * max_b.max(8))?,
                                          graphs: vec![], eager: vec![], batch_graphs: vec![], batch_eager: vec![] });
        }
        // Decode inputs sized for the largest batch before any graph is captured on them.
        let dl_max = DecodeLayout::new(max_b, w, self.sms());
        if self.decode_in.as_ref().map(|d| d.bytes < dl_max.bytes).unwrap_or(true) {
            if let Some(v) = self.serving.as_mut() { v.batch_graphs.clear(); v.batch_eager.clear(); }
            self.decode_in = Some(Buffer::alloc(dl_max.bytes)?);
        }
        self.enable_streams()?;
        let keep = self.last_logits_only;
        self.last_logits_only = true;
        let mut sv = self.serving.take().unwrap();
        let r = self.serve_steps(prompts, max_new_tokens, schedule, w, &mut sv);
        self.serving = Some(sv);
        self.last_logits_only = keep;
        r
    }

    fn serve_steps(&mut self, prompts: &[Vec<i64>], max_new_tokens: usize, schedule: &[Step], w: usize, sv: &mut Serving) -> Result<BatchRun> {
        let (s, n) = (self.s(), prompts.len());
        let reqs: Vec<Req> = (0..n).map(|i| Req { page0: 1 + i * w, rid: 1 + i as i64 }).collect();
        // Every sampled token gets a slot in `log`, in forward order; `owner` says whose it is
        // (usize::MAX: a sample nobody keeps, a packed chunk that does not end its request).
        let slots: usize = schedule.iter().map(|st| match st { Step::Decode { rows } => rows.len(), Step::Prefill { chunks } => chunks.len(), Step::Retract { .. } => 0 }).sum();
        let log = Buffer::alloc(8 * slots.max(1))?;
        let (mut used, mut owner) = (0usize, Vec::with_capacity(slots));
        // Each request's tokens (their log slots, in order) and the latest one's address.
        let mut held: Vec<Vec<usize>> = vec![vec![]; n];
        let mut last: Vec<Option<u64>> = vec![None; n];
        let mut generated = vec![0usize; n];
        // The rows of the decode step whose samples are in the decode inputs' token ids.
        let mut fed: Option<Vec<usize>> = None;
        let last_prefill = schedule.iter().rposition(|st| matches!(st, Step::Prefill { .. }));
        // Debug (DSV4_BATCH_CHECK=<gold dir>): compare every forward's logits and samples with the
        // reference's (eager steps); the schedule's k-th forward (Retract steps run none) is gold fwd-k.
        let check = std::env::var("DSV4_BATCH_CHECK").ok().map(std::path::PathBuf::from);
        let mut fwd = 0usize;
        // Decode batches padded as SGLang's graph server runs them (its eager server, whose golds
        // the check compares, runs the exact batch; padding can change the rows' low bits: at 76
        // requests (graph 80) it does). DSV4_PAD_BATCH=0/1 overrides.
        let pad = std::env::var("DSV4_PAD_BATCH").map(|v| v != "0").unwrap_or(check.is_none());
        let token = sv.token.ptr;
        s.synchronize()?;
        let t0 = std::time::Instant::now();
        let mut prefill_s = 0.0;
        let mut decode_steps = 0usize;
        for (k, st) in schedule.iter().enumerate() {
            match st {
                Step::Prefill { chunks } => {
                    let mut segs = Vec::with_capacity(chunks.len());
                    let mut ids: Vec<i64> = Vec::new();
                    // Batch positions whose ids are tokens the request sampled (device slots).
                    let mut patches: Vec<(usize, u64)> = Vec::new();
                    let mut ends = Vec::with_capacity(chunks.len());
                    for c in chunks {
                        let (i, r) = (c.req, reqs[c.req]);
                        let (plen, end) = (prompts[i].len(), c.start + c.len);
                        if end > plen + generated[i] { return Err(format!("step {k}: prefill past request {i}'s sequence")); }
                        self.cache.swa_hold(&r, c.start, end)?;
                        for p in c.start..end {
                            if p < plen { ids.push(prompts[i][p]); } else { patches.push((ids.len(), log.ptr + (8 * held[i][p - plen]) as u64)); ids.push(0); }
                        }
                        segs.push(Seg { req: r, start: c.start, len: c.len });
                        ends.push(end == plen + generated[i]);
                    }
                    sv.persist.reset();
                    sv.scratch.reset();
                    fwd += 1;
                    let b = Batch::upload_segs(&ids, &segs, &mut sv.persist, s)?;
                    for &(j, src) in &patches { crate::cuda::copy_async(b.ids + (8 * j) as u64, src, 8, s)?; }
                    let logits = self.forward(&b, &mut sv.persist, &mut sv.scratch, None)?;
                    for c in chunks { self.cache.swa_evict(&reqs[c.req], c.start + c.len); }
                    if ends.iter().any(|&e| e) {
                        // One request: its last row (`logits` indexes rows of the chunk); several:
                        // [requests, vocab].
                        let nr = chunks.len();
                        let wide = sv.scratch.alloc(nr * VOCAB * 4)?;
                        let first = log.ptr + (8 * used) as u64;
                        if nr == 1 {
                            torch_ops::copy_bf16_to_f32(self.k, s, wide, logits.wrapping_add(((b.t - 1) * VOCAB * 2) as u64), VOCAB)?;
                            torch_ops::argmax_f32_row(self.k, s, wide, VOCAB, first)?;
                        } else {
                            torch_ops::copy_bf16_to_f32(self.k, s, wide, logits, nr * VOCAB)?;
                            torch_ops::argmax_f32_rows(self.k, s, wide, nr, VOCAB, first)?;
                        }
                        if let Some(dir) = &check { check_forward(dir, fwd, wide, nr, first)?; }
                        for (j, c) in chunks.iter().enumerate() {
                            let slot = first + (8 * j) as u64;
                            if ends[j] {
                                held[c.req].push(used + j);
                                last[c.req] = Some(slot);
                                generated[c.req] += 1;
                                owner.push(c.req);
                            } else { owner.push(usize::MAX); }
                        }
                        used += nr;
                    }
                    if Some(k) == last_prefill {
                        s.synchronize()?;
                        prefill_s = t0.elapsed().as_secs_f64();
                    }
                }
                Step::Retract { req, keep } => {
                    // Its cache is gone (the SWA pages back to the pool; the rest is rewritten by
                    // its new prefill); the tokens past `keep` are dropped.
                    let i = *req;
                    self.cache.swa_release(&reqs[i]);
                    if *keep > held[i].len() { return Err(format!("step {k}: request {i} keeps {keep} of {} tokens", held[i].len())); }
                    for &slot in &held[i][*keep..] { owner[slot] = usize::MAX; }
                    held[i].truncate(*keep);
                    generated[i] = *keep;
                    last[i] = held[i].last().map(|&x| log.ptr + (8 * x) as u64);
                    fed = None;
                }
                Step::Decode { rows } => {
                    fwd += 1;
                    let bs = rows.len();
                    let mut place = Vec::with_capacity(bs);
                    for &i in rows {
                        if generated[i] == 0 || generated[i] >= max_new_tokens { return Err(format!("step {k}: request {i} decodes with {} tokens", generated[i])); }
                        place.push((reqs[i], prompts[i].len() + generated[i] - 1));
                    }
                    for &(r, pos) in &place { self.cache.swa_hold(&r, pos, pos + 1)?; }
                    let nb = if pad { graph_batch(bs) } else { bs };
                    place.resize(nb, (Req { page0: 0, rid: 0 }, 0));
                    let h = self.decode_host_inputs(&place)?;
                    let (din, dl) = (self.decode_in.as_ref().unwrap().ptr, self.dl);
                    // Everything but the token ids.
                    crate::cuda::upload_async(din + dl.pos as u64, &h[dl.pos..], s)?;
                    // The token ids: the previous decode step wrote its samples there when the rows
                    // are the same; otherwise each row's latest token, runs of consecutive slots
                    // in one copy.
                    if fed.as_deref() != Some(rows.as_slice()) {
                        if nb > bs { crate::cuda::memset_async(din + (dl.ids + 8 * bs) as u64, 0, 8 * (nb - bs), s)?; }
                        let mut a = 0;
                        while a < bs {
                            let src = last[rows[a]].ok_or("decode before the request's prefill")?;
                            let mut e = a + 1;
                            while e < bs && last[rows[e]] == Some(src + (8 * (e - a)) as u64) { e += 1; }
                            crate::cuda::copy_async(din + (dl.ids + 8 * a) as u64, src, 8 * (e - a), s)?;
                            a = e;
                        }
                    }
                    let b = self.decode_view(place[0].0, place[0].1);
                    let regime = self.dl_indexer;
                    let ids_in = din + dl.ids as u64;
                    let step_fn = |m: &mut Self, persist: &mut Arena, scratch: &mut Arena| -> Result<u64> {
                        persist.reset();
                        let logits = m.forward(&b, persist, scratch, None)?;
                        let wide = scratch.alloc(nb * VOCAB * 4)?;
                        torch_ops::copy_bf16_to_f32(m.k, m.s(), wide, logits, nb * VOCAB)?;
                        torch_ops::argmax_f32_rows(m.k, m.s(), wide, nb, VOCAB, token)?;
                        crate::cuda::copy_async(ids_in, token, 8 * nb, m.s())?;
                        Ok(wide)
                    };
                    // Graphs are keyed by batch size and indexer regime: the first step of a key
                    // runs eagerly (loading every function), the second captures.
                    let key = (nb, regime);
                    if let Some(dir) = &check {
                        let wide = step_fn(self, &mut sv.persist, &mut sv.scratch)?;
                        check_forward(dir, fwd, wide, bs, token)?;
                    } else if let Some(gr) = sv.batch_graphs.iter().find(|g| g.0 == key).map(|g| &g.1) { gr.launch(s)?; }
                    else if sv.batch_eager.contains(&key) {
                        let (gr, _) = crate::cuda::Graph::capture(s, || step_fn(self, &mut sv.persist, &mut sv.scratch))?;
                        gr.launch(s)?;
                        sv.batch_graphs.push((key, gr));
                    } else {
                        step_fn(self, &mut sv.persist, &mut sv.scratch)?;
                        sv.batch_eager.push(key);
                    }
                    crate::cuda::copy_async(log.ptr + (8 * used) as u64, token, 8 * bs, s)?;
                    for (j, &i) in rows.iter().enumerate() {
                        last[i] = Some(log.ptr + (8 * (used + j)) as u64);
                        held[i].push(used + j);
                        owner.push(i);
                        generated[i] += 1;
                        if generated[i] == max_new_tokens { self.cache.swa_release(&reqs[i]); } else { self.cache.swa_evict(&reqs[i], place[j].1 + 1); }
                    }
                    used += bs;
                    fed = Some(rows.clone());
                    decode_steps += 1;
                }
            }
        }
        s.synchronize()?;
        let wall_s = t0.elapsed().as_secs_f64();
        let raw = crate::cuda::download(log.ptr, used * 8)?;
        let mut tokens = vec![Vec::with_capacity(max_new_tokens); n];
        for (c, &i) in raw.chunks_exact(8).zip(&owner) { if i != usize::MAX { tokens[i].push(i64::from_le_bytes(c.try_into().unwrap()) as usize); } }
        Ok(BatchRun { tokens, prefill_s, decode_s: wall_s - prefill_s, decode_steps, wall_s, decode_tokens: 0 })
    }

    pub fn generate(&mut self, ids: &[i64], max_new_tokens: usize, eos: &[usize]) -> Result<Generation> {
        self.serving = None; // its graphs read the cache replaced here
        self.cache = KvCache::new((ids.len() + max_new_tokens).max(4096))?;
        self.enable_streams()?;
        let (persist_bytes, scratch_bytes) = arena_bytes(ids.len().min(PREFILL_CHUNK));
        let mut persist = Arena::new(persist_bytes)?;
        let mut scratch = Arena::new(scratch_bytes)?;
        let mut g = Generation { tokens: vec![], logprobs: vec![], ms: vec![], token_ms: vec![], forward_ms: vec![] };
        let mut pos = 0usize;
        let mut next = ids.to_vec();
        // Decode as SGLang does: one CUDA graph, captured on the second decode step (after an eager
        // step has loaded every function) and replayed for every later one (DSV4_DECODE_GRAPH=0: eager).
        let graphs = std::env::var("DSV4_DECODE_GRAPH").map(|v| v != "0").unwrap_or(true) && std::env::var_os("DSV4_STAGE_TIMES").is_none();
        // The graph is keyed by whether the C4 indexer runs (more than 512 compressed entries); when
        // that flips, one eager step runs and the next step captures again.
        let mut graph: Option<(crate::cuda::Graph, u64, bool)> = None;
        let mut eager_regime: Option<bool> = None;
        // Whether the next step's inputs (all but its token id) are already enqueued.
        let mut prepared = false;
        let s = self.s();
        for step in 0..max_new_tokens {
            let t0 = std::time::Instant::now();
            let (b, logits) = if step == 0 {
                let (b, logits) = self.prefill(ids, &mut persist, &mut scratch)?;
                pos = b.start;
                (b, logits)
            } else {
                let b = if prepared {
                    let base = self.decode_in.as_ref().unwrap().ptr;
                    crate::cuda::upload_async(base + self.dl.ids as u64, &next[0].to_le_bytes(), s)?;
                    self.decode_view(Req::SINGLE, pos)
                } else { self.decode_batch(next[0], pos)? };
                prepared = false;
                let regime = self.dl_indexer;
                if graph.as_ref().map(|g| g.2 != regime).unwrap_or(false) { graph = None; }
                if graphs && step >= 2 && (graph.is_some() || eager_regime == Some(regime)) {
                    if graph.is_none() {
                        persist.reset();
                        let (gr, logits) = crate::cuda::Graph::capture(s, || self.forward(&b, &mut persist, &mut scratch, None))?;
                        graph = Some((gr, logits, regime));
                    }
                    let (gr, logits, _) = graph.as_ref().unwrap();
                    gr.launch(s)?;
                    (b, *logits)
                } else {
                    persist.reset();
                    let logits = self.forward(&b, &mut persist, &mut scratch, None)?;
                    eager_regime = Some(regime);
                    (b, logits)
                }
            };
            // While the step runs: the next step's inputs except its token id (stream-ordered after
            // this step, which still reads the current ones).
            if step >= 1 && step + 1 < max_new_tokens {
                let h = self.decode_host_inputs(&[(Req::SINGLE, pos + 1)])?;
                let (base, from) = (self.decode_in.as_ref().unwrap().ptr, self.dl.pos);
                crate::cuda::upload_async(base + from as u64, &h[from..], s)?;
                prepared = true;
            }
            s.synchronize()?;
            let t_forward = t0.elapsed().as_secs_f64() * 1e3;
            let last = crate::cuda::download(logits + ((b.t - 1) * VOCAB * 2) as u64, VOCAB * 2)?;
            let token = crate::parity::argmax_bf16(&last);
            g.token_ms.push(t0.elapsed().as_secs_f64() * 1e3);
            g.forward_ms.push(t_forward);
            scratch.reset();
            let lp = if step == 0 && ids.len() <= PREFILL_CHUNK && !self.last_logits_only { self.logprobs(logits, b.t, Some(ids), token, &mut scratch)? }
                     else { self.logprobs(logits.wrapping_add(((b.t - 1) * VOCAB * 2) as u64), 1, None, token, &mut scratch)? };
            g.logprobs.push(lp);
            if step < 2 && std::env::var_os("DSV4_STAGE_TIMES").is_some() {
                println!("{}", serde_json::json!({"step": step, "forward_ms": t_forward, "with_logprobs_ms": t0.elapsed().as_secs_f64() * 1e3}));
            }
            g.tokens.push(token);
            g.ms.push(t0.elapsed().as_millis());
            pos += b.t;
            next = vec![token as i64];
            if eos.contains(&token) { break; }
        }
        // Diagnostics (DSV4_GRAPH_BENCH=N): N back-to-back replays of the decode graph, no host work
        // between them: the pure device time per decode step. (Replays rewrite the last step's cache
        // slots with the same values, so run it only after generation.)
        if let (Some((gr, _, _)), Some(n)) = (graph.as_ref(), std::env::var("DSV4_GRAPH_BENCH").ok().and_then(|v| v.parse::<usize>().ok())) {
            s.synchronize()?;
            let t = std::time::Instant::now();
            for _ in 0..n { gr.launch(s)?; }
            s.synchronize()?;
            println!("{}", serde_json::json!({"graph_replay_ms": t.elapsed().as_secs_f64() * 1e3 / n as f64, "replays": n}));
        }
        Ok(g)
    }

    /// Debug: compare `bytes` at `ptr` with the reference snapshot of the `nth` launch named
    /// `fragment` under the current snap label (layer `layer`'s self_attn), parameter word (param, word).
    #[allow(clippy::too_many_arguments)]
    fn snap(&self, layer: usize, what: &str, fragment: &str, nth: usize, param: usize, word: usize, ptr: u64, bytes: usize) -> Result<()> {
        let (Some(snaps), Some(label)) = (&self.snaps, &self.snap_label) else { return Ok(()) };
        if self.probe != Some(layer) { return Ok(()); }
        self.s().synchronize()?;
        let full = format!("{label}|model.layers.{layer}.self_attn");
        match snaps.find(&full, fragment, nth).and_then(|l| snaps.read(&l, param, word, "out")) {
            Ok(snap) => { crate::snapshots::check_prefix(what, &crate::cuda::download(ptr, bytes)?, &snap); }
            Err(e) => println!("{}", serde_json::json!({"snapshot": what, "error": e})),
        }
        Ok(())
    }

    fn s(&self) -> Stream { self.streams.cur.unwrap_or(self.dev.stream) }
    fn side(&self, i: usize) -> Option<Stream> { self.streams.side.get(i).copied() }

    /// Create the side streams (each with its own cuBLASLt workspace). DSV4_STREAMS=0 keeps one stream.
    pub fn enable_streams(&mut self) -> Result<()> {
        if !self.streams.side.is_empty() || std::env::var("DSV4_STREAMS").map(|v| v == "0").unwrap_or(false) { return Ok(()); }
        for _ in 0..3 {
            let st = Stream::new()?;
            self.lt.add_stream(st)?;
            self.streams.side.push(st);
        }
        self.streams.events = (0..64).map(|_| crate::cuda::Event::new()).collect::<Result<_>>()?;
        Ok(())
    }

    fn event(&self) -> &crate::cuda::Event {
        let i = self.streams.next.get();
        self.streams.next.set((i + 1) % self.streams.events.len());
        &self.streams.events[i]
    }

    /// Start side stream `to` after the work enqueued so far on the current stream.
    fn fork(&self, to: Option<Stream>) -> Result<()> {
        match to { Some(st) => crate::cuda::stream_wait(self.s(), st, self.event()), None => Ok(()) }
    }

    /// Continue the current stream after everything enqueued on `from`.
    fn join(&self, from: Option<Stream>) -> Result<()> {
        match from { Some(st) => crate::cuda::stream_wait(st, self.s(), self.event()), None => Ok(()) }
    }

    /// Run `f` with its launches on `st` (None: the current stream).
    fn on<T>(&mut self, st: Option<Stream>, f: impl FnOnce(&mut Self) -> Result<T>) -> Result<T> {
        let prev = self.streams.cur;
        if st.is_some() { self.streams.cur = st; }
        let r = f(self);
        self.streams.cur = prev;
        r
    }
    fn sms(&self) -> usize { self.dev.sm_count as usize }

    /// Embedding + mHC broadcast + the first layer's attention mHC pre (DeepGEMM prenorm partials).
    /// Complete a sum over the TP ranks in place (`ptr`: this rank's bf16 partial). No-op on one GPU.
    pub fn reduce(&mut self, ptr: u64, bytes: usize) -> Result<()> {
        let s = self.s();
        match &mut self.reduce {
            None => Ok(()),
            Some(Reduce::Shm { dir, rank, ranks, next, gold, differ, seconds }) => {
                let k = *next;
                *next += 1;
                if crate::runtime::dry() { return Ok(()); }
                s.synchronize()?;
                let started = std::time::Instant::now();
                let mine = crate::cuda::download(ptr, bytes)?;
                let sum = shm_exchange(dir, &format!("ar{k:04}"), *rank, *ranks, &mine, |parts| {
                    let n = bytes / 2;
                    let mut acc = vec![0f32; n];
                    for part in parts {
                        for (a, c) in acc.iter_mut().zip(part.chunks_exact(2)) { *a += crate::dense_prepare::bf16_to_f32(u16::from_le_bytes([c[0], c[1]])); }
                    }
                    acc.iter().flat_map(|&v| crate::dense_prepare::f32_to_bf16(v).to_le_bytes()).collect()
                })?;
                if let Some(g) = gold {
                    if let Ok(meta) = g.meta(&format!("ar-{k:04}.out")) {
                        if meta.sha256 != crate::gold::sha(&sum) {
                            if differ.is_empty() && *rank == 0 { println!("{}", serde_json::json!({"first_sum_differs_from_reference": k})); }
                            differ.push(k);
                        }
                    }
                }
                crate::cuda::upload_async(ptr, &sum, s)?;
                s.synchronize()?;
                *seconds += started.elapsed().as_secs_f64();
                Ok(())
            }
            Some(Reduce::Gold { gold, all, next, results }) => {
                let k = *next;
                *next += 1;
                if crate::runtime::dry() { return Ok(()); }
                s.synchronize()?;
                let native = crate::cuda::download(ptr, bytes)?;
                let mut c = crate::gold::compare(gold, &format!("ar-{k:04}.in"), &native)?;
                if !c.equal {
                    if let Some(r) = all.iter().position(|g| g.meta(&format!("ar-{k:04}.in")).map(|m| m.sha256 == c.native_sha256).unwrap_or(false)) {
                        c.equal = true;
                        c.name = format!("ar-{k:04}.in (reference rank {r})");
                    }
                }
                println!("{}", serde_json::to_string(&c).unwrap());
                if !c.equal {
                    // How far off: differing bf16 elements, the largest difference and the first few.
                    let native = crate::cuda::download(ptr, bytes)?;
                    let want = gold.bytes(&format!("ar-{k:04}.in"))?;
                    let f = |b: &[u8], i: usize| crate::dense_prepare::bf16_to_f32(u16::from_le_bytes([b[2 * i], b[2 * i + 1]]));
                    let n = bytes.min(want.len()) / 2;
                    let diff: Vec<usize> = (0..n).filter(|&i| native[2 * i..2 * i + 2] != want[2 * i..2 * i + 2]).collect();
                    let max = diff.iter().map(|&i| (f(&native, i) - f(&want, i)).abs()).fold(0f32, f32::max);
                    let zeros = (0..n).filter(|&i| f(&native, i) == 0.0).count();
                    let first: Vec<_> = diff.iter().take(6).map(|&i| serde_json::json!([i, f(&native, i), f(&want, i)])).collect();
                    println!("{}", serde_json::json!({"ar": k, "elements": n, "gold_bytes": want.len(), "differ": diff.len(), "max_abs_diff": max, "native_zeros": zeros, "first": first}));
                }
                results.push(c);
                let sum = gold.bytes(&format!("ar-{k:04}.out"))?;
                if sum.len() != bytes { return Err(format!("ar-{k:04}.out has {} bytes, the partial {bytes}", sum.len())); }
                crate::cuda::upload_async(ptr, &sum, s)?;
                s.synchronize()
            }
        }
    }

    pub fn embed(&mut self, b: &Batch, persist: &mut Arena, scratch: &mut Arena) -> Result<(Hc, u64)> {
        let (t, s, k) = (b.t, self.s(), self.k);
        let embed = scratch.alloc(t * HIDDEN * 2)?;
        if TP > 1 {
            // SGLang's vocab-parallel embedding (Triton; this rank's vocabulary range is compiled
            // in): rows of tokens outside the range are zero, and the all-reduce completes them.
            crate::runtime::triton(k, s, "_vocab_parallel_embedding_kernel", [t as u32, HIDDEN.div_ceil(1024) as u32, 1], 8, 0,
                &[crate::runtime::TArg::Ptr(b.ids), crate::runtime::TArg::Ptr(self.w.get("model.embed_tokens.weight")?), crate::runtime::TArg::Ptr(embed)], "")?;
            self.reduce(embed, t * HIDDEN * 2)?;
        } else {
            torch_ops::embedding_bf16(k, s, embed, self.w.get("model.embed_tokens.weight")?, b.ids, t, VOCAB, HIDDEN)?;
        }
        let mut hc = Hc::new(t, persist)?;
        let set = hc.sets[0];
        let residual = set.residual;
        mhc::hc_broadcast(k, s, embed, residual, t, HIDDEN)?;
        let splits = deepgemm::mhc_pre_splits(t, HC * HIDDEN, self.sms());
        let d = scratch.alloc(splits * t * MIX * 4)?;
        let sq = scratch.alloc(splits * t * 4)?;
        deepgemm::tf32_hc_prenorm_gemm(k, s, residual, self.w.get("model.layers.0.hc_attn_fn")?, d, sq, t, MIX, HC * HIDDEN, splits)?;
        let pre = mhc::MhcPreOut { post_mix: set.post_mix, comb_mix: set.comb_mix, layer_input: set.layer_input };
        mhc::pre_big_fuse_with_norm(k, s, d, sq, self.w.get("model.layers.0.hc_attn_scale")?, self.w.get("model.layers.0.hc_attn_base")?,
                                    residual, &pre, self.w.get("model.layers.0.input_layernorm.weight")?, t, splits)?;
        hc.residual = residual;
        Ok((hc, pre.layer_input))
    }

    /// The previous sublayer's mHC post (residual += its output, mixed) fused with the next mHC
    /// pre (`which` = "attn" | "ffn" of `layer`); returns the next sublayer's normed input.
    pub fn post_pre(&mut self, layer: usize, which: &str, hc: &mut Hc, out: u64, t: usize, persist: &mut Arena, scratch: &mut Arena) -> Result<u64> {
        let (s, k) = (self.s(), self.k);
        let l = format!("model.layers.{layer}");
        let norm = if which == "attn" { "input_layernorm" } else { "post_attention_layernorm" };
        if t > 32 {
            // Beyond 32 tokens SGLang runs the unfused sequence: mhc_post, the prenorm GEMM, mhc_pre.
            let _ = &persist;
            let nxt = hc.next();
            let residual = nxt.residual;
            mhc::post(k, s, hc.comb_mix, hc.residual, hc.post_mix, out, residual, t)?;
            let splits = deepgemm::mhc_pre_splits(t, HC * HIDDEN, self.sms());
            let d = scratch.alloc(splits * t * MIX * 4)?;
            let sq = scratch.alloc(splits * t * 4)?;
            deepgemm::tf32_hc_prenorm_gemm(k, s, residual, self.w.get(&format!("{l}.hc_{which}_fn"))?, d, sq, t, MIX, HC * HIDDEN, splits)?;
            let pre = mhc::MhcPreOut { post_mix: nxt.post_mix, comb_mix: nxt.comb_mix, layer_input: nxt.layer_input };
            mhc::pre_big_fuse_with_norm(k, s, d, sq, self.w.get(&format!("{l}.hc_{which}_scale"))?, self.w.get(&format!("{l}.hc_{which}_base"))?,
                                        residual, &pre, self.w.get(&format!("{l}.{norm}.weight"))?, t, splits)?;
            hc.advance();
            return Ok(pre.layer_input);
        }
        let (_, split_k) = mhc::fused_post_pre_shape(t);
        let mixes = scratch.alloc(split_k * t * MIX * 4)?;
        let sqrsum = scratch.alloc(split_k * t * 4)?;
        let nxt = hc.next();
        let residual = nxt.residual;
        mhc::fused_post_pre_fma(k, s, hc.comb_mix, hc.residual, hc.post_mix, out, self.w.get(&format!("{l}.hc_{which}_fn"))?,
                                mixes, sqrsum, residual, t)?;
        let pre = mhc::MhcPreOut { post_mix: nxt.post_mix, comb_mix: nxt.comb_mix, layer_input: nxt.layer_input };
        mhc::pre_big_fuse_with_norm(k, s, mixes, sqrsum, self.w.get(&format!("{l}.hc_{which}_scale"))?, self.w.get(&format!("{l}.hc_{which}_base"))?,
                                    residual, &pre, self.w.get(&format!("{l}.{norm}.weight"))?, t, split_k)?;
        hc.advance();
        Ok(pre.layer_input)
    }

    /// q / kv projections shared by every attention kind: returns (q roped [t, 64, 512], qkv_a [t, 1536]).
    /// The third value is the FP8 q_lora and its packed scales (the C4 indexer's wq_b reads them too).
    fn attn_qkv(&mut self, layer: usize, x: u64, b: &Batch, a: &mut Arena) -> Result<(u64, u64, (u64, u64))> {
        let qkv_a = self.attn_qkv_a(layer, x, b, a)?;
        let lq = self.attn_q_lora(layer, qkv_a, b, a)?;
        let q_rot = self.attn_q_rot(layer, lq, b, a)?;
        Ok((q_rot, qkv_a, lq))
    }

    /// FP8 x and the wqkv_a projection: qkv_a [t, 1024 q_lora | 512 kv].
    fn attn_qkv_a(&mut self, layer: usize, x: u64, b: &Batch, a: &mut Arena) -> Result<u64> {
        let (t, s, k, sms) = (b.t, self.s(), self.k, self.sms());
        let l = format!("model.layers.{layer}.self_attn");
        let xq = a.alloc(t * HIDDEN)?;
        let xs = a.alloc(sgl::act_scale_bytes(t, HIDDEN))?;
        sgl::quant_fp8(k, s, x, HIDDEN, xq, xs, t, HIDDEN)?;
        self.snap(layer, "quant(x)", "per_token_group_quant", 0, 0, 3, xq, t * HIDDEN)?;
        let qkv_n = Q_LORA + HEAD_DIM;
        let qkv_a = a.alloc(t * qkv_n * 2)?;
        deepgemm::fp8_gemm_nt(k, s, xq, xs, self.w.get(&format!("{l}.wqkv_a.weight"))?, self.w.get(&format!("{l}.wqkv_a.weight_scale_inv"))?,
                              qkv_a, t, qkv_n, HIDDEN, sms)?;
        self.snap(layer, "wqkv_a", "fp8_fp4_gemm", 0, 9, 0, qkv_a, t * qkv_n * 2)?;
        Ok(qkv_a)
    }

    /// q_norm and the FP8 q_lora (with its packed scales).
    fn attn_q_lora(&mut self, layer: usize, qkv_a: u64, b: &Batch, a: &mut Arena) -> Result<(u64, u64)> {
        let (t, s, k) = (b.t, self.s(), self.k);
        let l = format!("model.layers.{layer}.self_attn");
        let q_lora = a.alloc(t * Q_LORA * 2)?;
        sgl::rmsnorm_1024(k, s, qkv_a, Q_LORA + HEAD_DIM, self.w.get(&format!("{l}.q_norm.weight"))?, q_lora, t, EPS)?;
        let lq = a.alloc(t * Q_LORA)?;
        let ls = a.alloc(sgl::act_scale_bytes(t, Q_LORA))?;
        sgl::quant_fp8(k, s, q_lora, Q_LORA, lq, ls, t, Q_LORA)?;
        Ok((lq, ls))
    }

    /// wq_b and the q norm + RoPE: q roped [t, 64, 512].
    fn attn_q_rot(&mut self, layer: usize, (lq, ls): (u64, u64), b: &Batch, a: &mut Arena) -> Result<u64> {
        let (t, s, k, sms) = (b.t, self.s(), self.k, self.sms());
        let l = format!("model.layers.{layer}.self_attn");
        let qn = HEADS * HEAD_DIM;
        let q = a.alloc(t * qn * 2)?;
        deepgemm::fp8_gemm_nt(k, s, lq, ls, self.w.get(&format!("{l}.wq_b.weight"))?, self.w.get(&format!("{l}.wq_b.weight_scale_inv"))?,
                              q, t, qn, Q_LORA, sms)?;
        self.snap(layer, "wq_b", "fp8_fp4_gemm", 1, 9, 0, q, t * qn * 2)?;
        let qs = HEAD_SLOTS * HEAD_DIM;
        let q_rot = a.alloc(t * qs * 2)?;
        if qs != qn { crate::cuda::memset_async(q_rot, 0, t * qs * 2, s)?; }
        sgl::fused_q_norm_rope(k, s, q, q_rot, self.tables.freqs(layer), b.positions, qn, qs, t, HEADS, EPS)?;
        self.snap(layer, "q_rot", "fused_q_norm_rope", 0, 0, 1, q_rot, t * qs * 2)?;
        Ok(q_rot)
    }

    /// Store this chunk's KV (norm + rope + FP8) into the layer's SWA cache.
    fn store_swa(&mut self, layer: usize, qkv_a: u64, b: &Batch, a: &mut Arena) -> Result<u64> {
        let l = format!("model.layers.{layer}.self_attn");
        let out_loc = if b.decode {
            self.decode_in.as_ref().ok_or("decode without decode inputs")?.ptr + self.dl.swa_slot as u64
        } else {
            let _ = a;
            self.chunk.swa_out
        };
        sgl::fused_k_norm_rope_flashmla(self.k, self.s(), qkv_a + (Q_LORA * 2) as u64, self.w.get(&format!("{l}.kv_norm.weight"))?,
                                        self.tables.freqs(layer), b.positions, out_loc, self.cache.swa_layer(layer), Q_LORA + HEAD_DIM, b.t, EPS)?;
        Ok(out_loc)
    }

    /// Output projections: inverse rope of the attention output, grouped wo_a, wo_b.
    fn attn_out(&mut self, layer: usize, o: u64, b: &Batch, a: &mut Arena) -> Result<u64> {
        let (t, s, k, sms) = (b.t, self.s(), self.k, self.sms());
        let l = format!("model.layers.{layer}.self_attn");
        let qn = HEADS * HEAD_DIM;
        // `o` is [t, HEAD_SLOTS, 512]: this rank's heads are the first HEADS of each token.
        let qs = HEAD_SLOTS * HEAD_DIM;
        self.snap(layer, "attn_o", "deepseek_rope_kernel", 0, 0, 0, o, t * qs * 2)?;
        sgl::inverse_rope_q(k, s, o + ((HEAD_DIM - ROPE) * 2) as u64, self.tables.freqs(layer), b.positions, qs * 2, HEAD_DIM * 2, HEADS, t)?;
        let (groups, o_lora) = (O_GROUPS, O_LORA);
        let gd = qn / groups;
        let oq = a.alloc(t * qn)?;
        let os = a.alloc(groups * gd.div_ceil(512) * t.div_ceil(4) * 4 * 4)?;
        sgl::wo_a_group_quant(k, s, o, oq, os, t, groups, gd, qs)?;
        let wo_a = a.alloc(t * groups * o_lora * 2)?;
        deepgemm::fp8_bmm_token_major(k, s, oq, os, self.w.get(&format!("{l}.wo_a.weight"))?, self.w.get(&format!("{l}.wo_a.weight_scale_inv"))?,
                                      wo_a, groups, t, o_lora, gd, sms)?;
        let bk = groups * o_lora;
        let bq = a.alloc(t * bk)?;
        let bs = a.alloc(sgl::act_scale_bytes(t, bk))?;
        sgl::quant_fp8(k, s, wo_a, bk, bq, bs, t, bk)?;
        let y = a.alloc(t * HIDDEN * 2)?;
        deepgemm::fp8_gemm_nt(k, s, bq, bs, self.w.get(&format!("{l}.wo_b.weight"))?, self.w.get(&format!("{l}.wo_b.weight_scale_inv"))?,
                              y, t, HIDDEN, bk, sms)?;
        Ok(y)
    }

    /// Sliding-window attention (compress ratio 0) for a prefill chunk: FlashMLA sparse prefill over
    /// each token's window, read from the dequantized SWA cache.
    pub fn attention_swa(&mut self, layer: usize, x: u64, b: &Batch, a: &mut Arena) -> Result<u64> {
        let (q_rot, qkv_a, _) = self.attn_qkv(layer, x, b, a)?;
        self.store_swa(layer, qkv_a, b, a)?;
        let o = self.sparse_prefill(layer, q_rot, b, a, None, None)?;
        self.attn_out(layer, o, b, a)
    }

    /// C4 attention (even layers from 2) for a prefill chunk: the sparse indexer (its compressor into
    /// the index-K cache, then each query's top-512 compressed tokens), the C4 compressor (cuBLASLt
    /// wkv_gate, flash_c4 compress + state write, norm/rope/FP8 store into the compressed cache), then
    /// sparse prefill over [the query's top-k compressed tokens | its window].
    pub fn attention_c4(&mut self, layer: usize, x: u64, b: &Batch, a: &mut Arena) -> Result<u64> {
        let (t, s, k) = (b.t, self.s(), self.k);
        let l = format!("model.layers.{layer}.self_attn");
        let seq_len = b.start + t;
        let (q_rot, qkv_a, lq) = self.attn_qkv(layer, x, b, a)?;
        self.store_swa(layer, qkv_a, b, a)?;
        // Compress / write plans and output slots (shared by the indexer's and the main compressor).
        let m = self.chunk.c4;
        let topk = self.c4_indexer(layer, x, lq, b, (m.plan_c, m.num_c), (m.plan_w, m.num_w), m.out_loc, a)?;
        // Compressor.
        let kv_score = a.alloc(t * 4 * HEAD_DIM * 4)?;
        self.lt.linear(s, t, 4 * HEAD_DIM, HIDDEN, x, self.w.get(&format!("{l}.compressor.wkv_gate.weight"))?, kv_score, true)?;
        let comp = a.alloc(m.num_c.max(1) * HEAD_DIM * 4)?;
        compress::c4_prefill(k, s, HEAD_DIM, self.cache.c4_state_layer(layer), kv_score, comp, self.w.get(&format!("{l}.compressor.ape"))?,
                             m.plan_c, m.plan_w, m.num_c, m.num_w)?;
        compress::norm_rope_store_flashmla(k, s, comp, m.plan_c, self.w.get(&format!("{l}.compressor.norm.weight"))?, self.tables.freqs(layer),
                                           m.out_loc, self.cache.c4_layer(layer), 4, m.num_c, 6, false)?;
        let _ = seq_len;
        let o = self.sparse_prefill(layer, q_rot, b, a, Some(4), topk)?;
        self.attn_out(layer, o, b, a)
    }

    /// The compress and write plans of `segs` for one ratio, per request in order with ragged ids
    /// counting the batch's tokens (plan_compress_prefill's CPU path; a plan without buffered
    /// reads keeps its batch index in read_page_1, as plan_compress_prefill_kernel_1 leaves it),
    /// and each token's compressed output slot (0: none).
    fn ratio_plans(&self, segs: &[Seg], ratio: usize) -> (Vec<compress::PlanC>, Vec<compress::PlanW>, Vec<i64>) {
        let (mut pc, mut pw) = (Vec::new(), Vec::new());
        let mut out_loc: Vec<i64> = Vec::new();
        let mut off = 0usize;
        for (r, g) in segs.iter().enumerate() {
            let seq = g.start + g.len;
            let (c, w) = compress::prefill_plan(ratio, seq as i32, g.len as i32, if ratio == 4 { 1 } else { g.req.rid }, |p| self.cache.swa_slot(&g.req, p as usize));
            pc.extend(c.into_iter().map(|mut c| { c.ragged_id += off as u16; if c.read_page_0 == -1 { c.read_page_1 = r as i32; } c }));
            pw.extend(w.into_iter().map(|mut w| { w.ragged_id += off as u32; w }));
            let live = seq / ratio;
            out_loc.extend((g.start..g.start + g.len).map(|p| if (p + 1) % ratio == 0 {
                let e = (p + 1) / ratio - 1;
                (if ratio == 4 { g.req.c4_slot(e) } else if e < live { g.req.c128_slot(e) } else { 0 }) as i64
            } else { 0 }));
            off += g.len;
        }
        (pc, pw, out_loc)
    }

    /// A speculative verify's metadata (the compressors' plans and output slots, all a verify
    /// reads of `ChunkMeta`) uploaded into the fixed verify region at fixed offsets, so a verify
    /// forward replays as a CUDA graph (one per plan-count pattern, see `VerifyKey`).
    pub fn verify_meta(&mut self, segs: &[Seg]) -> Result<VerifyKey> {
        const SLOT: usize = 32 << 10;  // plans and output slots of up to 2048 verify rows
        if self.verify_in.is_none() { self.verify_in = Some(Buffer::alloc(6 * SLOT)?); }
        let base = self.verify_in.as_ref().unwrap().ptr;
        let mut h = vec![0u8; 6 * SLOT];
        let mut key = VerifyKey::default();
        for (i, ratio) in [4usize, 128].into_iter().enumerate() {
            let (pc, pw, out_loc) = self.ratio_plans(segs, ratio);
            let parts = [compress::plan_c_bytes(&pc), compress::plan_w_bytes(&pw), out_loc.iter().flat_map(|x| x.to_le_bytes()).collect()];
            for (j, bytes) in parts.iter().enumerate() {
                if bytes.len() > SLOT { return Err("verify metadata exceeds its slot".into()); }
                let off = (3 * i + j) * SLOT;
                h[off..off + bytes.len()].copy_from_slice(bytes);
            }
            let m = RatioMeta { plan_c: base + (3 * i * SLOT) as u64, num_c: pc.len(), plan_w: base + ((3 * i + 1) * SLOT) as u64, num_w: pw.len(),
                                out_loc: base + ((3 * i + 2) * SLOT) as u64, ..Default::default() };
            if ratio == 4 { self.chunk.c4 = m; key.c4 = (pc.len(), pw.len()); } else { self.chunk.c128 = m; key.c128 = (pc.len(), pw.len()); }
        }
        crate::cuda::upload_async(base, &h, self.s())?;
        self.verify_ready = true;
        Ok(key)
    }

    /// Build and upload the chunk's metadata (`ChunkMeta`) into `a`: per segment (request) in
    /// order, as SGLang's sparse-prefill metadata and compress plans lay out a packed batch.
    pub fn chunk_meta(&self, b: &Batch, a: &mut Arena) -> Result<ChunkMeta> {
        let t = b.t;
        let segs = &b.segs;
        let nr = segs.len();
        let mut buf: Vec<u8> = Vec::new();
        let mut put = |bytes: &[u8]| -> u64 { let off = buf.len().div_ceil(512) * 512; buf.resize(off, 0); buf.extend_from_slice(bytes); off as u64 };
        let i64s = |v: &[i64]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
        let seq: Vec<usize> = segs.iter().map(|g| g.start + g.len).collect();
        // Window union per request (build_swa_token_ids): [seq - gather, seq).
        let gather: Vec<usize> = segs.iter().zip(&seq).map(|(g, &sl)| sl.min(g.len + WINDOW - 1)).collect();
        let swa_off: Vec<usize> = gather.iter().scan(0, |acc, &g| { let o = *acc; *acc += g; Some(o) }).collect();
        let qsl: Vec<i32> = std::iter::once(0).chain(segs.iter().scan(0, |acc, g| { *acc += g.len; Some(*acc as i32) })).collect();
        // Combine metadata: query_start_loc [nr + 1], seq_lens, gather_lens, compressed_base,
        // swa_base [nr], each array in its own 16-byte aligned slot.
        let slot = (4 * (nr + 1)).div_ceil(16) * 16;
        let combine = |cbase: &[i32], sbase: &[i32]| -> Vec<u8> {
            let mut v = vec![0u8; 5 * slot];
            let arrays: [Vec<i32>; 5] = [qsl.clone(), seq.iter().map(|&x| x as i32).collect(), gather.iter().map(|&x| x as i32).collect(), cbase.to_vec(), sbase.to_vec()];
            for (i, arr) in arrays.iter().enumerate() { v[i * slot..i * slot + 4 * arr.len()].copy_from_slice(&le_i32(arr)); }
            v
        };
        let mut m = ChunkMeta { gather: gather.iter().sum(), nreqs: nr, meta_slot: slot, ..Default::default() };
        m.swa_out = put(&le_i32(&segs.iter().flat_map(|g| (g.start..g.start + g.len).map(|p| self.cache.swa_slot(&g.req, p))).collect::<Vec<_>>()));
        m.swa_ids = put(&le_i32(&segs.iter().zip(&seq).zip(&gather).flat_map(|((g, &sl), &gl)| (sl - gl..sl).map(|p| self.cache.swa_slot(&g.req, p))).collect::<Vec<_>>()));
        m.c0_combine = put(&combine(&vec![0; nr], &swa_off.iter().map(|&o| o as i32).collect::<Vec<_>>()));
        m.ke = put(&le_i32(&segs.iter().flat_map(|g| (g.start..g.start + g.len).map(|p| ((p + 1) / 4) as i32)).collect::<Vec<_>>()));
        let max_seq = seq.iter().copied().max().unwrap_or(0);
        for ratio in [4usize, 128] {
            // Compress / write plans per request in order, ragged ids counting the batch's tokens
            // (plan_compress_prefill's CPU path); a plan without buffered reads keeps its batch
            // index in read_page_1 (plan_compress_prefill_kernel_1 leaves it).
            let (pc, pw, out_loc) = self.ratio_plans(segs, ratio);
            // Workspace layout (SparsePrefillChunkCache): C4, each request's own compressed tokens
            // in turn; C128, a region of max(longest / 128, 1) rows per request (missing rows read
            // slot 0, masked). The SWA gathers follow.
            let (slots, cbase, total, per_req): (Vec<i32>, Vec<i32>, usize, usize) = if ratio == 4 {
                let lens: Vec<usize> = seq.iter().map(|&sl| sl / 4).collect();
                let cbase = lens.iter().scan(0, |acc, &l| { let o = *acc; *acc += l; Some(o as i32) }).collect();
                (segs.iter().zip(&lens).flat_map(|(g, &l)| (0..l).map(|e| g.req.c4_slot(e))).collect(), cbase, lens.iter().sum(), 0)
            } else {
                let cmax = (max_seq / 128).max(1);
                (segs.iter().zip(&seq).flat_map(|(g, &sl)| (0..cmax).map(move |e| if e < sl / 128 { g.req.c128_slot(e) } else { 0 })).collect(),
                 (0..nr).map(|r| (r * cmax) as i32).collect(), nr * cmax, cmax)
            };
            let sbase: Vec<i32> = swa_off.iter().map(|&o| (total + o) as i32).collect();
            let r = RatioMeta {
                plan_c: put(&compress::plan_c_bytes(&pc)), num_c: pc.len(), plan_w: put(&compress::plan_w_bytes(&pw)), num_w: pw.len(),
                out_loc: put(&i64s(&out_loc)), slots: put(&le_i32(&slots)), n: total, combine: put(&combine(&cbase, &sbase)), per_req,
            };
            if ratio == 4 { m.c4 = r } else { m.c128 = r }
        }
        m.ident = put(&le_i32(&(0..INDEX_TOPK.max(m.c128.per_req) as i32).collect::<Vec<_>>()));
        m.zeros = put(&vec![0u8; t * 4]);
        if nr > 1 {
            let width = seq.iter().map(|&sl| KvCache::page_width(sl)).max().unwrap_or(1);
            m.table_width = width;
            m.table = put(&le_i32(&segs.iter().flat_map(|g| std::iter::repeat(g.req.page0).take(g.len)).flat_map(|p0| (0..width).map(move |p| (p0 + p) as i32)).collect::<Vec<_>>()));
        }
        let base = a.alloc(buf.len())?;
        crate::cuda::upload_async(base, &buf, self.s())?;
        for p in [&mut m.swa_out, &mut m.swa_ids, &mut m.c0_combine, &mut m.ke, &mut m.ident, &mut m.zeros] { *p += base; }
        if nr > 1 { m.table += base; }
        for r in [&mut m.c4, &mut m.c128] { for p in [&mut r.plan_c, &mut r.plan_w, &mut r.out_loc, &mut r.slots, &mut r.combine] { *p += base; } }
        Ok(m)
    }

    /// The C4 sparse indexer of a prefill chunk (SGLang C4IndexerBackendMixin.forward_c4_indexer):
    /// weights_proj, wq_b on the FP8 q_lora, RoPE + Hadamard + FP8 of the query, the indexer
    /// compressor (its keys into the index-K cache), then per query the MQA logits over every
    /// compressed key it may see and their top-512 (flashinfer, deterministic). Returns the top-k raw
    /// indices [t, 512] (request-local compressed positions, sorted, -1 padded), or None while the
    /// request holds at most 512 compressed tokens: every query then takes all of its own, in order,
    /// whatever the scores (the top-k of a row no longer than k is the row).
    #[allow(clippy::too_many_arguments)]
    fn c4_indexer(&mut self, layer: usize, x: u64, (lq, ls): (u64, u64), b: &Batch, (plan_c, num_c): (u64, usize), (plan_w, num_w): (u64, usize),
                  out_loc: u64, a: &mut Arena) -> Result<Option<u64>> {
        let (t, s, k, sms) = (b.t, self.s(), self.k, self.sms());
        let l = format!("model.layers.{layer}.self_attn.indexer");
        let (heads, dim) = (INDEX_HEADS, INDEX_DIM);
        // Indexer compressor (head dim 128) into the index-K cache.
        let kv_score = a.alloc(t * 4 * dim * 4)?;
        self.lt.linear(s, t, 4 * dim, HIDDEN, x, self.w.get(&format!("{l}.compressor.wkv_gate.weight"))?, kv_score, true)?;
        let comp = a.alloc(num_c.max(1) * dim * 4)?;
        compress::c4_prefill(k, s, dim, self.cache.index_state_layer(layer), kv_score, comp, self.w.get(&format!("{l}.compressor.ape"))?,
                             plan_c, plan_w, num_c, num_w)?;
        compress::norm_rope_store_indexer(k, s, comp, plan_c, self.w.get(&format!("{l}.compressor.norm.weight"))?, self.tables.freqs(layer),
                                          out_loc, self.cache.index_k_layer(layer), 4, num_c, 6, false)?;
        // While every request holds at most 512 compressed tokens, each query takes all of its own.
        let seq_len = b.segs.iter().map(|g| g.start + g.len).max().unwrap_or(b.start + t);
        let n_c = seq_len / 4;
        if n_c <= INDEX_TOPK { return Ok(None); }
        // Query and per-head weights.
        let w = a.alloc(t * heads * 2)?;
        self.lt.linear(s, t, heads, HIDDEN, x, self.w.get(&format!("{l}.weights_proj.weight"))?, w, false)?;
        let q = a.alloc(t * heads * dim * 2)?;
        deepgemm::fp8_gemm_nt(k, s, lq, ls, self.w.get(&format!("{l}.wq_b.weight"))?, self.w.get(&format!("{l}.wq_b.weight_scale_inv"))?,
                              q, t, heads * dim, Q_LORA, sms)?;
        let q_fp8 = a.alloc(t * heads * dim)?;
        let weights = a.alloc(t * heads * 4)?;
        // weight_scale = head_dim^-0.5 * n_heads^-0.5 (Python float, cast to float).
        sgl::fused_q_indexer_rope_hadamard_quant(k, s, q, q_fp8, w, weights, f32::from_bits(0x3c3504f3), self.tables.freqs(layer), b.pos32,
                                                 heads, t, heads)?;
        // Every query scores [0, its compressed length) of the request's keys.
        let ke = self.chunk.ke;
        // One request: its page-table row for every query (stride 0). Several (a packed batch):
        // each query's own row.
        let (width, page_row, table_stride) = if b.segs.len() > 1 { (self.chunk.table_width, self.chunk.table, self.chunk.table_width) }
                                              else { (KvCache::page_width(seq_len), self.cache.req_page_row(&b.req), 0) };
        // SGLang's non-paged indexer serves a batch of one request only.
        let (logits, max_len, stride) = if t >= NONPAGED_MIN_QUERIES && b.segs.len() == 1 {
            // Non-paged: the keys gathered once from the index-K cache, contiguous MQA logits.
            let ks = a.alloc(t * 4)?;
            crate::cuda::memset32_async(ks, 0, t, s)?;
            let kv = a.alloc(n_c * dim)?;
            let kv_scales = a.alloc(n_c * 4)?;
            sgl::get_k_and_s(k, s, self.cache.index_k_layer(layer), page_row, kv, kv_scales, ke + 4 * (t as u64 - 1), n_c, width)?;
            let max_seqlen_k = n_c.div_ceil(C4_PAGE_TOKENS) * C4_PAGE_TOKENS;
            let stride = deepgemm::mqa_logits_stride(max_seqlen_k);
            let logits = a.alloc(t.div_ceil(2) * 2 * stride * 4)?;
            deepgemm::mqa_logits(k, s, q_fp8, kv, kv_scales, weights, ks, ke, logits, t, n_c, max_seqlen_k, heads, sms)?;
            (logits, max_seqlen_k, stride)
        } else {
            // Paged: MQA logits straight from the index-K cache pages (max context = the page
            // table's width in compressed entries).
            let max_context = width * C4_PAGE_TOKENS;
            let stride = deepgemm::mqa_logits_stride(max_context);
            let logits = a.alloc(t * stride * 4)?;
            let ke_host: Vec<i32> = b.segs.iter().flat_map(|g| (g.start..g.start + g.len).map(|p| ((p + 1) / 4) as i32)).collect();
            let sched = deepgemm::paged_mqa_schedule(&ke_host, sms);
            let sched_dev = a.alloc(sched.len() * 4)?;
            crate::cuda::upload_async(sched_dev, &le_i32(&sched), s)?;
            deepgemm::paged_mqa_logits(k, s, q_fp8, self.cache.index_k_layer(layer), self.cache.c4_pages, weights, ke, page_row, table_stride, sched_dev,
                                       logits, t, max_context, heads, sms)?;
            (logits, max_context, stride)
        };
        let pages = a.alloc(t * INDEX_TOPK * 4)?;
        let raw = a.alloc(t * INDEX_TOPK * 4)?;
        crate::topk::page_table_transform(k, s, logits, pages, raw, page_row, table_stride, ke, t, INDEX_TOPK, max_len, stride, 6)?;
        Ok(Some(raw))
    }

    /// C128 attention (odd layers from 3) for a prefill chunk: the C128 compressor (wkv_gate, state
    /// ring writes, compression at every 128th position) and sparse prefill over [every compressed
    /// token so far | the window] (ensure_c128: the compressed region has max(seq_len / 128, 1) rows).
    pub fn attention_c128(&mut self, layer: usize, x: u64, b: &Batch, a: &mut Arena) -> Result<u64> {
        let (t, s, k) = (b.t, self.s(), self.k);
        let l = format!("model.layers.{layer}.self_attn");
        let seq_len = b.start + t;
        let (q_rot, qkv_a, _) = self.attn_qkv(layer, x, b, a)?;
        self.store_swa(layer, qkv_a, b, a)?;
        let kv_score = a.alloc(t * 2 * HEAD_DIM * 4)?;
        self.lt.linear(s, t, 2 * HEAD_DIM, HIDDEN, x, self.w.get(&format!("{l}.compressor.wkv_gate.weight"))?, kv_score, true)?;
        let m = self.chunk.c128;
        let comp = if m.num_c == 0 { 0 } else { a.alloc(m.num_c * HEAD_DIM * 4)? };
        compress::c128_prefill(k, s, self.cache.c128_state_layer(layer), kv_score, comp, self.w.get(&format!("{l}.compressor.ape"))?,
                               m.plan_c, m.plan_w, m.num_c, m.num_w)?;
        if m.num_c > 0 {
            // Each compressed row (one per 128th position) into the C128 cache (pages of 2).
            compress::norm_rope_store_flashmla(k, s, comp, m.plan_c, self.w.get(&format!("{l}.compressor.norm.weight"))?, self.tables.freqs(layer),
                                               m.out_loc, self.cache.c128_layer(layer), 128, m.num_c, 1, false)?;
        }
        let _ = seq_len;
        let o = self.sparse_prefill(layer, q_rot, b, a, Some(128), None)?;
        self.attn_out(layer, o, b, a)
    }

    /// FlashMLA sparse prefill of a chunk over the workspace [compressed tokens | SWA window union]
    /// (SparsePrefillChunkCache): per query, its compressed tokens (`ratio`) then its window.
    fn sparse_prefill(&mut self, layer: usize, q_rot: u64, b: &Batch, a: &mut Arena, ratio: Option<usize>, topk_raw: Option<u64>) -> Result<u64> {
        let (t, s, k) = (b.t, self.s(), self.k);
        let cm = self.chunk;
        // The window union of the chunk's queries (build_swa_token_ids).
        let gather = cm.gather;
        // Compressed region: C4 gathers the request's compressed tokens and combines its indexer
        // top-k (width 512; while trivial, every token in order); C128 gathers max(seq_len / 128, 1)
        // rows (missing ones read slot 0 and stay masked) and every query takes all it may see.
        let (n_c, topk, slots, cache, kind, combine): (usize, usize, u64, u64, &str, u64) = match ratio {
            None => (0, 0, 0, 0, "meta", cm.c0_combine),
            Some(4) => (cm.c4.n, INDEX_TOPK, cm.c4.slots, self.cache.c4_layer(layer), "c4", cm.c4.combine),
            Some(_) => (cm.c128.n, cm.c128.per_req, cm.c128.slots, self.cache.c128_layer(layer), "c128", cm.c128.combine),
        };
        let workspace = a.alloc((n_c + gather) * HEAD_DIM * 2)?;
        if n_c > 0 { sgl::dequant_k_cache(k, s, workspace, cache, slots, HEAD_DIM, n_c, kind)?; }
        sgl::dequant_k_cache(k, s, workspace + (n_c * HEAD_DIM * 2) as u64, self.cache.swa_layer(layer), cm.swa_ids, HEAD_DIM, gather, "swa")?;
        // _combine_topk_swa_indices_kernel: [top-k (compressed_base 0) | window (swa_base n_c) | -1];
        // without an indexer top-k, the identity row read with stride 0 (every token in order).
        let width = (topk + WINDOW).div_ceil(128) * 128;
        let indices = a.alloc(t * width * 4)?;
        let topk_length = a.alloc(t * 4)?;
        let (rows, row_stride) = match (topk_raw, ratio) {
            (Some(raw), _) => (raw, INDEX_TOPK),
            (None, None) => (cm.ident, 1),                  // window-only: top_k 0, rows never read
            (None, Some(128)) if topk == 1 => (cm.zeros, 1),
            (None, Some(_)) => (cm.ident, 0),
        };
        sgl::combine_topk_swa(k, s, indices, width, topk_length, rows, row_stride, combine, cm.meta_slot, cm.nreqs, b.pos32, t, topk, kind)?;
        let qn = HEAD_SLOTS * HEAD_DIM;
        let o = a.alloc(t * qn * 2)?;
        // The sinks of this rank's heads, in the kernel's HEAD_SLOTS slots (the rest zero).
        let sink_param = self.w.get(&format!("model.layers.{layer}.self_attn.attn_sink"))? + (self.rank * HEADS * 4) as u64;
        let sink = if HEAD_SLOTS == HEADS { sink_param } else {
            let p = a.alloc(HEAD_SLOTS * 4)?;
            crate::cuda::memset_async(p, 0, HEAD_SLOTS * 4, s)?;
            crate::cuda::copy_async(p, sink_param, HEADS * 4, s)?;
            p
        };
        let args = flashmla::PrefillArgs {
            s_q: t as i32, s_kv: (n_c + gather) as i32, h_q: HEAD_SLOTS as i32, h_kv: 1, d_qk: HEAD_DIM as i32, d_v: HEAD_DIM as i32, topk: width as i32,
            sm_scale: f32::from_bits(0x3d3504f3), // head_dim ** -0.5
            q: q_rot, kv: workspace, indices, attn_sink: sink, topk_length,
            stride_q_s_q: qn as i32, stride_q_h_q: HEAD_DIM as i32, stride_kv_s_kv: HEAD_DIM as i32, stride_kv_h_kv: HEAD_DIM as i32,
            stride_indices_s_q: width as i32, stride_indices_h_kv: width as i32,
            out: o, max_logits: a.alloc(t * HEAD_SLOTS * 4)?, lse: a.alloc(t * HEAD_SLOTS * 4)?, num_sm: self.sms() as i32, stats_in_log2: 1,
        };
        flashmla::sparse_prefill(k, s, &args)?;
        Ok(o)
    }

    /// The shared expert: FP8 gate_up, SiLU-mul with clamp, FP8 down (DeepGEMM).
    fn shared_expert(&mut self, layer: usize, x: u64, t: usize, a: &mut Arena) -> Result<u64> {
        let (s, k, sms) = (self.s(), self.k, self.sms());
        let sh = format!("model.layers.{layer}.mlp.shared_experts");
        let q1 = a.alloc(t * HIDDEN)?;
        let s1 = a.alloc(sgl::act_scale_bytes(t, HIDDEN))?;
        sgl::quant_fp8(k, s, x, HIDDEN, q1, s1, t, HIDDEN)?;
        let gate_up = a.alloc(t * 2 * INTER * 2)?;
        deepgemm::fp8_gemm_nt(k, s, q1, s1, self.w.get(&format!("{sh}.gate_up_proj.weight"))?, self.w.get(&format!("{sh}.gate_up_proj.weight_scale_inv"))?,
                              gate_up, t, 2 * INTER, HIDDEN, sms)?;
        let act = a.alloc(t * INTER * 2)?;
        sgl::silu_mul_clamp(k, s, gate_up, act, t, INTER, 10.0)?;
        let q2 = a.alloc(t * INTER)?;
        let s2 = a.alloc(sgl::act_scale_bytes(t, INTER))?;
        sgl::quant_fp8(k, s, act, INTER, q2, s2, t, INTER)?;
        let shared = a.alloc(t * HIDDEN * 2)?;
        deepgemm::fp8_gemm_nt(k, s, q2, s2, self.w.get(&format!("{sh}.down_proj.weight"))?, self.w.get(&format!("{sh}.down_proj.weight_scale_inv"))?,
                              shared, t, HIDDEN, INTER, sms)?;
        Ok(shared)
    }

    /// The FFN half: routed experts (hash routing on the first layers) + the shared expert; returns
    /// routed + shared (the layer output; its mHC post runs in the next layer's first kernel).
    pub fn ffn(&mut self, layer: usize, x: u64, b: &Batch, a: &mut Arena) -> Result<u64> {
        let out = a.alloc(b.t * HIDDEN * 2)?;
        self.ffn_into(layer, x, b, a, out)
    }

    /// The FFN half writing routed + shared into `out` (the layer output buffer, so no copy follows).
    pub fn ffn_into(&mut self, layer: usize, x: u64, b: &Batch, a: &mut Arena, out: u64) -> Result<u64> {
        let (t, s, k, sms) = (b.t, self.s(), self.k, self.sms());
        let m = format!("model.layers.{layer}.mlp");
        // The shared expert runs beside the routed experts (side 0) in decode, as SGLang overlaps it
        // under graph capture only; the sum waits for both.
        let s_shared = if b.decode { self.side(0) } else { None };
        self.fork(s_shared)?;
        let shared = self.on(s_shared, |me| me.shared_expert(layer, x, t, a))?;
        let topk_ids = a.alloc(t * 6 * 4)?;
        let topk_ws = a.alloc(t * 6 * 4)?;
        // Router logits (MoEGate): the tiny GEMM up to 16 tokens, else cuBLAS bf16 x bf16 -> fp32.
        let gate = self.w.get(&format!("{m}.gate.weight"))?;
        let router = |me: &Self, logits: u64| -> Result<()> {
            if t <= 16 { sgl::router_tiny_gemm(k, s, logits, x, gate, t, HIDDEN) } else { me.lt.linear(s, t, EXPERTS, HIDDEN, x, gate, logits, true).map(|_| ()) }
        };
        if layer < 3 {
            let logits = a.alloc(t * EXPERTS * 4)?;
            router(self, logits)?;
            let topk_w = a.alloc(t * 6 * 4)?;
            sgl::hash_topk(k, s, logits, b.ids, self.w.get(&format!("{m}.topk.tid2eid"))?, topk_ids, topk_w, t, 6, EXPERTS, ROUTED_SCALE)?;
            torch_ops::mul_scalar_f32(k, s, topk_ws, topk_w, t * 6, ROUTED_SCALE)?;
        } else {
            let logits = a.alloc(t * EXPERTS * 4)?;
            router(self, logits)?;
            let packed = a.alloc(4)?;
            sgl::router_triton(k, s, logits, self.w.get(&format!("{m}.gate.e_score_correction_bias"))?, topk_ws, topk_ids, packed, t, ROUTED_SCALE)?;
        }
        if layer == DRAFT_LAYER {
            // The MTP draft's MXFP4 experts (Mxfp4FlashinferTrtllmMoEMethod): MXFP8 input, unit
            // output scales, the swiglu limit as GEMM1's clamp.
            let xq = a.alloc(t * HIDDEN)?;
            let xs = a.alloc(t * HIDDEN / 32)?;
            sgl::mxfp8_quantize_4096(k, s, x, xq, xs, t, sms)?;
            let experts = moe::Experts {
                w13: self.w.get(&format!("{m}.experts.w13_weight"))?, w13_scale: self.w.get(&format!("{m}.experts.w13_weight_scale_inv"))?,
                w2: self.w.get(&format!("{m}.experts.w2_weight"))?, w2_scale: self.w.get(&format!("{m}.experts.w2_weight_scale_inv"))?,
                clamp_limit: self.w.get(&format!("{m}.experts.gemm1_clamp_limit"))?, g1_scale_c: self.w.get(&format!("{m}.experts.output1_scale_scalar"))?,
                g1_alphas: self.w.get(&format!("{m}.experts.output1_scale_gate_scalar"))?, g2_alphas: self.w.get(&format!("{m}.experts.output2_scale_scalar"))?,
            };
            let routed = a.alloc(t * HIDDEN * 2)?;
            let scratch_bytes = moe_scratch_bytes(t, a.remaining());
            let scratch = a.alloc(scratch_bytes)?;
            moe::mxfp4_routed(k, s, &self.trtllm_cubins, &experts, topk_ids, topk_ws, xq, xs, routed, t, scratch, scratch_bytes)?;
            self.join(s_shared)?;
            torch_ops::add_bf16(k, s, out, routed, shared, t * HIDDEN)?;
            return Ok(out);
        }
        let xq = a.alloc(t * HIDDEN / 2)?;
        let xs = a.alloc(t * HIDDEN / 16)?;
        sgl::nvfp4_quantize_4096(k, s, x, xq, xs, t, self.w.get(&format!("{m}.experts.w13_input_scale_quant"))?, sms)?;
        let experts = moe::Experts {
            w13: self.w.get(&format!("{m}.experts.w13_weight"))?, w13_scale: self.w.get(&format!("{m}.experts.w13_weight_scale"))?,
            w2: self.w.get(&format!("{m}.experts.w2_weight"))?, w2_scale: self.w.get(&format!("{m}.experts.w2_weight_scale"))?,
            clamp_limit: self.w.get(&format!("{m}.experts.gemm1_clamp_limit"))?, g1_scale_c: self.w.get(&format!("{m}.experts.g1_scale_c"))?,
            g1_alphas: self.w.get(&format!("{m}.experts.g1_alphas"))?, g2_alphas: self.w.get(&format!("{m}.experts.g2_alphas"))?,
        };
        let routed = a.alloc(t * HIDDEN * 2)?;
        // Expert tiles padded per touched expert (up to all 256) dominate at small token counts.
        let scratch_bytes = moe_scratch_bytes(t, a.remaining());
        let scratch = a.alloc(scratch_bytes)?;
        moe::fp4_routed(k, s, &self.trtllm_cubins, &experts, topk_ids, topk_ws, xq, xs, routed, t, scratch, scratch_bytes)?;
        self.join(s_shared)?;
        if self.probe == Some(layer) {
            let r = a.alloc(t * HIDDEN * 2)?;
            crate::cuda::copy_async(r, routed, t * HIDDEN * 2, s)?;
            self.probes.push((format!("model.layers.{layer}.mlp.out.0"), r, t * HIDDEN * 2));
            self.probes.push((format!("model.layers.{layer}.mlp.out.1"), shared, t * HIDDEN * 2));
        }
        torch_ops::add_bf16(k, s, out, routed, shared, t * HIDDEN)?;
        Ok(out)
    }

    /// Decode attention (one token at position b.start): q/kv projections and SWA store as in prefill,
    /// the layer's compressor decode step, then FlashMLA sparse decode over the FP8 caches: the
    /// window (newest first) plus the compressed tokens (C4: every one in order, its top-k while at
    /// most 512; C128: SGLang's 8256-wide page-index row with the length clamped to 1).
    /// Fill the decode inputs for the token `id` at position `pos` (one upload on the stream) and
    /// return the decode batch that reads them.
    pub fn decode_batch(&mut self, id: i64, pos: usize) -> Result<Batch> {
        self.decode_batch_rows(&[DecodeRow { req: Req::SINGLE, pos, id }])
    }

    /// Fill the decode inputs for `rows` (one upload on the stream) and return the batch reading them.
    pub fn decode_batch_rows(&mut self, rows: &[DecodeRow]) -> Result<Batch> {
        let place: Vec<(Req, usize)> = rows.iter().map(|r| (r.req, r.pos)).collect();
        let mut h = self.decode_host_inputs(&place)?;
        let dl = self.dl;
        for (i, r) in rows.iter().enumerate() { h[dl.ids + 8 * i..dl.ids + 8 * i + 8].copy_from_slice(&r.id.to_le_bytes()); }
        crate::cuda::upload_async(self.decode_in.as_ref().unwrap().ptr, &h, self.s())?;
        Ok(self.decode_view(rows[0].req, rows[0].pos))
    }

    /// One speculative verify forward (SGLang's TARGET_VERIFY): decode rows `rows` (each request's
    /// draft tokens at their positions, request by request), the compressors run as an extend of
    /// each request's segment in `segs`. Token ids from the host (`ids`) or, when `ids_dev` is
    /// given, copied from the device. Returns the bf16 logits [rows, vocab].
    pub fn verify_step(&mut self, rows: &[(Req, usize)], segs: &[Seg], ids: &[i64], ids_dev: Option<u64>, persist: &mut Arena, scratch: &mut Arena) -> Result<u64> {
        let mut h = self.decode_host_inputs(rows)?;
        let dl = self.dl;
        for (i, id) in ids.iter().enumerate() { h[dl.ids + 8 * i..dl.ids + 8 * i + 8].copy_from_slice(&id.to_le_bytes()); }
        let din = self.decode_in.as_ref().unwrap().ptr;
        crate::cuda::upload_async(din, &h, self.s())?;
        if let Some(src) = ids_dev { crate::cuda::copy_async(din + dl.ids as u64, src, 8 * rows.len(), self.s())?; }
        let mut b = self.decode_view(rows[0].0, rows[0].1);
        b.verify = true;
        b.segs = segs.to_vec();
        self.verify_meta(segs)?;
        persist.reset();
        self.forward(&b, persist, scratch, None)
    }

    /// Load the MTP draft's parameters, rebuilt from the checkpoint at `ckpt` (Source::prepare_draft,
    /// checked against SGLang's loaded draft): the decoder's as `model.layers.{DRAFT_LAYER}.*`, the
    /// rest as `draft.*`; plus the MXFP4 experts' per-expert swiglu limit (10.0).
    pub fn load_draft_weights(&mut self, ckpt: &std::path::Path) -> Result<()> {
        let ck = crate::safetensors::Checkpoint::open(ckpt)?;
        let src = crate::weights::Source { ckpt: &ck };
        for &name in DRAFT_PARAMS {
            let logical = src.prepare_draft(name)?.ok_or_else(|| format!("draft parameter {name} is not ported"))?;
            let bytes = crate::weights::memory_layout(name, logical);
            let mapped = match name.strip_prefix("model.decoder.") {
                Some(rest) => format!("model.layers.{DRAFT_LAYER}.{rest}"),
                None => format!("draft.{}", &name["model.".len()..]),
            };
            self.w.insert(&mapped, Buffer::from_host(&bytes)?);
        }
        let limit: Vec<u8> = (0..256).flat_map(|_| 10.0f32.to_le_bytes()).collect();
        self.w.insert(&format!("model.layers.{DRAFT_LAYER}.mlp.experts.gemm1_clamp_limit"), Buffer::from_host(&limit)?);
        Ok(())
    }

    /// The MTP draft's hc_head (DeepseekV4ModelNextN.hc_head, torch ops on the pre-hc_head state
    /// `hc` [t, 4, hidden] bf16): x = hc.float(); rsqrt(mean(x^2) + eps); F.linear(x, hc_fn) * rsqrt;
    /// sigmoid(. * hc_scale + hc_base) + hc_eps; sum over the 4 copies of pre * x; to bf16. The
    /// reductions, rsqrt and sigmoid replay ATen's kernels, the fp32 GEMM runs torch's cuBLAS call,
    /// the remaining single IEEE operations run crate::exact. Returns bf16 [t, hidden].
    fn hc_head_draft(&mut self, hc: u64, t: usize, a: &mut Arena) -> Result<u64> {
        let (s, k) = (self.s(), self.k);
        let n = HC * HIDDEN;
        let x = a.alloc(t * n * 4)?;
        torch_ops::copy_bf16_to_f32(k, s, x, hc, t * n)?;
        let sq = a.alloc(t * n * 4)?;
        crate::exact::square_f32(s, x, sq, t * n)?;
        let mean = a.alloc(t * 4)?;
        torch_ops::mean_rows_f32(k, s, sq, t, n, mean)?;
        let me = a.alloc(t * 4)?;
        crate::exact::add_scalar_f32(s, mean, EPS, me, t)?;
        let rs = a.alloc(t * 4)?;
        torch_ops::rsqrt_f32(k, s, me, rs, t)?;
        let mixes = a.alloc(t * HC * 4)?;
        let ws = self.cublas_workspace()?;
        crate::exact::linear_f32(s, x, self.w.get("draft.hc_head_fn")?, mixes, t, HC, n, ws)?;
        let m1 = a.alloc(t * HC * 4)?;
        crate::exact::mul_rows_f32(s, mixes, rs, m1, t, HC)?;
        let m2 = a.alloc(t * HC * 4)?;
        crate::exact::mul_dev_scalar_f32(s, m1, self.w.get("draft.hc_head_scale")?, m2, t * HC)?;
        let m3 = a.alloc(t * HC * 4)?;
        crate::exact::add_cols_f32(s, m2, self.w.get("draft.hc_head_base")?, m3, t, HC)?;
        let sg = a.alloc(t * HC * 4)?;
        torch_ops::sigmoid_f32(k, s, m3, sg, t * HC)?;
        let pre = a.alloc(t * HC * 4)?;
        crate::exact::add_scalar_f32(s, sg, EPS, pre, t * HC)?;  // hc_eps (1e-6, as rms_norm_eps)
        let prod = a.alloc(t * n * 4)?;
        crate::exact::mul_bcast_last_f32(s, pre, x, prod, t * HC, HIDDEN)?;
        let y32 = a.alloc(t * HIDDEN * 4)?;
        torch_ops::sum_dim1_f32(k, s, prod, t, HC, HIDDEN, y32)?;
        let y = a.alloc(t * HIDDEN * 2)?;
        crate::exact::f32_to_bf16(s, y32, y, t * HIDDEN)?;
        Ok(y)
    }

    /// torch's cuBLAS workspace (allocated once; cuBLAS opened on first use).
    fn cublas_workspace(&mut self) -> Result<u64> {
        if self.blas_ws.is_none() {
            let lt_path = std::env::var("DSV4_CUBLASLT").unwrap_or_else(|_| "/opt/deps/cublas/nvidia/cu13/lib/libcublasLt.so.13".into());
            crate::exact::cublas_open(&lt_path)?;
            self.blas_ws = Some(Buffer::alloc(crate::exact::CUBLAS_WORKSPACE_BYTES)?);
        }
        Ok(self.blas_ws.as_ref().unwrap().ptr)
    }

    /// An FP8 block-quantized linear (per-token-group FP8 activations, DeepGEMM): x bf16 [rows, k]
    /// times `{name}.weight` [n, k] -> bf16 [rows, n].
    fn fp8_linear(&mut self, x: u64, name: &str, rows: usize, n: usize, kdim: usize, a: &mut Arena) -> Result<u64> {
        let (s, k, sms) = (self.s(), self.k, self.sms());
        let q = a.alloc(rows * kdim)?;
        let sc = a.alloc(sgl::act_scale_bytes(rows, kdim))?;
        sgl::quant_fp8(k, s, x, kdim, q, sc, rows, kdim)?;
        let y = a.alloc(rows * n * 2)?;
        deepgemm::fp8_gemm_nt(k, s, q, sc, self.w.get(&format!("{name}.weight"))?, self.w.get(&format!("{name}.weight_scale_inv"))?, y, rows, n, kdim, sms)?;
        Ok(y)
    }

    /// One forward of the MTP draft (SGLang's DeepseekV4ModelNextN): `b`'s token embeddings (enorm,
    /// e_proj) plus each row's hyper-connection state `hc_in` [t, 4, hidden] (hnorm, h_proj: the
    /// target's pre-hc_head state, or the draft's own from its previous step) through the draft's
    /// layer (DRAFT_LAYER, SWA only: a prefill chunk when `b` is one, else decode rows), its
    /// pre-hc_head state into `hc_out` [t, 4, hidden], hc_head, the shared head norm and the LM
    /// head over `rows`. Returns the bf16 logits [rows.len(), vocab].
    pub fn draft_forward(&mut self, b: &Batch, hc_in: u64, rows: &[usize], hc_out: u64, persist: &mut Arena, scratch: &mut Arena) -> Result<u64> {
        let (t, s, k, sms) = (b.t, self.s(), self.k, self.sms());
        if !b.decode { self.chunk = self.chunk_meta(b, persist)?; }
        self.c128_sched = Some((persist.alloc(148 * 32)?, persist.alloc(4 * (t + 1))?, false));
        scratch.reset();
        let emb = scratch.alloc(t * HIDDEN * 2)?;
        torch_ops::embedding_bf16(k, s, emb, self.w.get("model.embed_tokens.weight")?, b.ids, t, VOCAB, HIDDEN)?;
        let en = scratch.alloc(t * HIDDEN * 2)?;
        sgl::rmsnorm_4096(k, s, emb, self.w.get("draft.enorm.weight")?, en, t, EPS)?;
        let hn = scratch.alloc(HC * t * HIDDEN * 2)?;
        sgl::rmsnorm_4096(k, s, hc_in, self.w.get("draft.hnorm.weight")?, hn, HC * t, EPS)?;
        let e = self.fp8_linear(en, "draft.e_proj", t, HIDDEN, HIDDEN, scratch)?;
        let h = self.fp8_linear(hn, "draft.h_proj", HC * t, HIDDEN, HIDDEN, scratch)?;
        // The mixed state [t, 4, hidden] is the layer's residual; its attention input the mHC pre.
        let mut hc = Hc::new(t, persist)?;
        let set = hc.sets[0];
        crate::exact::add_bcast_hc_bf16(s, e, h, set.residual, t, HC, HIDDEN)?;
        let l = format!("model.layers.{DRAFT_LAYER}");
        let splits = deepgemm::mhc_pre_splits(t, HC * HIDDEN, sms);
        let d = scratch.alloc(splits * t * MIX * 4)?;
        let sq = scratch.alloc(splits * t * 4)?;
        deepgemm::tf32_hc_prenorm_gemm(k, s, set.residual, self.w.get(&format!("{l}.hc_attn_fn"))?, d, sq, t, MIX, HC * HIDDEN, splits)?;
        let pre = mhc::MhcPreOut { post_mix: set.post_mix, comb_mix: set.comb_mix, layer_input: set.layer_input };
        mhc::pre_big_fuse_with_norm(k, s, d, sq, self.w.get(&format!("{l}.hc_attn_scale"))?, self.w.get(&format!("{l}.hc_attn_base"))?,
                                    set.residual, &pre, self.w.get(&format!("{l}.input_layernorm.weight"))?, t, splits)?;
        hc.residual = set.residual;
        scratch.reset();
        let y = if b.decode { self.attention_decode(DRAFT_LAYER, pre.layer_input, b, scratch)? } else { self.attention_swa(DRAFT_LAYER, pre.layer_input, b, scratch)? };
        let xf = self.post_pre(DRAFT_LAYER, "ffn", &mut hc, y, t, persist, scratch)?;
        let layer_out = hc.layer_out();
        self.ffn_into(DRAFT_LAYER, xf, b, scratch, layer_out)?;
        scratch.reset();
        mhc::post(k, s, hc.comb_mix, hc.residual, hc.post_mix, layer_out, hc_out, t)?;
        let yh = self.hc_head_draft(hc_out, t, scratch)?;
        let normed = scratch.alloc(t * HIDDEN * 2)?;
        sgl::rmsnorm_4096(k, s, yh, self.w.get("draft.shared_head.norm.weight")?, normed, t, EPS)?;
        // The LM head over the requested rows (gathered when they are not all rows in order).
        let r = rows.len();
        let input = if rows.iter().enumerate().all(|(i, &x)| x == i) && r == t { normed } else {
            let g = scratch.alloc(r * HIDDEN * 2)?;
            for (i, &row) in rows.iter().enumerate() { crate::cuda::copy_async(g + (i * HIDDEN * 2) as u64, normed + (row * HIDDEN * 2) as u64, HIDDEN * 2, s)?; }
            g
        };
        let logits = persist.alloc(r * VOCAB * 2)?;
        self.lt.linear(s, r, VOCAB, HIDDEN, input, self.w.get("lm_head.weight")?, logits, false)?;
        Ok(logits)
    }

    /// Each row's greedy token of bf16 logits [rows, vocab] into `out` (int64 [rows]), as SGLang
    /// samples: the fp32 logits' first maximum (torch.argmax; its draft top-1 kernel picks the same
    /// index). With `softmax`, the maximum of the cake softmax probabilities instead (the draft's
    /// prefill extend). Device only (graph-capturable).
    fn greedy_dev(&self, logits: u64, rows: usize, wide: u64, probs: u64, out: u64, softmax: bool) -> Result<()> {
        let s = self.s();
        torch_ops::copy_bf16_to_f32(self.k, s, wide, logits, rows * VOCAB)?;
        let src = if softmax { crate::topk::cake_softmax(self.k, s, wide, probs, rows, VOCAB)?; probs } else { wide };
        torch_ops::argmax_f32_rows(self.k, s, src, rows, VOCAB, out)
    }

    /// Greedy generation of `max_new` tokens for one prompt with SGLang's MTP speculative decoding
    /// (EAGLE worker v2, 3 steps, top-1, 4 draft tokens), forward for forward as SGLang runs it.
    /// L is the committed length (the target's KV holds positions 0..L), x the pending token at L:
    /// - prefill: the target over the prompt (x = its last row's argmax), then the draft's extend
    ///   over the prompt shifted by one plus x with the target's pre-hc_head states (d1 = top-1 of
    ///   the cake softmax of its last row);
    /// - per round: two draft decodes (d1 at L, d2 at L + 1, each attending to the committed window
    ///   only, the hidden state the draft's own previous output) giving d2, d3; the target's verify
    ///   of [x, d1, d2, d3] at L..L+3 (tp = its rows' argmax); accept the leading d_i == tp[i-1]
    ///   (a of them), commit x, d1..da and make tp[a] the next x; clear the rejected rows' C128
    ///   ring slots; the draft's extend over the verified rows (ids tp[0..=a], 0 when rejected,
    ///   the verify's pre-hc_head states), the next d1 its row a's argmax; L += a + 1.
    /// The draft tokens pass between forwards on the device; the host reads the round's tokens
    /// once, after the verify. Each forward kind replays as a CUDA graph from its second use
    /// (DSV4_MTP_EAGER=1: never). Needs `compress::set_speculative(true)` before the model was
    /// built and the draft's weights.
    pub fn generate_mtp(&mut self, prompt: &[i64], max_new: usize) -> Result<MtpRun> {
        const DRAFTS: usize = 4;
        if !compress::speculative() { return Err("generate_mtp: speculative compressor rings are off".into()); }
        let t = prompt.len();
        if t < 2 || t > PREFILL_CHUNK { return Err(format!("generate_mtp prefills one chunk of 2..={PREFILL_CHUNK} tokens")); }
        let use_graphs = std::env::var_os("DSV4_MTP_EAGER").is_none();
        let s = self.s();
        self.serving = None;
        self.decode_w = 0;
        self.cache = KvCache::new(t + max_new + 2 * DRAFTS)?;
        self.enable_streams()?;
        let (pb, sb) = arena_bytes(PREFILL_CHUNK);
        let (mut persist, mut scratch) = (Arena::new(pb)?, Arena::new(sb)?);
        let hc_bytes = HC * HIDDEN * 2;
        // Target pre-hc_head states (prefill rows, then each verify's rows), the draft's extend
        // output, the hidden state the first draft decode reads, the two decodes' outputs.
        let (hc_t, hc_d) = (Buffer::alloc(t * hc_bytes)?, Buffer::alloc(t * hc_bytes)?);
        let (hc_src, hc_1, hc_2) = (Buffer::alloc(hc_bytes)?, Buffer::alloc(hc_bytes)?, Buffer::alloc(hc_bytes)?);
        let (wide, probs) = (Buffer::alloc(DRAFTS * VOCAB * 4)?, Buffer::alloc(VOCAB * 4)?);
        // res: the verify's ids [x, d1, d2, d3] and its argmax [4]; ext: the draft extend's argmax
        // per row; tok: a draft decode's token.
        let (res, ext, tok) = (Buffer::alloc(16 * DRAFTS)?, Buffer::alloc(8 * DRAFTS)?, Buffer::alloc(8)?);
        let (res, ext, tok, wide, probs) = (res.ptr, ext.ptr, tok.ptr, wide.ptr, probs.ptr);
        let cargs = Buffer::alloc(4 * (LAYERS + 3))?;
        s.synchronize()?;
        let t0 = std::time::Instant::now();
        // Target prefill.
        self.hc_dest = Some(hc_t.ptr);
        let keep = self.last_logits_only;
        self.last_logits_only = true;
        let b = Batch::upload(prompt, 0, &mut persist, s)?;
        let logits = self.forward(&b, &mut persist, &mut scratch, None);
        self.last_logits_only = keep;
        self.hc_dest = None;
        self.greedy_dev(logits?.wrapping_add(((t - 1) * VOCAB * 2) as u64), 1, wide, probs, res, false)?;
        // Draft extend over the shifted prompt (its last id, x, copied on the device).
        let mut ids: Vec<i64> = prompt[1..].to_vec();
        ids.push(0);
        persist.reset();
        scratch.reset();
        let b = Batch::upload(&ids, 0, &mut persist, s)?;
        crate::cuda::copy_async(b.ids + (8 * (t - 1)) as u64, res, 8, s)?;
        let logits = self.draft_forward(&b, hc_t.ptr, &[t - 1], hc_d.ptr, &mut persist, &mut scratch)?;
        self.greedy_dev(logits, 1, wide, probs, ext, true)?;
        s.synchronize()?;
        let mut x = i64::from_le_bytes(crate::cuda::download(res, 8)?.try_into().unwrap());
        let prefill_s = t0.elapsed().as_secs_f64();
        let mut tokens = vec![x as usize];
        let mut l = t;
        // Where the next round's d1 and first draft hidden state are.
        let (mut d1_src, mut hc_row) = (ext, hc_d.ptr + ((t - 1) * hc_bytes) as u64);
        let mut accepted = Vec::new();
        let (mut g_d1, mut g_d2, mut g_ext) = (GraphSlot::default(), GraphSlot::default(), GraphSlot::default());
        let mut g_verify: Vec<((VerifyKey, bool), GraphSlot)> = Vec::new();
        // Size the decode inputs region for the 4-row forwards now: graphs keep its address.
        self.decode_host_inputs(&(0..DRAFTS).map(|i| (Req::SINGLE, l + i)).collect::<Vec<_>>())?;
        while tokens.len() < max_new {
            // Draft decode 1: d1 at L (from the device), the draft's state for L - 1.
            let h = self.decode_host_inputs_opt(&[(Req::SINGLE, l)], Some(&[l - 1]), true)?;
            let (din, dl) = (self.decode_in.as_ref().unwrap().ptr, self.dl);
            crate::cuda::upload_async(din + dl.pos as u64, &h[dl.pos..], s)?;
            crate::cuda::copy_async(din + dl.ids as u64, d1_src, 8, s)?;
            crate::cuda::copy_async(res + 8, d1_src, 8, s)?;
            crate::cuda::copy_async(hc_src.ptr, hc_row, hc_bytes, s)?;
            let b = self.decode_view(Req::SINGLE, l);
            let (p, q) = (&mut persist, &mut scratch);
            g_d1.run(self, s, use_graphs, |m| {
                p.reset();
                q.reset();
                let lg = m.draft_forward(&b, hc_src.ptr, &[0], hc_1.ptr, p, q)?;
                m.greedy_dev(lg, 1, wide, probs, tok, false)?;
                crate::cuda::copy_async(din + dl.ids as u64, tok, 8, m.s())?;
                crate::cuda::copy_async(res + 16, tok, 8, m.s())
            })?;
            // Draft decode 2: d2 at L + 1, still attending to the committed window.
            let h = self.decode_host_inputs_opt(&[(Req::SINGLE, l + 1)], Some(&[l - 1]), true)?;
            crate::cuda::upload_async(din + dl.pos as u64, &h[dl.pos..], s)?;
            let b = self.decode_view(Req::SINGLE, l + 1);
            g_d2.run(self, s, use_graphs, |m| {
                p.reset();
                q.reset();
                let lg = m.draft_forward(&b, hc_1.ptr, &[0], hc_2.ptr, p, q)?;
                m.greedy_dev(lg, 1, wide, probs, tok, false)?;
                crate::cuda::copy_async(res + 24, tok, 8, m.s())
            })?;
            // Target verify of [x, d1, d2, d3].
            let rows: Vec<(Req, usize)> = (0..DRAFTS).map(|i| (Req::SINGLE, l + i)).collect();
            let segs = vec![Seg { req: Req::SINGLE, start: l, len: DRAFTS }];
            let h = self.decode_host_inputs(&rows)?;
            let (din, dl) = (self.decode_in.as_ref().unwrap().ptr, self.dl);
            crate::cuda::upload_async(din + dl.pos as u64, &h[dl.pos..], s)?;
            crate::cuda::upload_async(res, &x.to_le_bytes(), s)?;
            crate::cuda::copy_async(din + dl.ids as u64, res, 8 * DRAFTS, s)?;
            let key = (self.verify_meta(&segs)?, self.dl_indexer);
            let mut b = self.decode_view(Req::SINGLE, l);
            b.verify = true;
            b.segs = segs;
            let i = match g_verify.iter().position(|g| g.0 == key) { Some(i) => i, None => { g_verify.push((key, GraphSlot::default())); g_verify.len() - 1 } };
            let hc_v = hc_t.ptr;
            g_verify[i].1.run(self, s, use_graphs, |m| {
                p.reset();
                q.reset();
                m.hc_dest = Some(hc_v);
                let lg = m.forward(&b, p, q, None);
                m.hc_dest = None;
                m.greedy_dev(lg?, DRAFTS, wide, probs, res + (8 * DRAFTS) as u64, false)
            })?;
            s.synchronize()?;
            let v: Vec<i64> = crate::cuda::download(res, 16 * DRAFTS)?.chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().unwrap())).collect();
            let (vids, tp) = (&v[..DRAFTS], &v[DRAFTS..]);
            let mut a = 0;
            while a < DRAFTS - 1 && vids[a + 1] == tp[a] { a += 1; }
            accepted.push(a);
            tokens.extend(vids[1..=a].iter().map(|&v| v as usize));
            tokens.push(tp[a] as usize);
            self.cache.clear_c128_rejected_batch(&[(Req::SINGLE, l, a + 1)], DRAFTS, cargs.ptr, s)?;
            // Draft extend over the verified rows (the verify's decode inputs, new ids).
            let ext_ids: Vec<u8> = (0..DRAFTS).flat_map(|i| if i <= a { tp[i] } else { 0 }.to_le_bytes()).collect();
            crate::cuda::upload_async(din + dl.ids as u64, &ext_ids, s)?;
            let b = self.decode_view(Req::SINGLE, l);
            let all: Vec<usize> = (0..DRAFTS).collect();
            g_ext.run(self, s, use_graphs, |m| {
                p.reset();
                q.reset();
                let lg = m.draft_forward(&b, hc_t.ptr, &all, hc_d.ptr, p, q)?;
                m.greedy_dev(lg, DRAFTS, wide, probs, ext, false)
            })?;
            d1_src = ext + (8 * a) as u64;
            hc_row = hc_d.ptr + (a * hc_bytes) as u64;
            l += a + 1;
            x = tp[a];
        }
        s.synchronize()?;
        tokens.truncate(max_new);
        Ok(MtpRun { tokens, accepted, prefill_s, decode_s: t0.elapsed().as_secs_f64() - prefill_s })
    }

    /// Concurrent greedy MTP serving (SGLang's EAGLE loop of `generate_mtp`, batched): `schedule`
    /// prefills requests (the target's prefill, the draft's extend, the first x and d1) and runs
    /// rounds over the listed requests in batch order. A round's forwards take each request's
    /// rows in that order: the draft decodes one row per request, the verify and the draft extend
    /// four. Per-request state lives on the device indexed by request (`d1v`, `hcv`) and is
    /// gathered into batch order each round; the host reads the round's tokens once, after the
    /// verify. A request leaves the batch once it holds `max_new` tokens. Forwards replay as CUDA
    /// graphs per batch size (verify: per plan-count pattern too). DSV4_BATCH_CHECK=<gold dir>:
    /// compare every forward's logits with the reference's (eager).
    pub fn serve_mtp(&mut self, prompts: &[Vec<i64>], max_new: usize, schedule: &[MtpStep]) -> Result<BatchRun> {
        const DRAFTS: usize = 4;
        if !compress::speculative() { return Err("serve_mtp: speculative compressor rings are off".into()); }
        let (n, s) = (prompts.len(), self.s());
        let longest = prompts.iter().map(|p| p.len()).max().unwrap_or(0);
        if longest > PREFILL_CHUNK { return Err(format!("serve_mtp prefills one chunk of at most {PREFILL_CHUNK} tokens per request")); }
        let check = std::env::var("DSV4_BATCH_CHECK").ok().map(std::path::PathBuf::from);
        let use_graphs = check.is_none() && std::env::var_os("DSV4_MTP_EAGER").is_none();
        let max_b = schedule.iter().map(|st| match st { MtpStep::Round(rows) => rows.len(), _ => 1 }).max().unwrap_or(1);
        // Pages per request (a multiple of 4, as `serve`), the SWA pool: a prefill's pages, then
        // per request its window, compressor states and the pages a round writes.
        let w = ((longest + max_new + 3 * DRAFTS).div_ceil(SWA_PAGE_TOKENS) + 1).div_ceil(4) * 4;
        let swa_tokens = (4 * n + PREFILL_CHUNK.div_ceil(SWA_PAGE_TOKENS) + 3) * SWA_PAGE_TOKENS;
        let hc_bytes = HC * HIDDEN * 2;
        let rows_max = longest.max(DRAFTS * max_b);
        let key = (n, w, swa_tokens, max_b, rows_max);
        // The graphs and everything they read persist across calls of the same shape, as long as
        // nothing else reallocated the regions they were captured on (DSV4_MTP_FRESH=1: start
        // over at each call).
        let ptrs = |m: &Self| (m.decode_in.as_ref().map_or(0, |b| b.ptr), m.verify_in.as_ref().map_or(0, |b| b.ptr), m.cache.swa.ptr);
        let reuse = std::env::var_os("DSV4_MTP_FRESH").is_none() && self.mtp_serving.as_ref().map_or(false, |v| v.key == key && v.ptrs == ptrs(self));
        if reuse { self.cache.clear(s)?; } else {
            self.mtp_serving = None;
            self.serving = None;
            // The previous cache freed before the new one is allocated (both may not fit).
            self.cache = KvCache::new(SWA_PAGE_TOKENS)?;
            self.cache = KvCache::with_pools(n * w * SWA_PAGE_TOKENS, n + 1, Some(swa_tokens))?;
            self.decode_w = w;
            let dl_max = DecodeLayout::new(DRAFTS * max_b, w, self.sms());
            if self.decode_in.as_ref().map(|d| d.bytes < dl_max.bytes).unwrap_or(true) { self.decode_in = Some(Buffer::alloc(dl_max.bytes)?); }
            let (pb, sb) = arena_bytes(PREFILL_CHUNK);
            self.mtp_serving = Some(MtpServing {
                key, ptrs: (0, 0, 0), persist: Arena::new(pb)?, scratch: Arena::new(sb)?,
                hc_t: Buffer::alloc(rows_max * hc_bytes)?, hc_d: Buffer::alloc(rows_max * hc_bytes)?,
                hc_src: Buffer::alloc(max_b * hc_bytes)?, hc_1: Buffer::alloc(max_b * hc_bytes)?, hc_2: Buffer::alloc(max_b * hc_bytes)?,
                // Per request: its d1 and the draft's hidden state for its position L - 1, its x.
                d1v: Buffer::alloc(8 * n)?, hcv: Buffer::alloc(n * hc_bytes)?, xv: Buffer::alloc(8 * n)?,
                wide: Buffer::alloc(DRAFTS * max_b * VOCAB * 4)?, probs: Buffer::alloc(VOCAB * 4)?,
                // res: the verify's ids [B, 4] then its argmax [B, 4]; ext: the extend's argmax
                // [B, 4]; tok: a draft decode's tokens [B]; xb: the batch's x [B]; idx: index uploads.
                res: Buffer::alloc(16 * DRAFTS * max_b)?, ext: Buffer::alloc(8 * DRAFTS * max_b)?, tok: Buffer::alloc(8 * max_b)?,
                xb: Buffer::alloc(8 * max_b)?, idx: Buffer::alloc(4 * 3 * max_b)?, cargs: Buffer::alloc(4 * (LAYERS + 3 * max_b))?,
                // A prefill's samples ([chunks] target, then [chunks] draft).
                pre: Buffer::alloc(16 * n.max(8))?,
                g_d1: vec![], g_d2: vec![], g_ext: vec![], g_verify: vec![],
            });
        }
        self.enable_streams()?;
        let reqs: Vec<Req> = (0..n).map(|i| Req { page0: 1 + i * w, rid: 1 + i as i64 }).collect();
        let mut sv = self.mtp_serving.take().ok_or("serve_mtp: no serving state")?;
        let MtpServing { persist, scratch, hc_t, hc_d, hc_src, hc_1, hc_2, d1v, hcv, xv, wide, probs, res, ext, tok, xb, idx, cargs, pre,
                         g_d1, g_d2, g_ext, g_verify, .. } = &mut sv;
        let (res, ext, tok, xb, idx, wide, probs) = (res.ptr, ext.ptr, tok.ptr, xb.ptr, idx.ptr, wide.ptr, probs.ptr);
        let mut tokens: Vec<Vec<usize>> = vec![vec![]; n];
        // Whether a request's x waits on the device (sampled by its prefill, not yet read).
        let mut pending_x = vec![false; n];
        let mut tokens_at_prefill = 0usize;
        let mut l = vec![0usize; n];
        let mut x = vec![0i64; n];
        let mut fwd = 0usize;
        let checked = |m: &Self, fwd: usize, prefix: &str, rows: usize| -> Result<()> {
            let Some(dir) = &check else { return Ok(()) };
            crate::cuda::ctx_sync()?;
            let g = Gold::open(dir.join(format!("fwd-{fwd:03}")));
            let c = crate::gold::compare(&g, &format!("{prefix}logits_processor.out.next_token_logits"), &crate::cuda::download(wide, rows * VOCAB * 4)?)?;
            println!("{}", serde_json::json!({"fwd": fwd, "runner": if prefix.is_empty() { "target" } else { "draft" }, "rows": rows, "logits_equal": c.equal}));
            let _ = m;
            Ok(())
        };
        fn slot<K: PartialEq + Copy>(v: &mut Vec<(K, GraphSlot)>, k: K) -> usize {
            match v.iter().position(|g| g.0 == k) { Some(i) => i, None => { v.push((k, GraphSlot::default())); v.len() - 1 } }
        }
        let mut prefill_s = 0.0;
        let mut rounds = 0usize;
        // DSV4_MTP_TIMES=1: per-round host time building inputs and waiting for the verify.
        let times = std::env::var_os("DSV4_MTP_TIMES").is_some();
        // DSV4_MTP_TIMES=2 also prints one line per round: its step, batch, the tokens it commits
        // (capped at max_new) and the time its verify finished.
        let per_round = std::env::var("DSV4_MTP_TIMES").map_or(false, |v| v == "2");
        // The verify's compressed-entry lists and page table written on the device
        // (DSV4_MTP_HOST_IDS=1: built on the host, as before).
        let dev_fill = std::env::var_os("DSV4_MTP_HOST_IDS").is_none();
        let (mut host_s, mut wait_s) = (0.0f64, 0.0f64);
        let last_prefill = schedule.iter().rposition(|st| matches!(st, MtpStep::Prefill(_)));
        // Each request's last round (its SWA pages go back to the pool after it).
        let mut last_round = vec![usize::MAX; n];
        for (k, st) in schedule.iter().enumerate() { if let MtpStep::Round(rows) = st { for &i in rows { last_round[i] = k; } } }
        let run = || -> Result<BatchRun> {
        s.synchronize()?;
        let t0 = std::time::Instant::now();
        for (k, st) in schedule.iter().enumerate() {
            match st {
                MtpStep::Prefill(chunks) => {
                    // The target over the chunks (packed when several), each chunk's sequence the
                    // prompt then the tokens the request holds; a chunk that ends its sequence
                    // samples the request's x.
                    let (mut segs, mut ids, mut ends) = (Vec::with_capacity(chunks.len()), Vec::new(), Vec::with_capacity(chunks.len()));
                    let tok_at = |i: usize, p: usize| -> i64 { if p < prompts[i].len() { prompts[i][p] } else { tokens[i][p - prompts[i].len()] as i64 } };
                    for c in chunks {
                        let (i, r) = (c.req, reqs[c.req]);
                        let (known, end) = (prompts[i].len() + tokens[i].len(), c.start + c.len);
                        if end > known { return Err(format!("step {k}: prefill past request {i}'s sequence")); }
                        self.cache.swa_hold(&r, c.start, end)?;
                        ids.extend((c.start..end).map(|p| tok_at(i, p)));
                        segs.push(Seg { req: r, start: c.start, len: c.len });
                        ends.push(end == known);
                    }
                    let (t, nr) = (ids.len(), chunks.len());
                    if t > PREFILL_CHUNK || t > rows_max { return Err(format!("step {k}: a prefill of {t} tokens")); }
                    persist.reset();
                    scratch.reset();
                    fwd += 1;
                    let b = Batch::upload_segs(&ids, &segs, &mut *persist, s)?;
                    self.hc_dest = Some(hc_t.ptr);
                    self.last_logits_only = true;
                    let logits = self.forward(&b, &mut *persist, &mut *scratch, None);
                    self.last_logits_only = false;
                    self.hc_dest = None;
                    // One request: its last row (the base indexes the chunk's rows); several: [nr, vocab].
                    let rows_at = if nr == 1 { logits?.wrapping_add(((t - 1) * VOCAB * 2) as u64) } else { logits? };
                    self.greedy_dev(rows_at, nr, wide, probs, pre.ptr, false)?;
                    checked(self, fwd, "", nr)?;
                    // The draft's extend over the same chunks, each shifted by one: its last id is
                    // the sampled x when the chunk ends the sequence, else the sequence's next token.
                    let mut ids2 = Vec::with_capacity(t);
                    let mut patches = Vec::new();
                    let mut last_rows = Vec::with_capacity(nr);
                    for (j, c) in chunks.iter().enumerate() {
                        let end = c.start + c.len;
                        ids2.extend((c.start + 1..end).map(|p| tok_at(c.req, p)));
                        if ends[j] { patches.push((ids2.len(), j)); ids2.push(0); } else { ids2.push(tok_at(c.req, end)); }
                        last_rows.push(ids2.len() - 1);
                    }
                    persist.reset();
                    scratch.reset();
                    fwd += 1;
                    let b = Batch::upload_segs(&ids2, &segs, &mut *persist, s)?;
                    for &(row, j) in &patches { crate::cuda::copy_async(b.ids + (8 * row) as u64, pre.ptr + (8 * j) as u64, 8, s)?; }
                    let logits = self.draft_forward(&b, hc_t.ptr, &last_rows, hc_d.ptr, &mut *persist, &mut *scratch)?;
                    self.greedy_dev(logits, nr, wide, probs, pre.ptr + (8 * nr) as u64, true)?;
                    checked(self, fwd, "draft.", nr)?;
                    for (j, c) in chunks.iter().enumerate() {
                        let (i, end) = (c.req, c.start + c.len);
                        self.cache.swa_evict(&reqs[i], end);
                        if !ends[j] { continue; }
                        crate::cuda::copy_async(xv.ptr + (8 * i) as u64, pre.ptr + (8 * j) as u64, 8, s)?;
                        crate::cuda::copy_async(d1v.ptr + (8 * i) as u64, pre.ptr + (8 * (nr + j)) as u64, 8, s)?;
                        crate::cuda::copy_async(hcv.ptr + (i * hc_bytes) as u64, hc_d.ptr + (last_rows[j] * hc_bytes) as u64, hc_bytes, s)?;
                        l[i] = end;
                        pending_x[i] = true;
                    }
                    if Some(k) == last_prefill {
                        s.synchronize()?;
                        prefill_s = t0.elapsed().as_secs_f64();
                        // Each request's tokens so far, its pending x included.
                        tokens_at_prefill = tokens.iter().map(|t| t.len()).sum::<usize>() + pending_x.iter().filter(|&&p| p).count();
                    }
                }
                MtpStep::Retract { req, keep } => {
                    // Its cache is gone (re-prefilled later, prompt plus the tokens it keeps).
                    let i = *req;
                    self.cache.swa_release(&reqs[i]);
                    if *keep > tokens[i].len() { return Err(format!("step {k}: request {i} keeps {keep} of {} tokens", tokens[i].len())); }
                    tokens[i].truncate(*keep);
                    l[i] = 0;
                    pending_x[i] = false;
                }
                MtpStep::Round(rows) => {
                    let bsz = rows.len();
                    // Requests in their first round after a prefill: its sampled token, x.
                    if rows.iter().any(|&i| pending_x[i]) {
                        s.synchronize()?;
                        let xs = crate::cuda::download(xv.ptr, 8 * n)?;
                        for &i in rows {
                            if std::mem::take(&mut pending_x[i]) {
                                x[i] = i64::from_le_bytes(xs[8 * i..8 * i + 8].try_into().unwrap());
                                tokens[i].push(x[i] as usize);
                            }
                        }
                    }
                    for &i in rows {
                        if l[i] == 0 { return Err(format!("step {k}: request {i} decodes before its prefill")); }
                        // SGLang's overlap scheduler runs a request's next round before it reads the
                        // previous one's tokens, so a finished request can take one more round (its
                        // tokens dropped); more than that is a schedule error.
                        if tokens[i].len() >= max_new + DRAFTS { return Err(format!("step {k}: request {i} already holds {} tokens", tokens[i].len())); }
                        self.cache.swa_hold(&reqs[i], l[i], l[i] + DRAFTS)?;
                    }
                    let ridx: Vec<u8> = rows.iter().flat_map(|&i| (i as i32).to_le_bytes()).collect();
                    crate::cuda::upload_async(idx, &ridx, s)?;
                    // Draft decode 1: each request's d1 at L, its draft state for L - 1.
                    let place: Vec<(Req, usize)> = rows.iter().map(|&i| (reqs[i], l[i])).collect();
                    let win: Vec<usize> = rows.iter().map(|&i| l[i] - 1).collect();
                    let th = std::time::Instant::now();
                    let h = self.decode_host_inputs_opt(&place, Some(&win), true)?;
                    host_s += th.elapsed().as_secs_f64();
                    let (din, dl) = (self.decode_in.as_ref().unwrap().ptr, self.dl);
                    crate::cuda::upload_async(din + dl.pos as u64, &h[dl.pos..], s)?;
                    crate::exact::copy_rows(s, d1v.ptr, idx, din + dl.ids as u64, 0, bsz, 8)?;
                    crate::exact::scatter_strided_i64(s, din + dl.ids as u64, res, bsz, DRAFTS, 1)?;
                    crate::exact::copy_rows(s, hcv.ptr, idx, hc_src.ptr, 0, bsz, hc_bytes)?;
                    let all: Vec<usize> = (0..bsz).collect();
                    let b = self.decode_view(place[0].0, place[0].1);
                    let (p, q) = (&mut *persist, &mut *scratch);
                    fwd += 1;
                    let gi = slot(g_d1, bsz);
                    g_d1[gi].1.run(self, s, use_graphs, |m| {
                        p.reset();
                        q.reset();
                        let lg = m.draft_forward(&b, hc_src.ptr, &all, hc_1.ptr, p, q)?;
                        m.greedy_dev(lg, bsz, wide, probs, tok, false)?;
                        crate::cuda::copy_async(din + dl.ids as u64, tok, 8 * bsz, m.s())?;
                        crate::exact::scatter_strided_i64(m.s(), tok, res, bsz, DRAFTS, 2)
                    })?;
                    checked(self, fwd, "draft.", bsz)?;
                    // Draft decode 2: d2 at L + 1, the same committed windows.
                    let place2: Vec<(Req, usize)> = rows.iter().map(|&i| (reqs[i], l[i] + 1)).collect();
                    let th = std::time::Instant::now();
                    let h = self.decode_host_inputs_opt(&place2, Some(&win), true)?;
                    host_s += th.elapsed().as_secs_f64();
                    crate::cuda::upload_async(din + dl.pos as u64, &h[dl.pos..], s)?;
                    let b = self.decode_view(place2[0].0, place2[0].1);
                    fwd += 1;
                    let gi = slot(g_d2, bsz);
                    g_d2[gi].1.run(self, s, use_graphs, |m| {
                        p.reset();
                        q.reset();
                        let lg = m.draft_forward(&b, hc_1.ptr, &all, hc_2.ptr, p, q)?;
                        m.greedy_dev(lg, bsz, wide, probs, tok, false)?;
                        crate::exact::scatter_strided_i64(m.s(), tok, res, bsz, DRAFTS, 3)
                    })?;
                    checked(self, fwd, "draft.", bsz)?;
                    // Target verify of each request's [x, d1, d2, d3].
                    let vrows: Vec<(Req, usize)> = rows.iter().flat_map(|&i| (0..DRAFTS).map(move |j| (i, j))).map(|(i, j)| (reqs[i], l[i] + j)).collect();
                    let segs: Vec<Seg> = rows.iter().map(|&i| Seg { req: reqs[i], start: l[i], len: DRAFTS }).collect();
                    let th = std::time::Instant::now();
                    if dev_fill { self.upload_decode_inputs_dev(&vrows, s)?; } else {
                        let h = self.decode_host_inputs(&vrows)?;
                        let (din, dl) = (self.decode_in.as_ref().unwrap().ptr, self.dl);
                        crate::cuda::upload_async(din + dl.pos as u64, &h[dl.pos..], s)?;
                    }
                    host_s += th.elapsed().as_secs_f64();
                    let (din, dl) = (self.decode_in.as_ref().unwrap().ptr, self.dl);
                    let xs: Vec<u8> = rows.iter().flat_map(|&i| x[i].to_le_bytes()).collect();
                    crate::cuda::upload_async(xb, &xs, s)?;
                    crate::exact::scatter_strided_i64(s, xb, res, bsz, DRAFTS, 0)?;
                    crate::cuda::copy_async(din + dl.ids as u64, res, 8 * DRAFTS * bsz, s)?;
                    let key = (bsz, self.verify_meta(&segs)?, self.dl_indexer);
                    let mut b = self.decode_view(vrows[0].0, vrows[0].1);
                    b.verify = true;
                    b.segs = segs;
                    let hc_v = hc_t.ptr;
                    let tp_dev = res + (8 * DRAFTS * bsz) as u64;
                    fwd += 1;
                    let gi = slot(g_verify, key);
                    let vkind = g_verify[gi].1.next(use_graphs);
                    g_verify[gi].1.run(self, s, use_graphs, |m| {
                        p.reset();
                        q.reset();
                        m.hc_dest = Some(hc_v);
                        let lg = m.forward(&b, p, q, None);
                        m.hc_dest = None;
                        m.greedy_dev(lg?, DRAFTS * bsz, wide, probs, tp_dev, false)
                    })?;
                    checked(self, fwd, "", DRAFTS * bsz)?;
                    let tw = std::time::Instant::now();
                    s.synchronize()?;
                    wait_s += tw.elapsed().as_secs_f64();
                    let v: Vec<i64> = crate::cuda::download(res, 16 * DRAFTS * bsz)?.chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().unwrap())).collect();
                    let (vids, tp) = v.split_at(DRAFTS * bsz);
                    let t_verify = t0.elapsed().as_secs_f64();
                    let held: usize = rows.iter().map(|&i| tokens[i].len().min(max_new)).sum();
                    let mut acc = Vec::with_capacity(bsz);
                    let mut ext_ids: Vec<u8> = Vec::with_capacity(8 * DRAFTS * bsz);
                    for (j, &i) in rows.iter().enumerate() {
                        let (vj, tj) = (&vids[DRAFTS * j..DRAFTS * (j + 1)], &tp[DRAFTS * j..DRAFTS * (j + 1)]);
                        let mut a = 0;
                        while a < DRAFTS - 1 && vj[a + 1] == tj[a] { a += 1; }
                        tokens[i].extend(vj[1..=a].iter().map(|&v| v as usize));
                        tokens[i].push(tj[a] as usize);
                        ext_ids.extend((0..DRAFTS).flat_map(|r| if r <= a { tj[r] } else { 0 }.to_le_bytes()));
                        acc.push(a);
                    }
                    if per_round {
                        let now: usize = rows.iter().map(|&i| tokens[i].len().min(max_new)).sum();
                        println!("{}", serde_json::json!({"round": k, "bsz": bsz, "tokens": now - held,
                            "raw_tokens": acc.iter().map(|a| a + 1).sum::<usize>(), "t_verify": t_verify, "verify": vkind}));
                    }
                    let clears: Vec<(Req, usize, usize)> = rows.iter().zip(&acc).map(|(&i, &a)| (reqs[i], l[i], a + 1)).collect();
                    self.cache.clear_c128_rejected_batch(&clears, DRAFTS, cargs.ptr, s)?;
                    // Draft extend over every request's verified rows.
                    crate::cuda::upload_async(din + dl.ids as u64, &ext_ids, s)?;
                    let b = self.decode_view(vrows[0].0, vrows[0].1);
                    let all4: Vec<usize> = (0..DRAFTS * bsz).collect();
                    fwd += 1;
                    let gi = slot(g_ext, bsz);
                    g_ext[gi].1.run(self, s, use_graphs, |m| {
                        p.reset();
                        q.reset();
                        let lg = m.draft_forward(&b, hc_t.ptr, &all4, hc_d.ptr, p, q)?;
                        m.greedy_dev(lg, DRAFTS * bsz, wide, probs, ext, false)
                    })?;
                    checked(self, fwd, "draft.", DRAFTS * bsz)?;
                    // Each request's next d1 and draft state: its extend row a.
                    let moves: Vec<u8> = acc.iter().enumerate().flat_map(|(j, &a)| ((DRAFTS * j + a) as i32).to_le_bytes())
                        .chain(rows.iter().flat_map(|&i| (i as i32).to_le_bytes())).collect();
                    crate::cuda::upload_async(idx + (4 * max_b) as u64, &moves, s)?;
                    let (src_i, dst_i) = (idx + (4 * max_b) as u64, idx + (4 * (max_b + bsz)) as u64);
                    crate::exact::copy_rows(s, ext, src_i, d1v.ptr, dst_i, bsz, 8)?;
                    crate::exact::copy_rows(s, hc_d.ptr, src_i, hcv.ptr, dst_i, bsz, hc_bytes)?;
                    for (j, &i) in rows.iter().enumerate() {
                        let a = acc[j];
                        l[i] += a + 1;
                        x[i] = tp[DRAFTS * j + a];
                        if k == last_round[i] { self.cache.swa_release(&reqs[i]); } else { self.cache.swa_evict(&reqs[i], l[i]); }
                    }
                    rounds += 1;
                }
            }
        }
        s.synchronize()?;
        let wall_s = t0.elapsed().as_secs_f64();
        if times { println!("{}", serde_json::json!({"mtp_rounds": rounds, "host_inputs_s": host_s, "verify_wait_s": wait_s, "decode_s": wall_s - prefill_s})); }
        for t in tokens.iter_mut() { t.truncate(max_new); }
        let total: usize = tokens.iter().map(|t| t.len()).sum();
        Ok(BatchRun { tokens, prefill_s, decode_s: wall_s - prefill_s, decode_steps: rounds, wall_s, decode_tokens: total.saturating_sub(tokens_at_prefill) })
        };
        let r = run();
        // A failed run may leave a capture half done: its state is dropped.
        if r.is_ok() { sv.ptrs = ptrs(self); self.mtp_serving = Some(sv); }
        r
    }

    /// Debug: SGLang's MTP run `run` forward by forward (`mtp_target_check` for the target): the
    /// target's prefill and verifies, and every draft forward (its prefill extend, decode steps and
    /// extends after verify) fed the golds' token ids and the hidden states this port computed,
    /// each compared with the reference's logits and pre-hc_head states. Draft weights from `ckpt`.
    pub fn mtp_draft_check(&mut self, run: &std::path::Path, ckpt: &std::path::Path) -> Result<()> {
        self.load_draft_weights(ckpt)?;
        let prompts: Vec<Vec<i64>> = serde_json::from_slice(&std::fs::read(run.join("prompts.json")).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
        let prompt = &prompts[0];
        let t = prompt.len();
        if t > PREFILL_CHUNK { return Err("the MTP check prefills one chunk".into()); }
        let gold = run.join("gold");
        let s = self.s();
        self.serving = None;
        self.decode_w = 0;
        self.cache = KvCache::new(t + 256)?;
        self.enable_streams()?;
        let (pb, sb) = arena_bytes(PREFILL_CHUNK);
        let (mut persist, mut scratch) = (Arena::new(pb)?, Arena::new(sb)?);
        // Hidden states: the target's prefill (every row) and verify, the draft's last forward.
        let hc_bytes = HC * HIDDEN * 2;
        let (hc_t, hc_d) = (Buffer::alloc(t * hc_bytes)?, Buffer::alloc(t * hc_bytes)?);
        let mut d_rows = 0usize;       // rows of the draft's last output in hc_d
        let mut d_last_pos = 0usize;   // position of hc_d's first row
        let mut target_prefilled = false;
        for kf in 1.. {
            let dir = gold.join(format!("fwd-{kf:03}"));
            if !dir.join("meta.json").exists() { break; }
            let meta: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("meta.json")).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
            let g = Gold::open(&dir);
            let (ids, pos) = (gold_i64(&g, "batch.input_ids")?, gold_i64(&g, "batch.positions")?);
            let seq = gold_i64(&g, "batch.seq_lens")?;
            let mode = meta["forward_mode"].as_str().unwrap_or("").to_string();
            persist.reset();
            scratch.reset();
            if meta["runner"] == "target" {
                self.hc_dest = Some(hc_t.ptr);
                if !target_prefilled {
                    let keep = self.last_logits_only;
                    self.last_logits_only = true;
                    let b = Batch::upload(prompt, 0, &mut persist, s)?;
                    let logits = self.forward(&b, &mut persist, &mut scratch, None);
                    self.last_logits_only = keep;
                    self.compare_spec(&dir, logits?.wrapping_add(((t - 1) * VOCAB * 2) as u64), 1, hc_t.ptr, t, &mut scratch)?;
                    target_prefilled = true;
                } else {
                    let rows: Vec<(Req, usize)> = pos.iter().map(|&p| (Req::SINGLE, p as usize)).collect();
                    let segs = vec![Seg { req: Req::SINGLE, start: pos[0] as usize, len: pos.len() }];
                    let logits = self.verify_step(&rows, &segs, &ids, None, &mut persist, &mut scratch)?;
                    self.compare_spec(&dir, logits, rows.len(), hc_t.ptr, rows.len(), &mut scratch)?;
                }
                self.hc_dest = None;
                continue;
            }
            let n = ids.len();
            let out = Buffer::alloc(n * hc_bytes)?;
            let (logits, rows) = match mode.as_str() {
                // EXTEND: the draft's prefill of the prompt (ids shifted by one), the target's states.
                "1" => {
                    let b = Batch::upload(&ids, pos[0] as usize, &mut persist, s)?;
                    (self.draft_forward(&b, hc_t.ptr, &[n - 1], out.ptr, &mut persist, &mut scratch)?, 1)
                }
                // DECODE: one draft step at position p, attending to the committed window
                // (seq_lens), its state the draft's previous output row for position p - 1.
                "2" => {
                    let p = pos[0] as usize;
                    let src = hc_d.ptr + ((p - 1 - d_last_pos).min(d_rows - 1) * hc_bytes) as u64;
                    let rows_in = [(Req::SINGLE, p)];
                    let mut h = self.decode_host_inputs_win(&rows_in, Some(&[seq[0] as usize - 1]))?;
                    let dl = self.dl;
                    h[dl.ids..dl.ids + 8].copy_from_slice(&ids[0].to_le_bytes());
                    crate::cuda::upload_async(self.decode_in.as_ref().unwrap().ptr, &h, s)?;
                    let b = self.decode_view(Req::SINGLE, p);
                    let hc_in = Buffer::alloc(hc_bytes)?;
                    crate::cuda::copy_async(hc_in.ptr, src, hc_bytes, s)?;
                    (self.draft_forward(&b, hc_in.ptr, &[0], out.ptr, &mut persist, &mut scratch)?, 1)
                }
                // DRAFT_EXTEND_V2: the verified rows again (target verify states), each attending
                // to its own window.
                _ => {
                    let rows_in: Vec<(Req, usize)> = pos.iter().map(|&p| (Req::SINGLE, p as usize)).collect();
                    let mut h = self.decode_host_inputs(&rows_in)?;
                    let dl = self.dl;
                    for (i, id) in ids.iter().enumerate() { h[dl.ids + 8 * i..dl.ids + 8 * i + 8].copy_from_slice(&id.to_le_bytes()); }
                    crate::cuda::upload_async(self.decode_in.as_ref().unwrap().ptr, &h, s)?;
                    let b = self.decode_view(Req::SINGLE, pos[0] as usize);
                    let all: Vec<usize> = (0..n).collect();
                    (self.draft_forward(&b, hc_t.ptr, &all, out.ptr, &mut persist, &mut scratch)?, n)
                }
            };
            // The gold's hidden states: the rows the logits processor kept.
            let hrows = g.meta("draft.logits_processor.out.hidden_states").map(|m| m.shape[0]).unwrap_or(rows);
            let lrows = g.meta("draft.logits_processor.out.next_token_logits").map(|m| m.shape[0]).unwrap_or(rows);
            let hsrc = out.ptr + ((n - hrows) * hc_bytes) as u64;
            let lsrc = logits + ((rows - lrows.min(rows)) * VOCAB * 2) as u64;
            self.compare_spec_as(&dir, "draft.", lsrc, lrows.min(rows), hsrc, hrows, &mut scratch)?;
            crate::cuda::copy_async(hc_d.ptr, out.ptr, n * hc_bytes, s)?;
            d_rows = n;
            d_last_pos = pos[0] as usize;
        }
        Ok(())
    }

    /// Debug: the target half of SGLang's MTP run `run` (a concurrent gold run of one request with
    /// `--speculative-algorithm EAGLE`): its prefill, then every TARGET_VERIFY forward fed the
    /// golds' token ids and positions, each compared with the reference's next-token logits (fp32)
    /// and pre-hc_head hidden states. Needs `compress::set_speculative(true)` before the model's
    /// cache is built (the verify-sized compressor rings).
    pub fn mtp_target_check(&mut self, run: &std::path::Path) -> Result<()> {
        let prompts: Vec<Vec<i64>> = serde_json::from_slice(&std::fs::read(run.join("prompts.json")).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
        let ids = &prompts[0];
        if ids.len() > PREFILL_CHUNK { return Err("the MTP target check prefills one chunk".into()); }
        let gold = run.join("gold");
        let s = self.s();
        self.serving = None;
        self.decode_w = 0;
        self.cache = KvCache::new(ids.len() + 256)?;
        self.enable_streams()?;
        let (pb, sb) = arena_bytes(PREFILL_CHUNK);
        let (mut persist, mut scratch) = (Arena::new(pb)?, Arena::new(sb)?);
        let hc = Buffer::alloc(ids.len() * HC * HIDDEN * 2)?;
        self.hc_dest = Some(hc.ptr);
        let keep = self.last_logits_only;
        self.last_logits_only = true;
        let b = Batch::upload(ids, 0, &mut persist, s)?;
        let logits = self.forward(&b, &mut persist, &mut scratch, None);
        self.last_logits_only = keep;
        let t = ids.len();
        self.compare_spec(&gold.join("fwd-001"), logits?.wrapping_add(((t - 1) * VOCAB * 2) as u64), 1, hc.ptr, t, &mut scratch)?;
        for k in 2.. {
            let dir = gold.join(format!("fwd-{k:03}"));
            if !dir.join("meta.json").exists() { break; }
            let meta: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("meta.json")).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
            if meta["runner"] != "target" { continue; }
            let g = Gold::open(&dir);
            let (ids, pos) = (gold_i64(&g, "batch.input_ids")?, gold_i64(&g, "batch.positions")?);
            let rows: Vec<(Req, usize)> = pos.iter().map(|&p| (Req::SINGLE, p as usize)).collect();
            let segs = vec![Seg { req: Req::SINGLE, start: pos[0] as usize, len: pos.len() }];
            scratch.reset();
            let logits = self.verify_step(&rows, &segs, &ids, None, &mut persist, &mut scratch)?;
            self.compare_spec(&dir, logits, rows.len(), hc.ptr, rows.len(), &mut scratch)?;
        }
        self.hc_dest = None;
        Ok(())
    }

    /// Compare one forward's logits (bf16 [rows, vocab], as fp32) and hc state with a gold forward.
    fn compare_spec(&self, dir: &std::path::Path, logits: u64, rows: usize, hc: u64, hc_rows: usize, a: &mut Arena) -> Result<()> {
        self.compare_spec_as(dir, "", logits, rows, hc, hc_rows, a)
    }

    /// `compare_spec` against the gold's `{prefix}logits_processor.*` (`draft.` for the MTP draft).
    #[allow(clippy::too_many_arguments)]
    fn compare_spec_as(&self, dir: &std::path::Path, prefix: &str, logits: u64, rows: usize, hc: u64, hc_rows: usize, a: &mut Arena) -> Result<()> {
        let wide = a.alloc(rows * VOCAB * 4)?;
        torch_ops::copy_bf16_to_f32(self.k, self.s(), wide, logits, rows * VOCAB)?;
        crate::cuda::ctx_sync()?;
        let g = Gold::open(dir);
        let l = crate::gold::compare(&g, &format!("{prefix}logits_processor.out.next_token_logits"), &crate::cuda::download(wide, rows * VOCAB * 4)?)?;
        let h = crate::gold::compare(&g, &format!("{prefix}logits_processor.out.hidden_states"), &crate::cuda::download(hc, hc_rows * HC * HIDDEN * 2)?);
        println!("{}", serde_json::json!({"fwd": dir.file_name().map(|f| f.to_string_lossy().to_string()), "rows": rows, "logits_equal": l.equal,
                                         "hidden_equal": h.as_ref().map(|c| c.equal).map_err(|e| e.clone())}));
        Ok(())
    }

    /// The decode batch reading the decode inputs region (as last laid out by `decode_host_inputs`).
    fn decode_view(&self, req: Req, pos: usize) -> Batch {
        let (base, dl) = (self.decode_in.as_ref().map(|b| b.ptr).unwrap_or(0), self.dl);
        Batch { t: dl.b, start: pos, ids: base + dl.ids as u64, positions: base + dl.pos as u64, pos32: base + dl.pos32 as u64, decode: true, req, segs: vec![], verify: false, _own: vec![] }
    }

    /// The decode inputs of requests at positions `rows` on the host, except the token ids (which
    /// only the previous step's sample decides): everything else can be prepared while the previous
    /// step runs. Sets the layout and the indexer regime of the step.
    fn decode_host_inputs(&mut self, rows: &[(Req, usize)]) -> Result<Vec<u8>> { self.decode_host_inputs_win(rows, None) }

    /// `decode_host_inputs` with each row's SWA window ending at `win_end[i]` instead of its own
    /// position: the MTP draft's decode steps attend to the committed window only (SGLang builds
    /// their metadata from the committed seq_lens while the token sits past it).
    fn decode_host_inputs_win(&mut self, rows: &[(Req, usize)], win_end: Option<&[usize]>) -> Result<Vec<u8>> {
        self.decode_host_inputs_opt(rows, win_end, false)
    }

    /// `decode_host_inputs_win`; with `swa_only`, just the inputs an SWA-only layer reads (ids,
    /// positions, window slots: the region before `c4_plan`), as the MTP draft's decodes need.
    fn decode_host_inputs_opt(&mut self, rows: &[(Req, usize)], win_end: Option<&[usize]>, swa_only: bool) -> Result<Vec<u8>> {
        self.decode_host_inputs_full(rows, win_end, swa_only, false)
    }

    /// `decode_host_inputs_opt`; with `dev_fill`, the bytes before `dl.dev_start` only, with the
    /// parameters from which `exact::fill_decode_ids` writes c4_ids, c128_ids and the page table on
    /// the device (the same values; at 1,024 verify rows they are 35 MB of the 36 MB).
    fn decode_host_inputs_full(&mut self, rows: &[(Req, usize)], win_end: Option<&[usize]>, swa_only: bool, dev_fill: bool) -> Result<Vec<u8>> {
        let b = rows.len();
        let w = if self.decode_w == 0 { self.cache.capacity_width() } else { self.decode_w };
        let dl = DecodeLayout::new(b, w, self.sms());
        if self.decode_in.as_ref().map(|d| d.bytes < dl.bytes).unwrap_or(true) {
            self.serving = None; // graphs captured on the old region
            self.decode_in = Some(Buffer::alloc(dl.bytes)?);
        }
        self.dl = dl;
        let mut h = vec![0u8; if swa_only { dl.c4_plan } else if dev_fill { dl.dev_start } else { dl.bytes }];
        let put = |h: &mut Vec<u8>, off: usize, bytes: &[u8]| h[off..off + bytes.len()].copy_from_slice(bytes);
        let mut ctx = Vec::with_capacity(b);
        for (i, &(r, pos)) in rows.iter().enumerate() {
            let seq_len = pos + 1;
            if seq_len.div_ceil(SWA_PAGE_TOKENS) > w { return Err("decode past the request's page-table width".into()); }
            put(&mut h, dl.pos + 8 * i, &(pos as i64).to_le_bytes());
            put(&mut h, dl.pos32 + 4 * i, &(pos as i32).to_le_bytes());
            put(&mut h, dl.swa_slot + 4 * i, &self.cache.swa_slot(&r, pos).to_le_bytes());
            let last = win_end.map(|w| w[i]).unwrap_or(pos);
            let swa_len = (last + 1).min(WINDOW);
            put(&mut h, dl.swa_len + 4 * i, &(swa_len as i32).to_le_bytes());
            let mut idx = vec![-1i32; WINDOW];
            for j in 0..swa_len { idx[j] = self.cache.swa_slot(&r, last - j); }
            put(&mut h, dl.swa_idx + 4 * WINDOW * i, &le_i32(&idx));
            if swa_only { continue; }
            // C4: plan, the slot of the entry this step completes, every entry while at most 512.
            let n4 = seq_len / 4;
            ctx.push(n4 as i32);
            put(&mut h, dl.c4_ctx + 4 * i, &(n4 as i32).to_le_bytes());
            put(&mut h, dl.c4_plan + 16 * i, &compress::decode_plan(4, seq_len as i32, r.rid, |p| self.cache.swa_slot(&r, p as usize)).bytes());
            put(&mut h, dl.c4_loc + 8 * i, &(if seq_len % 4 == 0 { r.c4_slot(n4 - 1) } else { 0 } as i64).to_le_bytes());
            // SGLang's c4 sparse lengths: clamped to [1, 512] (FlashMLA's extra range is never empty).
            put(&mut h, dl.c4_len + 4 * i, &(n4.clamp(1, INDEX_TOPK) as i32).to_le_bytes());
            let n128 = seq_len / 128;
            if dev_fill {
                let f = [r.c4_slot(0), if n4 <= INDEX_TOPK { n4 as i32 } else { 0 }, r.c128_slot(0), n128.min(C128_DECODE_WIDTH) as i32, r.page0 as i32];
                put(&mut h, dl.fill + 20 * i, &le_i32(&f));
            }
            put(&mut h, dl.c128_plan + 16 * i, &compress::decode_plan(128, seq_len as i32, r.rid, |p| self.cache.swa_slot(&r, p as usize)).bytes());
            put(&mut h, dl.c128_loc + 8 * i, &(if seq_len % 128 == 0 { r.c128_slot(n128 - 1) } else { 0 } as i64).to_le_bytes());
            put(&mut h, dl.c128_len + 4 * i, &(n128.max(1) as i32).to_le_bytes());
            if dev_fill { continue; }
            let mut ids4 = vec![-1i32; INDEX_TOPK];
            if n4 <= INDEX_TOPK { for e in 0..n4 { ids4[e] = r.c4_slot(e); } }
            put(&mut h, dl.c4_ids + 4 * INDEX_TOPK * i, &le_i32(&ids4));
            // C128: same, with the length clamped to 1 (FlashMLA's extra range is never empty).
            let mut ids128 = vec![-1i32; C128_DECODE_WIDTH];
            for e in 0..n128 { ids128[e] = r.c128_slot(e); }
            put(&mut h, dl.c128_ids + 4 * C128_DECODE_WIDTH * i, &le_i32(&ids128));
            put(&mut h, dl.table + 4 * w * i, &le_i32(&(0..w).map(|p| (r.page0 + p) as i32).collect::<Vec<_>>()));
        }
        if swa_only { return Ok(h); }
        // The sparse indexer runs when some request holds more than 512 entries (it then picks the
        // entries of every request; a row of at most 512 takes all of its own, in order).
        self.dl_indexer = ctx.iter().any(|&n| n as usize > INDEX_TOPK);
        if self.dl_indexer { put(&mut h, dl.sched, &le_i32(&deepgemm::paged_mqa_schedule(&ctx, self.sms()))); }
        Ok(h)
    }

    /// The full decode inputs of `rows`: the host part (`decode_host_inputs_full` with `dev_fill`)
    /// uploaded, then c4_ids, c128_ids and the page table written on the device.
    fn upload_decode_inputs_dev(&mut self, rows: &[(Req, usize)], s: Stream) -> Result<()> {
        let h = self.decode_host_inputs_full(rows, None, false, true)?;
        let (din, dl) = (self.decode_in.as_ref().unwrap().ptr, self.dl);
        crate::cuda::upload_async(din + dl.pos as u64, &h[dl.pos..], s)?;
        crate::exact::fill_decode_ids(s, din + dl.fill as u64, din + dl.c4_ids as u64, din + dl.c128_ids as u64, din + dl.table as u64,
                                      dl.b, INDEX_TOPK, C128_DECODE_WIDTH, dl.w)
    }

    pub fn attention_decode(&mut self, layer: usize, x: u64, b: &Batch, a: &mut Arena) -> Result<u64> {
        let (s, bs) = (self.s(), b.t);
        let l = format!("model.layers.{layer}.self_attn");
        let ratio = compress_ratio(layer);
        let (din, dl) = (self.decode_in.as_ref().ok_or("decode without decode inputs")?.ptr, self.dl);
        let indexer = ratio == 4 && self.dl_indexer;
        // SGLang's overlap: the compressor (side 0) and the indexer's compressor and weights (side 1)
        // read only x; the KV store (side 2) starts after wqkv_a; the indexer's query, logits and
        // top-k (side 1) after the FP8 q_lora; all join before FlashMLA.
        let (s_comp, s_idx, s_kv) = (if ratio != 0 { self.side(0) } else { None }, if ratio == 4 { self.side(1) } else { None }, self.side(2));
        self.fork(s_comp)?;
        self.fork(s_idx)?;
        if ratio != 0 { self.on(s_comp, |m| m.compressor_decode(layer, x, b, a))?; }
        let index_w = if ratio == 4 { self.on(s_idx, |m| m.indexer_compressor_decode(layer, x, b, indexer, a))? } else { None };
        let qkv_a = self.attn_qkv_a(layer, x, b, a)?;
        self.fork(s_kv)?;
        self.on(s_kv, |m| m.store_swa(layer, qkv_a, b, a))?;
        let lq = self.attn_q_lora(layer, qkv_a, b, a)?;
        // C4 beyond 512 entries: the sparse indexer picks the entries FlashMLA reads.
        let c4_topk = match index_w {
            Some(w) => { self.fork(s_idx)?; Some(self.on(s_idx, |m| m.c4_indexer_decode(layer, lq, w, b, a))?) }
            None => None,
        };
        let q_rot = self.attn_q_rot(layer, lq, b, a)?;
        self.join(s_comp)?;
        self.join(s_idx)?;
        self.join(s_kv)?;
        // Window indices (newest first) and the compressed entries, from the decode inputs.
        let (indices, topk_length) = (din + dl.swa_idx as u64, din + dl.swa_len as u64);
        let (extra_kv, extra_blocks, extra_page, extra_page_bytes, extra_topk, extra_indices, extra_topk_length) = match ratio {
            0 => (0u64, 0usize, 0usize, 0usize, 0usize, 0u64, 0u64),
            4 => (self.cache.c4_layer(layer), self.cache.c4_pages, C4_PAGE_TOKENS, C4_PAGE_BYTES, INDEX_TOPK, c4_topk.unwrap_or(din + dl.c4_ids as u64),
                  din + dl.c4_len as u64),
            _ => (self.cache.c128_layer(layer), self.cache.c128_pages, C128_PAGE_TOKENS, C128_PAGE_BYTES, C128_DECODE_WIDTH,
                  din + dl.c128_ids as u64, din + dl.c128_len as u64),
        };
        let qn = HEADS * HEAD_DIM;
        let o = a.alloc(bs * qn * 2)?;
        let lse = a.alloc(bs * HEADS * 4)?;
        // Split-KV partials for (b + sm parts) splits of [64 heads x 512] fp32, plus the schedule.
        let scratch_bytes = (24usize << 20).max((bs + self.sms() + 1) * HEADS * (HEAD_DIM + 1) * 4 + (4 << 20));
        let scratch = a.alloc(scratch_bytes)?;
        let (sched, num_splits, sched_ready) = self.c128_sched.ok_or("decode attention outside a forward")?;
        if ratio == 128 { self.c128_sched = Some((sched, num_splits, true)); }
        let args = flashmla::DecodeArgs {
            b: bs as i32, s_q: 1, h_q: HEADS as i32, topk: WINDOW as i32, num_blocks: self.cache.swa_pages as i32, page_block_size: SWA_PAGE_TOKENS as i32,
            sm_scale: f32::from_bits(0x3d3504f3),
            q: q_rot, kv: self.cache.swa_layer(layer), indices, topk_length, attn_sink: self.w.get(&format!("{l}.attn_sink"))?, lse, out: o,
            extra_num_blocks: extra_blocks as i32, extra_page_block_size: extra_page as i32, extra_topk: extra_topk as i32,
            extra_kv, extra_indices, extra_topk_length,
            stride_q_b: qn as i32, stride_q_s_q: qn as i32, stride_q_h_q: HEAD_DIM as i32,
            stride_kv_block: SWA_PAGE_BYTES as i32, stride_kv_row: 584,
            stride_indices_b: WINDOW as i32, stride_indices_s_q: WINDOW as i32,
            stride_lse_b: HEADS as i32, stride_lse_s_q: HEADS as i32,
            stride_o_b: qn as i32, stride_o_s_q: qn as i32, stride_o_h_q: HEAD_DIM as i32,
            stride_extra_kv_block: extra_page_bytes as i32, stride_extra_kv_row: if extra_topk > 0 { 584 } else { 0 },
            stride_extra_indices_b: extra_topk as i32, stride_extra_indices_s_q: extra_topk as i32,
            stream: s.0, num_sms: self.sms() as i32, sched, num_splits, sched_ready: sched_ready as i32, scratch, scratch_bytes,
        };
        flashmla::sparse_decode(&args)?;
        if ratio == 128 {
            // Split-KV scratch layout of kernels/flashmla_decode.cu (1 KiB-aligned): scheduling
            // metadata (148 x 32 B), num_splits, lse_accum [149, 64], o_accum [149, 64, 512].
            self.snap(layer, "decode num_splits", "flash_fwd_splitkv", 0, 0, 35, num_splits, 8)?;
            self.snap(layer, "decode lse_accum", "flash_fwd_splitkv", 0, 0, 29, scratch, 2 * HEADS * 4)?;
            self.snap(layer, "decode o_accum", "flash_fwd_splitkv", 0, 0, 30, scratch + 38912, 2 * HEADS * 512 * 4)?;
            self.snap(layer, "decode combine out", "flash_fwd_mla_combine", 0, 0, 3, o, qn * 2)?;
        } else {
            self.snap(layer, "decode out", "flash_fwd_splitkv", 0, 0, 13, o, qn * 2)?;
        }
        self.attn_out(layer, o, b, a)
    }

    /// The compressor's decode step: this token's kv/score row into the state, and when the token
    /// completes a group, the compressed row into the C4 / C128 cache.
    fn compressor_decode(&mut self, layer: usize, x: u64, b: &Batch, a: &mut Arena) -> Result<()> {
        let (s, k, bs) = (self.s(), self.k, b.t);
        let l = format!("model.layers.{layer}.self_attn");
        let ratio = compress_ratio(layer);
        let (din, dl) = (self.decode_in.as_ref().ok_or("decode without decode inputs")?.ptr, self.dl);
        let width = if ratio == 4 { 4 * HEAD_DIM } else { 2 * HEAD_DIM };
        let kv_score = a.alloc(bs * width * 4)?;
        let info = self.lt.linear(s, bs, width, HIDDEN, x, self.w.get(&format!("{l}.compressor.wkv_gate.weight"))?, kv_score, true)?;
        if self.probe == Some(layer) && self.snap_label.is_some() { println!("{}", serde_json::json!({"cublaslt_algo": info})); }
        self.snap(layer, "kv_score", "splitKreduce", 0, 2, 0, kv_score, bs * width * 4)?;
        if b.verify {
            // SGLang's target verify compresses as an extend of each request's draft tokens: the
            // prefill kernels on the chunk metadata's plans (the uniform-extend plan path).
            let ape = self.w.get(&format!("{l}.compressor.ape"))?;
            let norm = self.w.get(&format!("{l}.compressor.norm.weight"))?;
            let m = if ratio == 4 { self.chunk.c4 } else { self.chunk.c128 };
            let comp = a.alloc(m.num_c.max(1) * HEAD_DIM * 4)?;
            if ratio == 4 {
                compress::c4_prefill(k, s, HEAD_DIM, self.cache.c4_state_layer(layer), kv_score, comp, ape, m.plan_c, m.plan_w, m.num_c, m.num_w)?;
                compress::norm_rope_store_flashmla(k, s, comp, m.plan_c, norm, self.tables.freqs(layer), m.out_loc, self.cache.c4_layer(layer), 4, m.num_c, 6, false)?;
            } else {
                compress::c128_prefill(k, s, self.cache.c128_state_layer(layer), kv_score, comp, ape, m.plan_c, m.plan_w, m.num_c, m.num_w)?;
                if m.num_c > 0 {
                    compress::norm_rope_store_flashmla(k, s, comp, m.plan_c, norm, self.tables.freqs(layer), m.out_loc, self.cache.c128_layer(layer), 128, m.num_c, 1, false)?;
                }
            }
            return Ok(());
        }
        let comp = a.alloc(bs * HEAD_DIM * 4)?;
        let ape = self.w.get(&format!("{l}.compressor.ape"))?;
        let (plan_d, out_loc, cache, page_bits) = if ratio == 4 {
            let plan_d = din + dl.c4_plan as u64;
            compress::c4_decode(k, s, HEAD_DIM, self.cache.c4_state_layer(layer), kv_score, comp, ape, plan_d, bs)?;
            (plan_d, din + dl.c4_loc as u64, self.cache.c4_layer(layer), 6)
        } else {
            let plan_d = din + dl.c128_plan as u64;
            compress::c128_decode(k, s, self.cache.c128_state_layer(layer), kv_score, comp, ape, plan_d, bs)?;
            (plan_d, din + dl.c128_loc as u64, self.cache.c128_layer(layer), 1)
        };
        compress::norm_rope_store_flashmla(k, s, comp, plan_d, self.w.get(&format!("{l}.compressor.norm.weight"))?, self.tables.freqs(layer),
                                           out_loc, cache, ratio, bs, page_bits, true)
    }

    /// The indexer compressor's decode step (this token's row into its state, every 4th token a
    /// compressed key into the index-K cache); with the indexer active, also this token's per-head
    /// weights (weights_proj, bf16 [64]), returned for `c4_indexer_decode`.
    fn indexer_compressor_decode(&mut self, layer: usize, x: u64, b: &Batch, active: bool, a: &mut Arena) -> Result<Option<u64>> {
        let (s, k, bs) = (self.s(), self.k, b.t);
        let li = format!("model.layers.{layer}.self_attn.indexer");
        let (din, dl) = (self.decode_in.as_ref().ok_or("decode without decode inputs")?.ptr, self.dl);
        let kv_score = a.alloc(bs * 4 * INDEX_DIM * 4)?;
        self.lt.linear(s, bs, 4 * INDEX_DIM, HIDDEN, x, self.w.get(&format!("{li}.compressor.wkv_gate.weight"))?, kv_score, true)?;
        if b.verify {
            // As the main compressor: an extend of each request's draft tokens.
            let m = self.chunk.c4;
            let comp = a.alloc(m.num_c.max(1) * INDEX_DIM * 4)?;
            compress::c4_prefill(k, s, INDEX_DIM, self.cache.index_state_layer(layer), kv_score, comp, self.w.get(&format!("{li}.compressor.ape"))?,
                                 m.plan_c, m.plan_w, m.num_c, m.num_w)?;
            compress::norm_rope_store_indexer(k, s, comp, m.plan_c, self.w.get(&format!("{li}.compressor.norm.weight"))?, self.tables.freqs(layer),
                                              m.out_loc, self.cache.index_k_layer(layer), 4, m.num_c, 6, false)?;
        } else {
            let comp = a.alloc(bs * INDEX_DIM * 4)?;
            let plan_d = din + dl.c4_plan as u64;
            compress::c4_decode(k, s, INDEX_DIM, self.cache.index_state_layer(layer), kv_score, comp, self.w.get(&format!("{li}.compressor.ape"))?, plan_d, bs)?;
            compress::norm_rope_store_indexer(k, s, comp, plan_d, self.w.get(&format!("{li}.compressor.norm.weight"))?, self.tables.freqs(layer),
                                              din + dl.c4_loc as u64, self.cache.index_k_layer(layer), 4, bs, 6, true)?;
        }
        if !active { return Ok(None); }
        let w = a.alloc(bs * INDEX_HEADS * 2)?;
        self.lt.linear(s, bs, INDEX_HEADS, HIDDEN, x, self.w.get(&format!("{li}.weights_proj.weight"))?, w, false)?;
        Ok(Some(w))
    }

    /// The C4 sparse indexer of a decode step (paged): this token's query and weights, its MQA
    /// logits over every compressed key of the request read from the index-K cache pages, and the
    /// top-512 as cache slots, sorted (FlashMLA's extra indices). Everything position-dependent comes
    /// from the decode inputs, so the step stays a replayable graph.
    fn c4_indexer_decode(&mut self, layer: usize, (lq, ls): (u64, u64), w: u64, b: &Batch, a: &mut Arena) -> Result<u64> {
        let (s, k, sms, bs) = (self.s(), self.k, self.sms(), b.t);
        let l = format!("model.layers.{layer}.self_attn.indexer");
        let (din, dl) = (self.decode_in.as_ref().ok_or("decode without decode inputs")?.ptr, self.dl);
        let (heads, dim) = (INDEX_HEADS, INDEX_DIM);
        let q = a.alloc(bs * heads * dim * 2)?;
        deepgemm::fp8_gemm_nt(k, s, lq, ls, self.w.get(&format!("{l}.wq_b.weight"))?, self.w.get(&format!("{l}.wq_b.weight_scale_inv"))?,
                              q, bs, heads * dim, Q_LORA, sms)?;
        let q_fp8 = a.alloc(bs * heads * dim)?;
        let weights = a.alloc(bs * heads * 4)?;
        sgl::fused_q_indexer_rope_hadamard_quant(k, s, q, q_fp8, w, weights, f32::from_bits(0x3c3504f3), self.tables.freqs(layer), b.pos32,
                                                 heads, bs, heads)?;
        // Each request's row scores its own keys through its page-table row.
        let width = dl.w;
        let max_context = width * C4_PAGE_TOKENS;
        let stride = deepgemm::mqa_logits_stride(max_context);
        let logits = a.alloc(bs * stride * 4)?;
        let (ctx, table) = (din + dl.c4_ctx as u64, din + dl.table as u64);
        deepgemm::paged_mqa_logits(k, s, q_fp8, self.cache.index_k_layer(layer), self.cache.c4_pages, weights, ctx, table, width,
                                   din + dl.sched as u64, logits, bs, max_context, heads, sms)?;
        let pages = a.alloc(bs * INDEX_TOPK * 4)?;
        crate::topk::page_table_transform(k, s, logits, pages, 0, table, width, ctx, bs, INDEX_TOPK, max_context, stride, 6)?;
        Ok(pages)
    }

    /// The head: the last FFN's mHC post, hc_head, final RMSNorm and lm_head (cuBLASLt, bf16 logits
    /// [t, vocab]). Returns (hc_head output, normed hidden, logits).
    pub fn head(&mut self, hc: &Hc, layer_out: u64, t: usize, a: &mut Arena) -> Result<(u64, u64, u64)> {
        let (s, k) = (self.s(), self.k);
        let residual = a.alloc(t * HC * HIDDEN * 2)?;
        mhc::post(k, s, hc.comb_mix, hc.residual, hc.post_mix, layer_out, residual, t)?;
        if let Some(dst) = self.hc_dest { crate::cuda::copy_async(dst, residual, t * HC * HIDDEN * 2, s)?; }
        let y = a.alloc(t * HIDDEN * 2)?;
        mhc::head(k, s, residual, self.w.get("model.hc_head_fn")?, self.w.get("model.hc_head_scale")?, self.w.get("model.hc_head_base")?, y, t)?;
        let normed = a.alloc(t * HIDDEN * 2)?;
        sgl::rmsnorm_4096(k, s, y, self.w.get("model.norm.weight")?, normed, t, EPS)?;
        // Without input logprobs SGLang's logits processor keeps only each request's last position
        // before the LM head (pruned states); with them, every position. Several requests: their
        // rows gathered, one GEMM over them (returns [rows, vocab] itself).
        if self.last_logits_only && self.prune_rows.len() > 1 {
            let rows = std::mem::take(&mut self.prune_rows);
            let pruned = a.alloc(rows.len() * HIDDEN * 2)?;
            for (i, &r) in rows.iter().enumerate() {
                crate::cuda::copy_async(pruned + (i * HIDDEN * 2) as u64, normed + (r * HIDDEN * 2) as u64, HIDDEN * 2, s)?;
            }
            let logits = a.alloc(rows.len() * VOCAB_LOCAL * 2)?;
            self.lt.linear(s, rows.len(), VOCAB_LOCAL, HIDDEN, pruned, self.w.get("lm_head.weight")?, logits, false)?;
            self.prune_rows = rows;
            return Ok((y, normed, logits));
        }
        if self.last_logits_only && t > 1 {
            let row = normed + ((t - 1) * HIDDEN * 2) as u64;
            let last = a.alloc(VOCAB_LOCAL * 2)?;
            self.lt.linear(s, 1, VOCAB_LOCAL, HIDDEN, row, self.w.get("lm_head.weight")?, last, false)?;
            // Only row t - 1 exists: return the base address that puts row t - 1 at `last` (callers
            // index rows from the base).
            return Ok((y, normed, last.wrapping_sub(((t - 1) * VOCAB_LOCAL * 2) as u64)));
        }
        let logits = a.alloc(t * VOCAB_LOCAL * 2)?;
        self.lt.linear(s, t, VOCAB_LOCAL, HIDDEN, normed, self.w.get("lm_head.weight")?, logits, false)?;
        Ok((y, normed, logits))
    }

    /// One full forward (prefill chunk or decode step): every layer, then the head. Returns the bf16
    /// logits [t, vocab]; `layer_outs` (if given) receives each layer's output (persist-allocated).
    pub fn forward(&mut self, b: &Batch, persist: &mut Arena, scratch: &mut Arena, mut layer_outs: Option<&mut Vec<u64>>) -> Result<u64> {
        let t = b.t;
        // One split-KV schedule per forward (148 parts x 32 B + num_splits), built by its first C128 layer.
        self.c128_sched = Some((persist.alloc(148 * 32)?, persist.alloc(4 * (t + 1))?, false)); // num_splits: int32 [b + 1]
        // Diagnostics (DSV4_STAGE_TIMES=1): synchronize after each stage and print its wall time.
        let timed = std::env::var_os("DSV4_STAGE_TIMES").is_some();
        let mut stages: Vec<(String, f64)> = Vec::new();
        let mut mark = std::time::Instant::now();
        let mut stage = |name: String, s: Stream| -> Result<()> {
            if timed { s.synchronize()?; stages.push((name, mark.elapsed().as_secs_f64() * 1e3)); mark = std::time::Instant::now(); }
            Ok(())
        };
        if b.verify {
            // Prepared (verify_meta) before the forward, outside any graph capture.
            if !std::mem::take(&mut self.verify_ready) { return Err("verify forward without verify_meta".into()); }
        } else if !b.decode { self.chunk = self.chunk_meta(b, persist)?; }
        let (mut hc, mut x) = self.embed(b, persist, scratch)?;
        stage("embed".into(), self.s())?;
        let mut layer_out = 0u64;
        for layer in 0..LAYERS {
            scratch.reset();
            if layer > 0 { x = self.post_pre(layer, "attn", &mut hc, layer_out, t, persist, scratch)?; }
            let y = self.attention(layer, x, b, scratch)?;
            stage(format!("l{layer}.attn"), self.s())?;
            let xf = self.post_pre(layer, "ffn", &mut hc, y, t, persist, scratch)?;
            if self.probe == Some(layer) {
                self.probes.clear();
                self.probes.push((format!("model.layers.{layer}.self_attn.in.x"), x, t * HIDDEN * 2));
                self.probes.push((format!("model.layers.{layer}.self_attn.out"), y, t * HIDDEN * 2));
                self.probes.push((format!("model.layers.{layer}.mlp.in.0"), xf, t * HIDDEN * 2));
            }
            // Debug captures of every layer keep their own copies; otherwise the two alternating buffers.
            layer_out = if layer_outs.is_some() { persist.alloc(t * HIDDEN * 2)? } else { hc.layer_out() };
            self.ffn_into(layer, xf, b, scratch, layer_out)?;
            if self.probe == Some(layer) {
                // Probed scratch tensors outlive the layer: move them to the persist arena.
                let stream = self.s();
                for p in self.probes.iter_mut() {
                    let keep = persist.alloc(p.2)?;
                    crate::cuda::copy_async(keep, p.1, p.2, stream)?;
                    p.1 = keep;
                }
            }
            if let Some(v) = layer_outs.as_deref_mut() { v.push(layer_out); }
            stage(format!("l{layer}.ffn"), self.s())?;
        }
        scratch.reset();
        // Last-position pruning concerns a prefill chunk (one request); every decode row is a request.
        let keep = self.last_logits_only;
        self.last_logits_only &= !b.decode;
        // A packed batch: each request's last row.
        self.prune_rows = if b.segs.len() > 1 { b.segs.iter().scan(0, |acc, g| { *acc += g.len; Some(*acc - 1) }).collect() } else { vec![] };
        let head = self.head(&hc, layer_out, t, scratch);
        let last_only = self.last_logits_only;
        self.last_logits_only = keep;
        let pruned = std::mem::take(&mut self.prune_rows).len();
        let (_, _, logits) = head?;
        if last_only && pruned > 1 {
            // [requests, vocab], row r the r-th segment's last token.
            let out = persist.alloc(pruned * VOCAB * 2)?;
            crate::cuda::copy_async(out, logits, pruned * VOCAB * 2, self.s())?;
            return Ok(out);
        }
        // The logits rows that exist (all, or the last one), moved out of the scratch arena; the
        // returned base indexes rows like a full [t, vocab] tensor.
        let first = if last_only { t - 1 } else { 0 };
        let out = persist.alloc((t - first) * VOCAB * 2)?;
        crate::cuda::copy_async(out, logits.wrapping_add((first * VOCAB * 2) as u64), (t - first) * VOCAB * 2, self.s())?;
        let out = out.wrapping_sub((first * VOCAB * 2) as u64);
        stage("head".into(), self.s())?;
        if timed {
            let slow: Vec<_> = stages.iter().filter(|(_, ms)| *ms > 5.0).map(|(n, ms)| serde_json::json!([n, (ms * 10.0).round() / 10.0])).collect();
            println!("{}", serde_json::json!({"forward_t": t, "total_ms": stages.iter().map(|s| s.1).sum::<f64>(), "stages_over_5ms": slow}));
        }
        Ok(out)
    }

    /// The logprobs the reference returns for one forward (SGLang with return_logprob,
    /// top_logprobs_num 5, logprob_start_len 0): `logits` bf16 [t, vocab] from [`Model::forward`],
    /// `token` the sampled id, `input_ids` the prefill's tokens (None in decode). The device runs
    /// the reference's kernels (`logits.float()`, the fused logsumexp / top-k, log_softmax); the
    /// host finishes with the same fp32 arithmetic and ATen's top-k order (crate::logprobs).
    pub fn logprobs(&self, logits: u64, t: usize, input_ids: Option<&[i64]>, token: usize, a: &mut Arena) -> Result<crate::logprobs::Logprobs> {
        let (s, k) = (self.s(), self.k);
        let f32s = |b: Vec<u8>| -> Vec<f32> { b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect() };
        let wide = a.alloc(t * VOCAB * 4)?;
        torch_ops::copy_bf16_to_f32(k, s, wide, logits, t * VOCAB)?;
        let stats = match input_ids {
            Some(_) => {
                let (mx, ls, tv, ti) = (a.alloc(t * 4)?, a.alloc(t * 4)?, a.alloc(t * 5 * 4)?, a.alloc(t * 5 * 8)?);
                sgl::row_logsumexp_topk(k, s, wide, mx, ls, tv, ti, t, VOCAB)?;
                Some((mx, ls))
            }
            None => None,
        };
        let lsm = a.alloc(VOCAB * 4)?;
        torch_ops::log_softmax_f32(k, s, lsm, wide + ((t - 1) * VOCAB * 4) as u64, 1, VOCAB)?;
        s.synchronize()?;
        let mut input = Vec::new();
        if let (Some(ids), Some((mx, ls))) = (input_ids, stats) {
            let (mx, ls) = (f32s(crate::cuda::download(mx, t * 4)?), f32s(crate::cuda::download(ls, t * 4)?));
            let rows = f32s(crate::cuda::download(wide, t * VOCAB * 4)?);
            for i in 0..t - 1 {
                input.push(crate::logprobs::input_logprob(rows[i * VOCAB + ids[i + 1] as usize], mx[i], ls[i]));
            }
        }
        let (tok, top) = crate::logprobs::output(&f32s(crate::cuda::download(lsm, VOCAB * 4)?), token, 5);
        Ok(crate::logprobs::Logprobs { token: tok, top, input })
    }

    /// Attention of `layer` by its kind.
    pub fn attention(&mut self, layer: usize, x: u64, b: &Batch, a: &mut Arena) -> Result<u64> {
        if b.decode { return self.attention_decode(layer, x, b, a); }
        match compress_ratio(layer) {
            0 => self.attention_swa(layer, x, b, a),
            4 => self.attention_c4(layer, x, b, a),
            128 => self.attention_c128(layer, x, b, a),
            r => Err(format!("layer {layer}: compress ratio {r} attention not ported yet")),
        }
    }
}
