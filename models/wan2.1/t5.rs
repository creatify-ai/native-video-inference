//! umt5-xxl encoder (wan/modules/t5.py T5EncoderModel) in bf16 eager semantics: per-layer
//! relative position bias, T5LayerNorm (fp32 mean of squares, rsqrt), unscaled attention via
//! einsum -> bmm with an fp32 softmax, and the gated tanh-GELU feed-forward. The reference runs
//! [prompt, negative prompt] padded to text_len 512 and keeps each context up to its length.
use crate::cuda::{check, Buffer, Device, Stream};
use crate::weights::Source;
use crate::linear::{Dtype, Linear};
use crate::encoders::{mean_f32, softmax_f32, Blas};
use crate::Result;
use std::ffi::{c_int, c_void};

extern "C" {
    fn wan_t5_embed(ids: *const i64, table: *const c_void, x: *mut c_void, tokens: i64, dim: i64, stream: Stream) -> c_int;
    fn wan_t5_position_bias(table: *const c_void, bias: *mut c_void, len: i64, heads: i64, stream: Stream) -> c_int;
    fn wan_t5_square(x: *const c_void, sq: *mut c_void, n: i64, stream: Stream) -> c_int;
    fn wan_t5_norm_apply(x: *const c_void, mean: *const c_void, weight: *const c_void, out: *mut c_void, dim: i64, eps: f64,
                         n: i64, stream: Stream) -> c_int;
    fn wan_t5_heads(x: *const c_void, out: *mut c_void, b: i64, len: i64, heads: i64, c: i64, transpose: c_int,
                    stream: Stream) -> c_int;
    fn wan_t5_merge(x: *const c_void, out: *mut c_void, b: i64, len: i64, heads: i64, c: i64, stream: Stream) -> c_int;
    fn wan_t5_scores(scores: *const c_void, pos: *const c_void, mask: *const i64, out: *mut c_void, b: i64, heads: i64,
                     len: i64, stream: Stream) -> c_int;
    fn wan_f32_to_bf16_enc(x: *const c_void, y: *mut c_void, n: i64, stream: Stream) -> c_int;
    fn wan_add_bf16_enc(a: *const c_void, b: *const c_void, out: *mut c_void, n: i64, stream: Stream) -> c_int;
    fn wan_t5_gated_gelu(fc1: *const c_void, gate: *const c_void, out: *mut c_void, n: i64, stream: Stream) -> c_int;
}
fn p(a: u64) -> *mut c_void { a as *mut c_void }

pub const TEXT_LEN: usize = 512;
pub const DIM: usize = 4096;
const HEADS: usize = 64;
const HEAD_DIM: usize = 64;
const FFN: usize = 10240;
const LAYERS: usize = 24;
const EPS: f64 = 1e-6;

struct Layer { norm1: Buffer, q: Buffer, k: Buffer, v: Buffer, o: Buffer, norm2: Buffer, gate: Buffer, fc1: Buffer,
               fc2: Buffer, pos: Buffer }

pub struct T5<'d> {
    device: &'d Device,
    linear: Linear,
    blas: Blas,
    embedding: Buffer,
    layers: Vec<Layer>,
    norm: Buffer,
}

impl<'d> T5<'d> {
    /// `weights`: the t5 component (bf16), as the reference holds it in memory.
    pub fn load(weights: &dyn Source, device: &'d Device) -> Result<Self> {
        let mut names = vec!["token_embedding.weight".to_string(), "norm.weight".to_string()];
        for i in 0..LAYERS {
            for part in ["norm1", "attn.q", "attn.k", "attn.v", "attn.o", "norm2", "ffn.gate.0", "ffn.fc1", "ffn.fc2",
                         "pos_embedding.embedding"] {
                names.push(format!("blocks.{i}.{part}.weight"));
            }
        }
        let mut buffers = weights.buffers(&names, device)?.into_iter();
        let mut next = || -> Result<Buffer> { buffers.next().ok_or_else(|| "missing T5 tensor".into()) };
        let embedding = next()?;
        let norm = next()?;
        let layers = (0..LAYERS).map(|_| Ok(Layer { norm1: next()?, q: next()?, k: next()?, v: next()?, o: next()?,
                                                    norm2: next()?, gate: next()?, fc1: next()?, fc2: next()?, pos: next()? }))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { device, linear: Linear::new(device)?, blas: Blas::new(device)?, embedding, layers, norm })
    }

    /// T5LayerNorm over rows of DIM: weight * (x * rsqrt(mean(x.float()^2) + eps)).type_as(weight).
    fn norm(&self, x: u64, weight: u64, sq: u64, mean: u64, out: u64, batch: usize) -> Result<()> {
        let (s, n) = (self.device.stream, (batch * TEXT_LEN * DIM) as i64);
        check(unsafe { wan_t5_square(p(x), p(sq), n, s) }, "t5 square")?;
        let (b, l, d) = (batch as i64, TEXT_LEN as i64, DIM as i64);
        mean_f32(sq, mean, &[b, l, d], &[l * d, d, 1], 0b100, s)?;
        check(unsafe { wan_t5_norm_apply(p(x), p(mean), p(weight), p(out), d, EPS, n, s) }, "t5 norm")
    }

    /// ids, mask: [batch, 512] int64 (tokenizer output, padded); out: [batch, 512, 4096] bf16.
    pub fn forward(&self, ids: u64, mask: u64, batch: usize, out: u64) -> Result<()> {
        let s = self.device.stream;
        let rows = batch * TEXT_LEN;
        let (b, l, h, c) = (batch as i64, TEXT_LEN as i64, HEADS as i64, HEAD_DIM as i64);
        let x = Buffer::new(rows * DIM * 2)?;
        check(unsafe { wan_t5_embed(ids as *const i64, p(self.embedding.dev()), p(x.dev()), rows as i64, DIM as i64, s) }, "embed")?;
        let normed = Buffer::new(rows * DIM * 2)?;
        let sq = Buffer::new(rows * DIM * 4)?;
        let mean = Buffer::new(rows * 4)?;
        let (q, k, v, t) = (Buffer::new(rows * DIM * 2)?, Buffer::new(rows * DIM * 2)?, Buffer::new(rows * DIM * 2)?,
                            Buffer::new(rows * DIM * 2)?);
        let pos = Buffer::new(HEADS * TEXT_LEN * TEXT_LEN * 2)?;
        let scores = Buffer::new(batch * HEADS * TEXT_LEN * TEXT_LEN * 2)?;
        let probs = Buffer::new(batch * HEADS * TEXT_LEN * TEXT_LEN * 4)?;
        let (ff1, gate) = (Buffer::new(rows * FFN * 2)?, Buffer::new(rows * FFN * 2)?);
        let lin = |x: u64, w: &Buffer, y: u64, n: usize, k: usize| self.linear.forward_at(x, w.dev(), None, y, rows, n, k, Dtype::Bf16);
        let bh = batch * HEADS;
        let attn_elems = (bh * TEXT_LEN * TEXT_LEN) as i64;
        for layer in &self.layers {
            check(unsafe { wan_t5_position_bias(p(layer.pos.dev()), p(pos.dev()), l, h, s) }, "position bias")?;
            // x = x + attn(norm1(x), mask, pos_bias)
            self.norm(x.dev(), layer.norm1.dev(), sq.dev(), mean.dev(), normed.dev(), batch)?;
            lin(normed.dev(), &layer.q, t.dev(), DIM, DIM)?;
            check(unsafe { wan_t5_heads(p(t.dev()), p(q.dev()), b, l, h, c, 0, s) }, "q heads")?;
            lin(normed.dev(), &layer.k, t.dev(), DIM, DIM)?;
            check(unsafe { wan_t5_heads(p(t.dev()), p(k.dev()), b, l, h, c, 1, s) }, "k heads")?;
            lin(normed.dev(), &layer.v, t.dev(), DIM, DIM)?;
            check(unsafe { wan_t5_heads(p(t.dev()), p(v.dev()), b, l, h, c, 0, s) }, "v heads")?;
            // einsum("binc,bjnc->bnij"): bmm([B*N, L, C], [B*N, C, L]).
            self.blas.bmm(true, q.dev(), [l * c, c, 1], k.dev(), [c * l, l, 1], scores.dev(), bh, TEXT_LEN, HEAD_DIM, TEXT_LEN)?;
            check(unsafe { wan_t5_scores(p(scores.dev()), p(pos.dev()), mask as *const i64, p(probs.dev()), b, h, l, s) }, "scores")?;
            softmax_f32(probs.dev(), probs.dev(), bh * TEXT_LEN, TEXT_LEN, s)?;
            check(unsafe { wan_f32_to_bf16_enc(p(probs.dev()), p(scores.dev()), attn_elems, s) }, "probs bf16")?;
            // einsum("bnij,bjnc->binc"): bmm([B*N, L, L], [B*N, L, C]) -> [B, L, N*C].
            self.blas.bmm(true, scores.dev(), [l * l, l, 1], v.dev(), [l * c, c, 1], q.dev(), bh, TEXT_LEN, TEXT_LEN, HEAD_DIM)?;
            check(unsafe { wan_t5_merge(p(q.dev()), p(t.dev()), b, l, h, c, s) }, "merge heads")?;
            lin(t.dev(), &layer.o, normed.dev(), DIM, DIM)?;
            check(unsafe { wan_add_bf16_enc(p(x.dev()), p(normed.dev()), p(x.dev()), (rows * DIM) as i64, s) }, "residual")?;
            // x = x + ffn(norm2(x))
            self.norm(x.dev(), layer.norm2.dev(), sq.dev(), mean.dev(), normed.dev(), batch)?;
            lin(normed.dev(), &layer.fc1, ff1.dev(), FFN, DIM)?;
            lin(normed.dev(), &layer.gate, gate.dev(), FFN, DIM)?;
            check(unsafe { wan_t5_gated_gelu(p(ff1.dev()), p(gate.dev()), p(gate.dev()), (rows * FFN) as i64, s) }, "gated gelu")?;
            lin(gate.dev(), &layer.fc2, normed.dev(), DIM, FFN)?;
            check(unsafe { wan_add_bf16_enc(p(x.dev()), p(normed.dev()), p(x.dev()), (rows * DIM) as i64, s) }, "residual")?;
        }
        self.norm(x.dev(), self.norm.dev(), sq.dev(), mean.dev(), out, batch)?;
        self.device.sync()
    }
}
