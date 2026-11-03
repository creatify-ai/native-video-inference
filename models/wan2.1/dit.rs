//! The official Wan 2.1 I2V-14B DiT (wan/modules/model.py, WanModel with model_type 'i2v') with its
//! parameters converted to bf16 and run under bf16 autocast, as the reference configuration runs it:
//! bf16 GEMMs, fp32 residual stream, fp32 time embedding / head GEMMs, eager LayerNorm and RMSNorm,
//! float64 RoPE, cuDNN SDPA (the reference's FlashAttention swap). Every step rounds where the reference
//! rounds; see kernels/dit.cu.
//!
//! Context k/v of the cross-attention depend only on the prompt and image, so they are computed
//! once per CFG pass and reused for every step (the reference recomputes the same values).
use crate::conv::{contiguous, Conv, ConvSpec};
use crate::cuda::{check, upload_into, Buffer, Device, Stream};
use crate::eager;
use crate::linear::{Dtype, Linear};
use crate::sdpa::{self, Io, View};
use crate::weights::{SafeTensors, Source};
use crate::Result;
use std::ffi::{c_int, c_void};

pub const DIM: usize = 5120;
pub const FFN: usize = 13824;
pub const HEADS: usize = 40;
pub const HEAD_DIM: usize = 128;
pub const LAYERS: usize = 40;
pub const IN_CHANNELS: usize = 36;
pub const OUT_CHANNELS: usize = 16;
pub const FREQ_DIM: usize = 256;
pub const TEXT_DIM: usize = 4096;
pub const TEXT_LEN: usize = 512;
pub const CLIP_TOKENS: usize = 257;
pub const CLIP_DIM: usize = 1280;
const EPS: f32 = 1e-6;
const LN_EPS_MLP: f32 = 1e-5;  // torch.nn.LayerNorm default (MLPProj)

extern "C" {
    fn w21_cat_cast_bf16(a: *const c_void, y: *const c_void, out: *mut c_void, a_n: i64, y_n: i64, s: Stream) -> c_int;
    fn w21_sinusoidal(t: *const c_void, out: *mut c_void, dim: c_int, s: Stream) -> c_int;
    fn w21_layer_norm(mode: c_int, in_dtype: c_int, x: *const c_void, gamma: *const c_void, beta: *const c_void,
                      e0: *const c_void, e1: *const c_void, out: *mut c_void, rows: i64, cols: c_int, eps: f32,
                      round_out: c_int, s: Stream) -> c_int;
    fn w21_rms_apply(x: *const c_void, mean: *const c_void, w: *const c_void, out: *mut c_void, rows: i64, cols: i64,
                     eps: f32, s: Stream) -> c_int;
    fn w21_rope(x: *const c_void, freqs: *const c_void, out: *mut c_void, out_f32: *mut c_void, heads: c_int, f: c_int,
                h: c_int, w: c_int, s: Stream) -> c_int;
    fn w21_rms_rope(x: *const c_void, mean: *const c_void, weight: *const c_void, freqs: *const c_void, out: *mut c_void,
                    heads: c_int, f: c_int, h: c_int, w: c_int, eps: f32, s: Stream) -> c_int;
    fn w21_residual_gate(x: *const c_void, x_bf16: c_int, y: *const c_void, e: *const c_void, out: *mut c_void, rows: i64,
                         cols: i64, s: Stream) -> c_int;
    fn w21_residual_add(x: *const c_void, y: *const c_void, out: *mut c_void, n: i64, s: Stream) -> c_int;
    fn w21_residual_layer_norm(mode: c_int, gate: c_int, x: *mut c_void, y: *const c_void, eg: *const c_void,
                               gamma: *const c_void, beta: *const c_void, e0: *const c_void, e1: *const c_void,
                               out: *mut c_void, rows: i64, cols: c_int, eps: f32, s: Stream) -> c_int;
    fn w21_add_bf16_f32(a: *const c_void, e: *const c_void, out: *mut c_void, n: i64, e_period: i64, s: Stream) -> c_int;
    fn w21_cast(kind: c_int, x: *const c_void, out: *mut c_void, n: i64, s: Stream) -> c_int;
    fn w21_unpatchify(u: *const c_void, out: *mut c_void, f: c_int, h: c_int, w: c_int, c: c_int, s: Stream) -> c_int;
    fn wan_square_mean_bf16(x: *const c_void, out: *mut c_void, rows: i64, cols: i64, stream: Stream) -> c_int;
    fn wan_square_mean_warp_rows(x: *const c_void, rows: i64, cols: i64, factor: *mut f32) -> c_int;
    fn w21_rms_rows(x: *const c_void, weight: *const c_void, freqs: *const c_void, out: *mut c_void, rows: i64, f: c_int, h: c_int,
                    w: c_int, eps: f32, factor: f32, s: Stream) -> c_int;
    fn wan_gelu_erf(x: *const c_void, y: *mut c_void, rows: i64, cols: i64, in_stride: i64, dtype: c_int, stream: Stream) -> c_int;
}
fn p(a: u64) -> *mut c_void { a as *mut c_void }
/// The factor of torch's mean of squares over bf16 [rows, DIM] at x when that reduction is
/// square_mean_rows, whose lanes and order kernels/dit.cu's rms_rows reproduce in registers.
fn warp_rows(x: u64, rows: usize) -> Option<f32> {
    let mut factor = 0.0f32;
    (unsafe { wan_square_mean_warp_rows(p(x), rows as i64, DIM as i64, &mut factor) } == 1).then_some(factor)
}

/// LayerNorm epilogues of kernels/dit.cu.
#[derive(Clone, Copy)]
enum LnMode { Modulate = 0, AffineBf16 = 1, AffineF32 = 2, ModulateF32 = 3 }
#[derive(Clone, Copy)]
enum In { F32 = 0, Bf16 = 1, F16 = 2 }

/// Latent grid of one request: [16, frames, height, width]; tokens are 1x2x2 patches.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Grid { pub frames: usize, pub height: usize, pub width: usize }
impl Grid {
    pub fn tokens(&self) -> usize { self.frames * (self.height / 2) * (self.width / 2) }
    pub fn latent_elements(&self, channels: usize) -> usize { channels * self.frames * self.height * self.width }
}

fn block_param(i: usize, name: &str) -> String { format!("blocks.{i}.{name}") }

/// BHSD views of BSHD memory, with the strides torch sees in the reference's SDPA call
/// (`x.transpose(0, 1).unsqueeze(0)` of a contiguous [L, heads, 128] tensor).
fn bshd(rows: usize) -> View {
    let (h, d, row) = (HEADS as i64, HEAD_DIM as i64, DIM as i64);
    View { dims: [1, h, rows as i64, d], strides: [h * d, d, row, 1], offset_bytes: 0 }
}

/// Plans for the grid's three attention shapes (self, image, text): the largest workspace.
fn attention_plans(grid: Grid) -> Result<usize> {
    let s = grid.tokens();
    Ok([(s, s), (s, CLIP_TOKENS), (s, TEXT_LEN)].iter()
        .map(|&(q, k)| sdpa::prepare(Io::Bf16, &bshd(q), &bshd(k), &bshd(k)).map(|p| p.workspace))
        .collect::<Result<Vec<_>>>()?.into_iter().max().unwrap_or(0))
}

/// The largest cuDNN SDPA workspace of the grid's three attention shapes.
fn attention_workspace(grid: Grid) -> Result<Option<Buffer>> {
    let ws = attention_plans(grid)?;
    Ok(if ws > 0 { Some(Buffer::new(ws)?) } else { None })
}

/// One CFG pass's context: the text embedding and the image embedding, and every block's
/// cross-attention keys/values for it.
pub struct PassContext {
    pub text: Buffer,      // bf16 [512, 5120] text_embedding output
    pub image_f32: Buffer, // fp32 [257, 5120] img_emb output (the reference's context_clip)
    k: Vec<Buffer>, v: Vec<Buffer>, k_img: Vec<Buffer>, v_img: Vec<Buffer>,
}

pub struct Dit<'d> {
    pub device: &'d Device,
    pub grid: Grid,
    linear: Linear,
    conv: Conv,
    bf16: std::collections::HashMap<String, Buffer>,
    f32: std::collections::HashMap<String, Buffer>,
    freqs: Buffer,
    ws: Option<Buffer>,
    sc: Scratch,
    /// Keep fp32 RoPE outputs of block 0 for the fixture comparison (`observe` names "b0/...").
    pub debug: Option<[Buffer; 2]>,
    /// Seconds into `load_from` at which each setup stage finished (startup diagnostics).
    pub setup: Vec<(&'static str, f64)>,
}

/// Activation buffers of one forward (sized for the grid, reused across blocks and steps).
struct Scratch {
    x_in: Buffer, conv: Buffer, tokens: Buffer, x: Buffer, xm: Buffer, q: Buffer, k: Buffer, v: Buffer, sq: Buffer,
    mean: Buffer, qn: Buffer, kn: Buffer, qr: Buffer, kr: Buffer, attn: Buffer, y: Buffer, img: Buffer, txt: Buffer,
    sum: Buffer, h: Buffer, g: Buffer, e_blk: Buffer, e_next: Buffer, sinus: Buffer, t1: Buffer, t1s: Buffer, e: Buffer, es: Buffer,
    e0: Buffer, eh: Buffer, head: Buffer, t_dev: Buffer,
}
impl Scratch {
    fn new(grid: Grid) -> Result<Self> {
        let s = grid.tokens();
        let act = |w: usize| Buffer::new(s * DIM * w);
        Ok(Self {
            x_in: Buffer::new(grid.latent_elements(IN_CHANNELS) * 2)?, conv: act(2)?, tokens: act(2)?, x: act(4)?,
            xm: act(2)?, q: act(2)?, k: act(2)?, v: act(2)?, sq: Buffer::new(s.max(TEXT_LEN) * DIM * 4)?,
            mean: Buffer::new(s.max(TEXT_LEN) * 4)?, qn: act(2)?, kn: act(2)?, qr: act(2)?, kr: act(2)?, attn: act(2)?,
            y: act(2)?, img: act(2)?, txt: act(2)?, sum: act(2)?, h: Buffer::new(s * FFN * 2)?, g: Buffer::new(s * FFN * 2)?,
            e_blk: Buffer::new(6 * DIM * 4)?, e_next: Buffer::new(6 * DIM * 4)?, sinus: Buffer::new(FREQ_DIM * 4)?, t1: Buffer::new(DIM * 4)?,
            t1s: Buffer::new(DIM * 4)?, e: Buffer::new(DIM * 4)?, es: Buffer::new(DIM * 4)?, e0: Buffer::new(6 * DIM * 4)?,
            eh: Buffer::new(2 * DIM * 4)?, head: Buffer::new(s * 64 * 4)?, t_dev: Buffer::new(8)?,
        })
    }
}

/// Parameters the reference uses in fp32 (autocast fp32 regions and LayerNorm weights): widened
/// from their bf16 values, which is exactly the autocast cast.
const F32_PARAMS: [&str; 10] = ["time_embedding.0.weight", "time_embedding.0.bias", "time_embedding.2.weight",
    "time_embedding.2.bias", "time_projection.1.weight", "time_projection.1.bias", "head.head.weight", "head.head.bias",
    "img_emb.proj.0.weight", "img_emb.proj.0.bias"];
const F32_PARAMS_2: [&str; 2] = ["img_emb.proj.4.weight", "img_emb.proj.4.bias"];

impl<'d> Dit<'d> {
    /// Builds the grid's attention plans in the process-wide cache; may run on another thread
    /// while weights load.
    pub fn prewarm(grid: Grid) -> Result<()> { attention_plans(grid).map(|_| ()) }
    /// `path`: a converted dit.safetensors (bf16 parameters + float64 `freqs`; parity tooling).
    pub fn load(path: &std::path::Path, grid: Grid, device: &'d Device) -> Result<Self> {
        Self::load_from(&SafeTensors::open(path)?, grid, device)
    }
    /// The DiT from any tensor source holding its bf16 parameters and the float64 `freqs` table.
    pub fn load_from(src: &dyn Source, grid: Grid, device: &'d Device) -> Result<Self> {
        let t = std::time::Instant::now();
        let mut bf16 = std::collections::HashMap::new();
        let mut f32 = std::collections::HashMap::new();
        let names: Vec<String> = src.names()?.into_iter().filter(|n| n.as_str() != "freqs").collect();
        for name in &names {
            if src.dtype(name)? != "BF16" { return Err(format!("{name}: expected BF16").into()); }
            let buffer = src.buffer(name, device)?;
            let wide = F32_PARAMS.contains(&name.as_str()) || F32_PARAMS_2.contains(&name.as_str())
                || name.ends_with("norm3.weight") || name.ends_with("norm3.bias");
            if wide {
                // The autocast fp32 regions read the bf16 parameter widened (exact).
                let out = Buffer::new(buffer.bytes() * 2)?;
                eager::bf16_to_f32(buffer.dev(), out.dev(), buffer.bytes() / 2, device.stream)?;
                f32.insert(name.clone(), out);
            }
            bf16.insert(name.clone(), buffer);
        }
        if src.dtype("freqs")? != "F64" { return Err("freqs: expected F64".into()); }
        let freqs = src.buffer("freqs", device)?;
        device.sync()?;
        let mut setup = vec![("weights", t.elapsed().as_secs_f64())];
        let mut mark = |name: &'static str| -> Result<()> { device.sync()?; setup.push((name, t.elapsed().as_secs_f64())); Ok(()) };
        let linear = Linear::new(device)?;
        mark("cublas")?;
        let conv = Conv::new(16 << 30)?;
        mark("cudnn")?;
        let ws = attention_workspace(grid)?;
        mark("sdpa_plans")?;
        let sc = Scratch::new(grid)?;
        mark("scratch")?;
        Ok(Self { device, grid, linear, conv, bf16, f32, freqs, ws, sc, debug: None, setup })
    }
    /// Re-sizes the activation buffers and the attention workspace for another latent grid.
    pub fn set_grid(&mut self, grid: Grid) -> Result<()> {
        if grid == self.grid { return Ok(()); }
        self.ws = None;
        self.sc = Scratch::new(grid)?;
        self.ws = attention_workspace(grid)?;
        self.grid = grid;
        if self.debug.is_some() { self.enable_debug()?; }
        Ok(())
    }
    /// fp32 RoPE copies of block 0 (q, k) for fixture checks.
    pub fn enable_debug(&mut self) -> Result<()> {
        let n = self.grid.tokens() * DIM * 4;
        self.debug = Some([Buffer::new(n)?, Buffer::new(n)?]);
        Ok(())
    }
    /// (e, e0) of the last `time` call.
    pub fn time_buffers(&self) -> (u64, u64) { (self.sc.e.dev(), self.sc.e0.dev()) }
    /// Block `i`'s cross-attention (k, v, k_img, v_img) of a pass context.
    pub fn context_kv(ctx: &PassContext, i: usize) -> [u64; 4] { [ctx.k[i].dev(), ctx.v[i].dev(), ctx.k_img[i].dev(), ctx.v_img[i].dev()] }
    fn b(&self, name: &str) -> u64 { self.bf16[name].dev() }
    fn f(&self, name: &str) -> u64 { self.f32[name].dev() }
    fn s(&self) -> Stream { self.device.stream }

    fn lin(&self, x: u64, name: &str, y: u64, m: usize, n: usize, k: usize) -> Result<()> {
        self.linear.forward_at(x, self.b(&format!("{name}.weight")), Some(self.b(&format!("{name}.bias"))), y, m, n, k, Dtype::Bf16)
    }
    fn lin_f32(&self, x: u64, name: &str, y: u64, m: usize, n: usize, k: usize) -> Result<()> {
        self.linear.forward_at(x, self.f(&format!("{name}.weight")), Some(self.f(&format!("{name}.bias"))), y, m, n, k, Dtype::F32)
    }
    #[allow(clippy::too_many_arguments)]
    fn ln(&self, mode: LnMode, input: In, x: u64, gamma: u64, beta: u64, e0: u64, e1: u64, out: u64, rows: usize, eps: f32,
          round_out: bool) -> Result<()> {
        check(unsafe { w21_layer_norm(mode as c_int, input as c_int, p(x), p(gamma), p(beta), p(e0), p(e1), p(out), rows as i64,
                                      DIM as c_int, eps, c_int::from(round_out), self.s()) }, "layer norm")
    }
    /// x += y * gate (or x += y) in place, then LayerNorm `mode` of the new x into xm: the bytes of the
    /// residual kernel followed by `ln`, in one pass over the rows.
    #[allow(clippy::too_many_arguments)]
    fn residual_ln(&self, mode: LnMode, gate: Option<u64>, y: u64, gamma: u64, beta: u64, e0: u64, e1: u64, rows: usize)
                   -> Result<()> {
        check(unsafe { w21_residual_layer_norm(mode as c_int, c_int::from(gate.is_some()), p(self.sc.x.dev()), p(y),
                                               p(gate.unwrap_or(0)), p(gamma), p(beta), p(e0), p(e1), p(self.sc.xm.dev()),
                                               rows as i64, DIM as c_int, EPS, self.s()) }, "residual layer norm")
    }
    /// WanRMSNorm over the last dim of x (bf16 [rows, 5120]); the mean of squares is torch's reduction
    /// of the fp32 square, squared at the load.
    fn rms(&self, x: u64, weight: &str, out: u64, rows: usize) -> Result<()> {
        if let Some(factor) = warp_rows(x, rows) {
            return check(unsafe { w21_rms_rows(p(x), p(self.b(weight)), p(0), p(out), rows as i64, 0, 0, 0, EPS, factor, self.s()) },
                         "rms rows");
        }
        check(unsafe { wan_square_mean_bf16(p(x), p(self.sc.mean.dev()), rows as i64, DIM as i64, self.s()) }, "rms mean")?;
        check(unsafe { w21_rms_apply(p(x), p(self.sc.mean.dev()), p(self.b(weight)), p(out), rows as i64, DIM as i64, EPS,
                                     self.s()) }, "rms apply")
    }
    /// WanRMSNorm then rope_apply of a self-attention projection in one pass over the rows, without
    /// the normalized tensor (the same bytes as `rms` followed by `w21_rope`).
    fn rms_rope(&self, x: u64, weight: &str, out: u64, rows: usize, (fr, hh, ww): (usize, usize, usize)) -> Result<()> {
        if let Some(factor) = warp_rows(x, rows) {
            return check(unsafe { w21_rms_rows(p(x), p(self.b(weight)), p(self.freqs.dev()), p(out), rows as i64, fr as c_int,
                                               hh as c_int, ww as c_int, EPS, factor, self.s()) }, "rms rope rows");
        }
        check(unsafe { wan_square_mean_bf16(p(x), p(self.sc.mean.dev()), rows as i64, DIM as i64, self.s()) }, "rms mean")?;
        check(unsafe { w21_rms_rope(p(x), p(self.sc.mean.dev()), p(self.b(weight)), p(self.freqs.dev()), p(out), HEADS as c_int,
                                    fr as c_int, hh as c_int, ww as c_int, EPS, self.s()) }, "rms rope")
    }
    fn attention(&self, q: u64, k: u64, v: u64, out: u64, q_rows: usize, kv_rows: usize) -> Result<()> {
        let scale = sdpa::default_scale(HEAD_DIM as i64);
        let (qv, kv) = (bshd(q_rows), bshd(kv_rows));
        sdpa::forward(Io::Bf16, (q, &qv), (k, &kv), (v, &kv), out, self.ws.as_ref(), scale, self.s())
    }

    /// text_embedding(pad(context)) and img_emb(clip_fea), then every block's cross-attention keys
    /// and values. `context`: T5 output bf16 [rows, 4096]; `clip_fea`: fp16 [257, 1280].
    pub fn pass_context(&self, context: u64, rows: usize, clip_fea: u64) -> Result<PassContext> {
        let st = self.s();
        let padded = Buffer::new(TEXT_LEN * TEXT_DIM * 2)?;
        padded.zero(self.device)?;
        padded.copy_from_device(context, rows * TEXT_DIM * 2, self.device)?;
        let (t0, t1, text) = (Buffer::new(TEXT_LEN * DIM * 2)?, Buffer::new(TEXT_LEN * DIM * 2)?, Buffer::new(TEXT_LEN * DIM * 2)?);
        self.lin(padded.dev(), "text_embedding.0", t0.dev(), TEXT_LEN, DIM, TEXT_DIM)?;
        eager::gelu_tanh_bf16(t0.dev(), t1.dev(), TEXT_LEN * DIM, st)?;
        self.lin(t1.dev(), "text_embedding.2", text.dev(), TEXT_LEN, DIM, DIM)?;
        // MLPProj: LayerNorm (fp32 out) -> Linear -> GELU (erf) -> Linear -> LayerNorm (fp32 out).
        let n = CLIP_TOKENS;
        let (a, g1, g2, b4) = (Buffer::new(n * CLIP_DIM * 2)?, Buffer::new(n * CLIP_DIM * 2)?, Buffer::new(n * CLIP_DIM * 2)?,
                               Buffer::new(n * DIM * 2)?);
        check(unsafe { w21_layer_norm(LnMode::AffineBf16 as c_int, In::F16 as c_int, p(clip_fea), p(self.f("img_emb.proj.0.weight")),
                                      p(self.f("img_emb.proj.0.bias")), p(0), p(0), p(a.dev()), n as i64, CLIP_DIM as c_int,
                                      LN_EPS_MLP, 0, st) }, "img_emb ln0")?;
        self.linear.forward_at(a.dev(), self.b("img_emb.proj.1.weight"), Some(self.b("img_emb.proj.1.bias")), g1.dev(), n, CLIP_DIM,
                               CLIP_DIM, Dtype::Bf16)?;
        check(unsafe { wan_gelu_erf(p(g1.dev()), p(g2.dev()), n as i64, CLIP_DIM as i64, CLIP_DIM as i64, 2, st) }, "img_emb gelu")?;
        self.linear.forward_at(g2.dev(), self.b("img_emb.proj.3.weight"), Some(self.b("img_emb.proj.3.bias")), b4.dev(), n, DIM,
                               CLIP_DIM, Dtype::Bf16)?;
        let image_f32 = Buffer::new(n * DIM * 4)?;
        self.ln(LnMode::AffineF32, In::Bf16, b4.dev(), self.f("img_emb.proj.4.weight"), self.f("img_emb.proj.4.bias"), 0, 0,
                image_f32.dev(), n, LN_EPS_MLP, false)?;
        let image = Buffer::new(n * DIM * 2)?;
        check(unsafe { w21_cast(0, p(image_f32.dev()), p(image.dev()), (n * DIM) as i64, st) }, "context_clip cast")?;
        let (mut k, mut v, mut k_img, mut v_img) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let raw = Buffer::new(TEXT_LEN * DIM * 2)?;
        for i in 0..LAYERS {
            let (kb, vb, kib, vib) = (Buffer::new(TEXT_LEN * DIM * 2)?, Buffer::new(TEXT_LEN * DIM * 2)?, Buffer::new(n * DIM * 2)?,
                                      Buffer::new(n * DIM * 2)?);
            self.lin(text.dev(), &block_param(i, "cross_attn.k"), raw.dev(), TEXT_LEN, DIM, DIM)?;
            self.rms(raw.dev(), &block_param(i, "cross_attn.norm_k.weight"), kb.dev(), TEXT_LEN)?;
            self.lin(text.dev(), &block_param(i, "cross_attn.v"), vb.dev(), TEXT_LEN, DIM, DIM)?;
            self.lin(image.dev(), &block_param(i, "cross_attn.k_img"), raw.dev(), n, DIM, DIM)?;
            self.rms(raw.dev(), &block_param(i, "cross_attn.norm_k_img.weight"), kib.dev(), n)?;
            self.lin(image.dev(), &block_param(i, "cross_attn.v_img"), vib.dev(), n, DIM, DIM)?;
            k.push(kb); v.push(vb); k_img.push(kib); v_img.push(vib);
        }
        Ok(PassContext { text, image_f32, k, v, k_img, v_img })
    }

    /// e (fp32 [5120]) and e0 (fp32 [6, 5120]) of timestep `t` (the scheduler's int64 timestep).
    pub fn time(&self, t: i64) -> Result<()> {
        let (st, sc) = (self.s(), &self.sc);
        upload_into(&sc.t_dev, &t.to_le_bytes(), self.device)?;
        check(unsafe { w21_sinusoidal(p(sc.t_dev.dev()), p(sc.sinus.dev()), FREQ_DIM as c_int, st) }, "sinusoidal")?;
        self.lin_f32(sc.sinus.dev(), "time_embedding.0", sc.t1.dev(), 1, DIM, FREQ_DIM)?;
        eager::silu_f32(sc.t1.dev(), sc.t1s.dev(), DIM, st)?;
        self.lin_f32(sc.t1s.dev(), "time_embedding.2", sc.e.dev(), 1, DIM, DIM)?;
        eager::silu_f32(sc.e.dev(), sc.es.dev(), DIM, st)?;
        self.lin_f32(sc.es.dev(), "time_projection.1", sc.e0.dev(), 1, 6 * DIM, DIM)
    }

    /// One forward on the latent (fp32 [16, F, H, W]) with y (fp32 [20, F, H, W]); `time(t)` must
    /// have run for this step. Returns the fp32 prediction [16, F, H, W] in `out`.
    pub fn forward(&self, latent: u64, y: u64, ctx: &PassContext, out: u64, observe: &mut dyn FnMut(&str, u64) -> Result<()>)
                   -> Result<()> {
        let (st, sc, g) = (self.s(), &self.sc, self.grid);
        native_core_infra::execution_start::arm();
        native_core_infra::execution_start::enqueue_if_armed(st)?;
        let s = g.tokens();
        let (fr, hh, ww) = (g.frames, g.height / 2, g.width / 2);
        // Patch embedding: bf16(cat([x, y])) -> Conv3d(36 -> 5120, 1x2x2) -> + bias -> tokens.
        check(unsafe { w21_cat_cast_bf16(p(latent), p(y), p(sc.x_in.dev()), g.latent_elements(OUT_CHANNELS) as i64,
                                         g.latent_elements(IN_CHANNELS - OUT_CHANNELS) as i64, st) }, "patch input")?;
        let input = [1, IN_CHANNELS as i64, fr as i64, g.height as i64, g.width as i64];
        let weight = [DIM as i64, IN_CHANNELS as i64, 1, 2, 2];
        let output = [1, DIM as i64, fr as i64, hh as i64, ww as i64];
        let spec = ConvSpec { ndim: 5, dtype: 2, transposed: 0, allow_tf32: 1, deterministic: 1, channels_last: 0,
            input_dims: input, input_strides: contiguous(&input), weight_dims: weight, weight_strides: contiguous(&weight),
            output_dims: output, output_strides: contiguous(&output), padding: [0; 3], stride: [1, 2, 2], dilation: [1; 3] };
        self.conv.forward(&spec, sc.x_in.dev(), self.b("patch_embedding.weight"), sc.conv.dev(), st)?;
        eager::add_channel_bias_bf16(sc.conv.dev(), self.b("patch_embedding.bias"), DIM, s, st)?;
        eager::transpose_bf16(sc.conv.dev(), sc.tokens.dev(), DIM, s, st)?;
        observe("patch_embedding/out", sc.conv.dev())?;
        let row4 = (DIM * 4) as u64;
        // Block i's six modulation rows; blocks alternate between two buffers so that block i + 1's
        // are ready when block i's FFN residual is fused with block i + 1's norm1.
        let e_bufs = [sc.e_blk.dev(), sc.e_next.dev()];
        let modulation = |i: usize, dst: u64| -> Result<()> {
            check(unsafe { w21_add_bf16_f32(p(self.b(&block_param(i, "modulation"))), p(sc.e0.dev()), p(dst), (6 * DIM) as i64,
                                            (6 * DIM) as i64, st) }, "modulation")
        };
        modulation(0, e_bufs[0])?;
        for i in 0..LAYERS {
            let w = |n: &str| block_param(i, n);
            let (x_in, x_bf16) = if i == 0 { (sc.tokens.dev(), true) } else { (sc.x.dev(), false) };
            let eb = e_bufs[i % 2];
            let e = |k: u64| eb + k * row4;
            // Self-attention.
            let mut b0 = |name: &str, addr: u64| if i == 0 { observe(&format!("b0/{name}"), addr) } else { Ok(()) };
            // norm1: block 0's here; a later block's was written by the previous block's FFN residual.
            if i == 0 {
                self.ln(LnMode::Modulate, In::Bf16, x_in, 0, 0, e(0), e(1), sc.xm.dev(), s, EPS, x_bf16)?;
            }
            b0("xm_self", sc.xm.dev())?;
            self.lin(sc.xm.dev(), &w("self_attn.q"), sc.q.dev(), s, DIM, DIM)?;
            self.lin(sc.xm.dev(), &w("self_attn.k"), sc.k.dev(), s, DIM, DIM)?;
            self.lin(sc.xm.dev(), &w("self_attn.v"), sc.v.dev(), s, DIM, DIM)?;
            b0("q", sc.q.dev())?;
            b0("k", sc.k.dev())?;
            b0("v", sc.v.dev())?;
            if i == 0 {
                // Block 0 keeps every intermediate observable.
                self.rms(sc.q.dev(), &w("self_attn.norm_q.weight"), sc.qn.dev(), s)?;
                self.rms(sc.k.dev(), &w("self_attn.norm_k.weight"), sc.kn.dev(), s)?;
                b0("qn", sc.qn.dev())?;
                b0("kn", sc.kn.dev())?;
                let dbg = |k: usize| self.debug.as_ref().map_or(0, |d| d[k].dev());
                check(unsafe { w21_rope(p(sc.qn.dev()), p(self.freqs.dev()), p(sc.qr.dev()), p(dbg(0)), HEADS as c_int,
                                        fr as c_int, hh as c_int, ww as c_int, st) }, "rope q")?;
                check(unsafe { w21_rope(p(sc.kn.dev()), p(self.freqs.dev()), p(sc.kr.dev()), p(dbg(1)), HEADS as c_int,
                                        fr as c_int, hh as c_int, ww as c_int, st) }, "rope k")?;
                if dbg(0) != 0 { b0("qr_f32", dbg(0))?; b0("kr_f32", dbg(1))?; }
            } else {
                self.rms_rope(sc.q.dev(), &w("self_attn.norm_q.weight"), sc.qr.dev(), s, (fr, hh, ww))?;
                self.rms_rope(sc.k.dev(), &w("self_attn.norm_k.weight"), sc.kr.dev(), s, (fr, hh, ww))?;
            }
            self.attention(sc.qr.dev(), sc.kr.dev(), sc.v.dev(), sc.attn.dev(), s, s)?;
            b0("attn", sc.attn.dev())?;
            self.lin(sc.attn.dev(), &w("self_attn.o"), sc.y.dev(), s, DIM, DIM)?;
            b0("o", sc.y.dev())?;
            // Cross-attention (image keys first, then text), norm3 affine.
            if i == 0 {
                check(unsafe { w21_residual_gate(p(x_in), c_int::from(x_bf16), p(sc.y.dev()), p(e(2)), p(sc.x.dev()), s as i64,
                                                 DIM as i64, st) }, "residual self")?;
                b0("x1", sc.x.dev())?;
                self.ln(LnMode::AffineBf16, In::F32, sc.x.dev(), self.f(&w("norm3.weight")), self.f(&w("norm3.bias")), 0, 0,
                        sc.xm.dev(), s, EPS, false)?;
                b0("xm_cross", sc.xm.dev())?;
            } else {
                self.residual_ln(LnMode::AffineBf16, Some(e(2)), sc.y.dev(), self.f(&w("norm3.weight")), self.f(&w("norm3.bias")),
                                 0, 0, s)?;
            }
            self.lin(sc.xm.dev(), &w("cross_attn.q"), sc.q.dev(), s, DIM, DIM)?;
            b0("cq", sc.q.dev())?;
            self.rms(sc.q.dev(), &w("cross_attn.norm_q.weight"), sc.qn.dev(), s)?;
            b0("cqn", sc.qn.dev())?;
            self.attention(sc.qn.dev(), ctx.k_img[i].dev(), ctx.v_img[i].dev(), sc.img.dev(), s, CLIP_TOKENS)?;
            self.attention(sc.qn.dev(), ctx.k[i].dev(), ctx.v[i].dev(), sc.txt.dev(), s, TEXT_LEN)?;
            b0("img", sc.img.dev())?;
            b0("txt", sc.txt.dev())?;
            eager::add_bf16(sc.txt.dev(), sc.img.dev(), sc.sum.dev(), s * DIM, st)?;
            b0("sum", sc.sum.dev())?;
            self.lin(sc.sum.dev(), &w("cross_attn.o"), sc.y.dev(), s, DIM, DIM)?;
            b0("co", sc.y.dev())?;
            // FFN.
            if i == 0 {
                check(unsafe { w21_residual_add(p(sc.x.dev()), p(sc.y.dev()), p(sc.x.dev()), (s * DIM) as i64, st) },
                      "residual cross")?;
                b0("x2", sc.x.dev())?;
                self.ln(LnMode::Modulate, In::F32, sc.x.dev(), 0, 0, e(3), e(4), sc.xm.dev(), s, EPS, false)?;
                b0("xm_ffn", sc.xm.dev())?;
            } else {
                self.residual_ln(LnMode::Modulate, None, sc.y.dev(), 0, 0, e(3), e(4), s)?;
            }
            self.lin(sc.xm.dev(), &w("ffn.0"), sc.h.dev(), s, FFN, DIM)?;
            b0("h", sc.h.dev())?;
            eager::gelu_tanh_bf16(sc.h.dev(), sc.g.dev(), s * FFN, st)?;
            b0("g", sc.g.dev())?;
            self.lin(sc.g.dev(), &w("ffn.2"), sc.y.dev(), s, DIM, FFN)?;
            b0("f", sc.y.dev())?;
            if i + 1 < LAYERS {
                // x += f * e5, then the next block's norm1 of the new x (residual_gate's and
                // layer_norm's expressions in one pass).
                let en = e_bufs[(i + 1) % 2];
                modulation(i + 1, en)?;
                self.residual_ln(LnMode::Modulate, Some(e(5)), sc.y.dev(), 0, 0, en, en + row4, s)?;
            } else {
                check(unsafe { w21_residual_gate(p(sc.x.dev()), 0, p(sc.y.dev()), p(e(5)), p(sc.x.dev()), s as i64, DIM as i64, st) },
                      "residual ffn")?;
            }
            observe(&format!("blocks.{i}/out"), sc.x.dev())?;
        }
        // Head (fp32 autocast region): modulation + e, LayerNorm * (1 + e1) + e0, fp32 Linear.
        check(unsafe { w21_add_bf16_f32(p(self.b("head.modulation")), p(sc.e.dev()), p(sc.eh.dev()), (2 * DIM) as i64, DIM as i64,
                                        st) }, "head modulation")?;
        self.ln(LnMode::ModulateF32, In::F32, sc.x.dev(), 0, 0, sc.eh.dev(), sc.eh.dev() + row4, sc.sq.dev(), s, EPS, false)?;
        self.lin_f32(sc.sq.dev(), "head.head", sc.head.dev(), s, 64, DIM)?;
        observe("head/out", sc.head.dev())?;
        check(unsafe { w21_unpatchify(p(sc.head.dev()), p(out), fr as c_int, hh as c_int, ww as c_int, OUT_CHANNELS as c_int, st) },
              "unpatchify")
    }
}
