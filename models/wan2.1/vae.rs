//! The official Wan 2.1 VAE (wan/modules/vae.py, WanVAE with dtype fp32): causal 3D encoder and
//! decoder with the CACHE_T = 2 feature caches, encode in chunks of 1, 4, 4, ... frames and decode one
//! latent frame at a time. Convolutions are cuDNN fp32 with TF32 allowed (torch's default for
//! cudnn); RMS_norm is F.normalize over channels (torch's norm reduction), each eager op rounded once.
//!
//! A tensor is [C, T, H, W] fp32, contiguous or (after Resample's `rearrange` back) the permuted
//! view of [T, C, H, W] memory; the layout decides torch's reduction order in F.normalize, so it is
//! tracked (`tc`).
use crate::conv::{contiguous, Conv, ConvSpec};
use crate::cuda::{check, Buffer, Device, Stream};
use crate::weights::Source;
use crate::Result;
use std::collections::HashMap;
use std::ffi::{c_int, c_void};

extern "C" {
    fn w21_vae_pad_cat(cache: *const c_void, cache_frames: i64, x: *const c_void, x_tc: c_int, frames: i64, out: *mut c_void,
                       channels: i64, front: i64, out_frames: i64, h: i64, w: i64, ph0: i64, ph1: i64, pw0: i64, pw1: i64,
                       s: Stream) -> c_int;
    fn w21_vae_normalize(x: *const c_void, norm: *const c_void, gamma: *const c_void, out: *mut c_void, scale: f32, silu: c_int,
                         tc: c_int, c: i64, t: i64, hw: i64, s: Stream) -> c_int;
    fn w21_vae_bias(x: *mut c_void, b: *const c_void, channels: i64, inner: i64, batch: i64, s: Stream) -> c_int;
    fn w21_vae_add(x: *const c_void, h: *const c_void, h_tc: c_int, out: *mut c_void, c: i64, t: i64, hw: i64, s: Stream) -> c_int;
    fn w21_vae_swap01(x: *const c_void, out: *mut c_void, a: i64, b: i64, s_: i64, s: Stream) -> c_int;
    fn w21_vae_upsample2x(x: *const c_void, out: *mut c_void, nc: i64, h: i64, w: i64, s: Stream) -> c_int;
    fn w21_vae_time_interleave(x: *const c_void, out: *mut c_void, c: i64, t: i64, hw: i64, s: Stream) -> c_int;
    fn w21_vae_slice_time(x: *const c_void, tc: c_int, out: *mut c_void, c: i64, t: i64, start: i64, count: i64, hw: i64,
                          s: Stream) -> c_int;
    fn w21_vae_place_time(x: *const c_void, tc: c_int, out: *mut c_void, c: i64, t_out: i64, start: i64, count: i64, hw: i64,
                          s: Stream) -> c_int;
    fn w21_vae_scale_encode(mu: *const c_void, mean: *const c_void, inv: *const c_void, out: *mut c_void, channels: i64,
                            inner: i64, s: Stream) -> c_int;
    fn w21_vae_scale_decode(z: *const c_void, mean: *const c_void, inv: *const c_void, out: *mut c_void, channels: i64,
                            inner: i64, s: Stream) -> c_int;
    fn w21_vae_clamp(x: *mut c_void, n: i64, s: Stream) -> c_int;
    fn wan_norm2_f32(x: *const c_void, out: *mut c_void, ndim: c_int, sizes: *const i64, strides: *const i64, mask: u32,
                     s: Stream) -> c_int;
    fn w21_mem_eff_attention_f32(q: *const c_void, k: *const c_void, v: *const c_void, out: *mut c_void, batch: i64,
                                 queries: i64, keys: i64, heads: i64, head_dim: i64, q_sb: i64, q_sm: i64, q_sh: i64, k_sb: i64,
                                 k_sm: i64, k_sh: i64, v_sb: i64, v_sm: i64, v_sh: i64, s: Stream) -> c_int;
}
fn p(a: u64) -> *mut c_void { a as *mut c_void }

pub const Z_DIM: usize = 16;
const DIM: usize = 96;

/// fp32 [c, t, h, w]; `tc`: memory order [t, c, h, w].
pub struct VTensor { pub buf: Buffer, pub c: usize, pub t: usize, pub h: usize, pub w: usize, pub tc: bool }
impl VTensor {
    pub fn new(c: usize, t: usize, h: usize, w: usize) -> Result<Self> {
        Ok(Self { buf: Buffer::new(c * t * h * w * 4)?, c, t, h, w, tc: false })
    }
    fn hw(&self) -> usize { self.h * self.w }
    pub fn elements(&self) -> usize { self.c * self.t * self.h * self.w }
    pub fn dev(&self) -> u64 { self.buf.dev() }
    /// Element strides of the [1, c, t, h, w] tensor torch sees.
    fn strides5(&self) -> [i64; 5] {
        let (c, t, hw, w) = (self.c as i64, self.t as i64, self.hw() as i64, self.w as i64);
        if self.tc { [t * c * hw, hw, c * hw, w, 1] } else { [c * t * hw, t * hw, hw, w, 1] }
    }
}

/// Feature cache slot: never set, the upsample "Rep" marker, or saved frames (contiguous).
enum Cache { Empty, Rep, Frames(VTensor) }
struct Cursor<'c> { slots: &'c mut Vec<Cache>, idx: usize }
impl Cursor<'_> {
    fn slot(&mut self) -> usize {
        if self.idx >= self.slots.len() { self.slots.resize_with(self.idx + 1, || Cache::Empty); }
        self.idx
    }
}

/// WanVAE mean / std (fp32 tensors of the Python lists); scale[1] = 1.0 / std is torch's reciprocal.
const MEAN: [f32; 16] = [-0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508, 0.4134, -0.0715, 0.5517, -0.3632,
                         -0.1922, -0.9497, 0.2503, -0.2921];
const STD: [f32; 16] = [2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743, 3.2687, 2.1526, 2.8652, 1.5579,
                        1.6382, 1.1253, 2.8251, 1.9160];

pub struct Vae<'d> {
    device: &'d Device,
    conv: Conv,
    weights: HashMap<String, Buffer>,
    mean: Buffer,
    inv_std: Buffer,
}

impl<'d> Vae<'d> {
    pub fn load(weights: &dyn Source, device: &'d Device) -> Result<Self> {
        let names = weights.names()?;
        let buffers = weights.buffers(&names, device)?;
        let map: HashMap<String, Buffer> = names.into_iter().zip(buffers).collect();
        let mean: Vec<u8> = MEAN.iter().flat_map(|v| v.to_le_bytes()).collect();
        let inv: Vec<u8> = STD.iter().flat_map(|v| (1.0f32 / *v).to_le_bytes()).collect();
        Ok(Self { device, conv: Conv::new(16 << 30)?, weights: map, mean: Buffer::upload(&mean, device)?,
                  inv_std: Buffer::upload(&inv, device)? })
    }
    fn st(&self) -> Stream { self.device.stream }
    fn w(&self, name: &str) -> Result<u64> {
        self.weights.get(name).map(|b| b.dev()).ok_or_else(|| format!("VAE weight {name} missing").into())
    }

    // ---- primitives -----------------------------------------------------------------------------
    /// x[:, :, start:start+count] as a contiguous tensor.
    fn slice(&self, x: &VTensor, start: usize, count: usize) -> Result<VTensor> {
        let out = VTensor::new(x.c, count, x.h, x.w)?;
        check(unsafe { w21_vae_slice_time(p(x.dev()), c_int::from(x.tc), p(out.dev()), x.c as i64, x.t as i64, start as i64,
                                          count as i64, x.hw() as i64, self.st()) }, "slice_time")?;
        Ok(out)
    }
    fn cat_time(&self, a: &VTensor, b: &VTensor) -> Result<VTensor> {
        let out = VTensor::new(a.c, a.t + b.t, a.h, a.w)?;
        for (x, start) in [(a, 0), (b, a.t)] {
            check(unsafe { w21_vae_place_time(p(x.dev()), c_int::from(x.tc), p(out.dev()), x.c as i64, out.t as i64, start as i64,
                                              x.t as i64, x.hw() as i64, self.st()) }, "place_time")?;
        }
        Ok(out)
    }

    /// nn.Conv3d / nn.Conv2d (fp32, TF32 allowed) on a contiguous padded input, then the bias add_.
    #[allow(clippy::too_many_arguments)]
    fn conv_raw(&self, input: u64, dims_in: &[i64], weight: &str, bias: &str, cout: usize, kernel: &[i64], stride: &[i64],
                padding: &[i64], dims_out: &[i64], out: u64) -> Result<()> {
        let nd = dims_in.len();
        let mut spec = ConvSpec { ndim: nd as i32, dtype: 0, transposed: 0, allow_tf32: 1, deterministic: 1, channels_last: 0,
                                  ..Default::default() };
        spec.input_dims[..nd].copy_from_slice(dims_in);
        spec.input_strides = contiguous(dims_in);
        let mut wd = vec![cout as i64, dims_in[1]];
        wd.extend_from_slice(kernel);
        spec.weight_dims[..nd].copy_from_slice(&wd);
        spec.weight_strides = contiguous(&wd);
        spec.output_dims[..nd].copy_from_slice(dims_out);
        spec.output_strides = contiguous(dims_out);
        for i in 0..nd - 2 { spec.padding[i] = padding[i]; spec.stride[i] = stride[i]; spec.dilation[i] = 1; }
        self.conv.forward(&spec, input, self.w(weight)?, out, self.st())?;
        let inner: i64 = dims_out[2..].iter().product();
        check(unsafe { w21_vae_bias(p(out), p(self.w(bias)?), dims_out[1], inner, dims_out[0], self.st()) }, "vae bias")
    }

    /// CausalConv3d.forward(x, cache_x): cat the cache in time, F.pad (W, H symmetric; T front
    /// 2*pt minus cached frames), conv with zero padding. Output contiguous.
    #[allow(clippy::too_many_arguments)]
    fn causal_conv(&self, x: &VTensor, prefix: &str, cout: usize, k: [usize; 3], pad: [usize; 3], stride_t: usize,
                   cache: Option<&VTensor>) -> Result<VTensor> {
        let [pt, ph, pw] = pad;
        let mut front = 2 * pt;
        let (cache_ptr, cache_frames) = match cache {
            Some(c) if front > 0 => { front -= c.t.min(front); (c.dev(), c.t) }
            _ => (0, 0),
        };
        let frames = cache_frames + x.t;
        let tp = front + frames;
        let (hp, wp) = (x.h + 2 * ph, x.w + 2 * pw);
        let padded = VTensor::new(x.c, tp, hp, wp)?;
        check(unsafe { w21_vae_pad_cat(p(cache_ptr), cache_frames as i64, p(x.dev()), c_int::from(x.tc), x.t as i64,
                                       p(padded.dev()), x.c as i64, front as i64, tp as i64, x.h as i64, x.w as i64, ph as i64,
                                       ph as i64, pw as i64, pw as i64, self.st()) }, "pad_cat")?;
        let to = (tp - k[0]) / stride_t + 1;
        let (ho, wo) = (hp - k[1] + 1, wp - k[2] + 1);
        let out = VTensor::new(cout, to, ho, wo)?;
        self.conv_raw(padded.dev(), &[1, x.c as i64, tp as i64, hp as i64, wp as i64], &format!("{prefix}.weight"),
                      &format!("{prefix}.bias"), cout, &[k[0] as i64, k[1] as i64, k[2] as i64], &[stride_t as i64, 1, 1],
                      &[0, 0, 0], &[1, cout as i64, to as i64, ho as i64, wo as i64], out.dev())?;
        Ok(out)
    }

    /// RMS_norm (F.normalize over channels * sqrt(C) * gamma + 0.), optionally followed by nn.SiLU.
    /// The output keeps x's layout.
    fn rms_norm(&self, x: &VTensor, gamma: &str, silu: bool) -> Result<VTensor> {
        let norm = Buffer::new(x.t * x.hw() * 4)?;
        let sizes = [1, x.c as i64, x.t as i64, x.h as i64, x.w as i64];
        let strides = x.strides5();
        check(unsafe { wan_norm2_f32(p(x.dev()), p(norm.dev()), 5, sizes.as_ptr(), strides.as_ptr(), 0b10, self.st()) }, "norm")?;
        let mut out = VTensor::new(x.c, x.t, x.h, x.w)?;
        out.tc = x.tc;
        let scale = (x.c as f64).sqrt() as f32;
        check(unsafe { w21_vae_normalize(p(x.dev()), p(norm.dev()), p(self.w(gamma)?), p(out.dev()), scale, c_int::from(silu),
                                         c_int::from(x.tc), x.c as i64, x.t as i64, x.hw() as i64, self.st()) }, "normalize")?;
        Ok(out)
    }

    /// cache_x = x[:, :, -2:].clone(); with fewer than two frames, the slot's last frame first.
    fn cache_frames(&self, x: &VTensor, previous: &Cache) -> Result<VTensor> {
        let keep = x.t.min(2);
        let tail = self.slice(x, x.t - keep, keep)?;
        match previous {
            Cache::Frames(prev) if keep < 2 => { let last = self.slice(prev, prev.t - 1, 1)?; self.cat_time(&last, &tail) }
            _ => Ok(tail),
        }
    }

    /// A cached CausalConv3d slot (residual convs, conv1, head conv).
    fn cached_conv(&self, cur: &mut Cursor, x: &VTensor, prefix: &str, cout: usize) -> Result<VTensor> {
        let idx = cur.slot();
        let cache_x = self.cache_frames(x, &cur.slots[idx])?;
        let cache = match &cur.slots[idx] { Cache::Frames(t) => Some(t), _ => None };
        let out = self.causal_conv(x, prefix, cout, [3, 3, 3], [1, 1, 1], 1, cache)?;
        cur.slots[idx] = Cache::Frames(cache_x);
        cur.idx += 1;
        Ok(out)
    }

    fn residual_block(&self, cur: &mut Cursor, x: &VTensor, prefix: &str, cin: usize, cout: usize) -> Result<VTensor> {
        let shortcut = if cin != cout { Some(self.causal_conv(x, &format!("{prefix}.shortcut"), cout, [1, 1, 1], [0, 0, 0], 1, None)?) }
                       else { None };
        let y = self.rms_norm(x, &format!("{prefix}.residual.0.gamma"), true)?;
        let y = self.cached_conv(cur, &y, &format!("{prefix}.residual.2"), cout)?;
        let y = self.rms_norm(&y, &format!("{prefix}.residual.3.gamma"), true)?;
        let y = self.cached_conv(cur, &y, &format!("{prefix}.residual.6"), cout)?;
        // x + h: the conv output is contiguous; h is the shortcut output or x itself (any layout).
        let h = shortcut.as_ref().unwrap_or(x);
        let out = VTensor::new(y.c, y.t, y.h, y.w)?;
        check(unsafe { w21_vae_add(p(y.dev()), p(h.dev()), c_int::from(h.tc), p(out.dev()), y.c as i64, y.t as i64,
                                   y.hw() as i64, self.st()) }, "residual add")?;
        Ok(out)
    }

    /// AttentionBlock (t = 1 in both middles): RMS_norm, 1x1 qkv conv, single-head fp32 SDPA (torch's
    /// mem-efficient kernel), 1x1 proj conv, + identity.
    fn attention_block(&self, x: &VTensor, prefix: &str) -> Result<VTensor> {
        if x.t != 1 { return Err("the VAE attention block runs on single frames".into()); }
        let (st, c, hw) = (self.st(), x.c, x.hw());
        let normed = self.rms_norm(x, &format!("{prefix}.norm.gamma"), false)?;
        let qkv = Buffer::new(3 * c * hw * 4)?;
        let dims = [1, c as i64, x.h as i64, x.w as i64];
        self.conv_raw(normed.dev(), &dims, &format!("{prefix}.to_qkv.weight"), &format!("{prefix}.to_qkv.bias"), 3 * c, &[1, 1],
                      &[1, 1], &[0, 0], &[1, 3 * c as i64, x.h as i64, x.w as i64], qkv.dev())?;
        // reshape(1, 1, 3c, hw).permute(0, 1, 3, 2).contiguous(): [3c, hw] -> [hw, 3c]; chunk(3, -1).
        let packed = Buffer::new(3 * c * hw * 4)?;
        check(unsafe { w21_vae_swap01(p(qkv.dev()), p(packed.dev()), (3 * c) as i64, hw as i64, 1, st) }, "qkv transpose")?;
        let out = Buffer::new(hw * c * 4)?;
        let (sb, sm, sh) = ((hw * 3 * c) as i64, (3 * c) as i64, (hw * 3 * c) as i64);
        let (q, k, v) = (packed.dev(), packed.dev() + (c * 4) as u64, packed.dev() + (2 * c * 4) as u64);
        check(unsafe { w21_mem_eff_attention_f32(p(q), p(k), p(v), p(out.dev()), 1, hw as i64, hw as i64, 1, c as i64, sb, sm, sh,
                                                 sb, sm, sh, sb, sm, sh, st) }, "vae attention")?;
        // squeeze(1).permute(0, 2, 1).reshape(1, c, h, w): [hw, c] -> [c, hw].
        let chw = Buffer::new(c * hw * 4)?;
        check(unsafe { w21_vae_swap01(p(out.dev()), p(chw.dev()), hw as i64, c as i64, 1, st) }, "attention transpose")?;
        let proj = VTensor::new(c, 1, x.h, x.w)?;
        self.conv_raw(chw.dev(), &dims, &format!("{prefix}.proj.weight"), &format!("{prefix}.proj.bias"), c, &[1, 1], &[1, 1],
                      &[0, 0], &dims, proj.dev())?;
        let y = VTensor::new(c, 1, x.h, x.w)?;
        check(unsafe { w21_vae_add(p(proj.dev()), p(x.dev()), c_int::from(x.tc), p(y.dev()), c as i64, 1, hw as i64, st) },
              "attention residual")?;
        Ok(y)
    }

    /// Resample.forward (upsample with nearest-exact, downsample with ZeroPad2d + stride-2 conv).
    fn resample(&self, cur: &mut Cursor, x: &VTensor, prefix: &str, mode: &str, dim: usize) -> Result<VTensor> {
        let st = self.st();
        let mut x_owned: Option<VTensor> = None;
        if mode == "upsample3d" {
            let idx = cur.slot();
            if matches!(cur.slots[idx], Cache::Empty) {
                cur.slots[idx] = Cache::Rep;
                cur.idx += 1;
            } else {
                let keep = x.t.min(2);
                let tail = self.slice(x, x.t - keep, keep)?;
                let cache_x = match &cur.slots[idx] {
                    Cache::Frames(prev) if keep < 2 => { let last = self.slice(prev, prev.t - 1, 1)?; self.cat_time(&last, &tail)? }
                    Cache::Rep if keep < 2 => {
                        let zeros = VTensor::new(tail.c, tail.t, tail.h, tail.w)?;
                        zeros.buf.zero(self.device)?;
                        self.cat_time(&zeros, &tail)?
                    }
                    _ => tail,
                };
                let conv = match &cur.slots[idx] {
                    Cache::Rep => self.causal_conv(x, &format!("{prefix}.time_conv"), 2 * dim, [3, 1, 1], [1, 0, 0], 1, None)?,
                    Cache::Frames(prev) => self.causal_conv(x, &format!("{prefix}.time_conv"), 2 * dim, [3, 1, 1], [1, 0, 0], 1, Some(prev))?,
                    Cache::Empty => unreachable!(),
                };
                cur.slots[idx] = Cache::Frames(cache_x);
                cur.idx += 1;
                let inter = VTensor::new(dim, 2 * conv.t, conv.h, conv.w)?;
                check(unsafe { w21_vae_time_interleave(p(conv.dev()), p(inter.dev()), dim as i64, conv.t as i64, conv.hw() as i64, st) },
                      "time interleave")?;
                x_owned = Some(inter);
            }
        }
        let x = x_owned.as_ref().unwrap_or(x);
        // rearrange b c t h w -> (b t) c h w (a copy unless already [t, c] ordered), resample, back (a view).
        let t = x.t;
        let frames = Buffer::new(x.elements() * 4)?;
        if x.tc || t == 1 {
            frames.copy_from_device(x.dev(), x.elements() * 4, self.device)?;
        } else {
            check(unsafe { w21_vae_swap01(p(x.dev()), p(frames.dev()), x.c as i64, t as i64, x.hw() as i64, st) }, "to frames")?;
        }
        let mut y = match mode {
            "upsample2d" | "upsample3d" => {
                let up = Buffer::new(t * x.c * 4 * x.hw() * 4)?;
                check(unsafe { w21_vae_upsample2x(p(frames.dev()), p(up.dev()), (t * x.c) as i64, x.h as i64, x.w as i64, st) },
                      "upsample")?;
                let (oh, ow) = (2 * x.h, 2 * x.w);
                let out = VTensor { buf: Buffer::new(t * (dim / 2) * oh * ow * 4)?, c: dim / 2, t, h: oh, w: ow, tc: true };
                self.conv_raw(up.dev(), &[t as i64, x.c as i64, oh as i64, ow as i64], &format!("{prefix}.resample.1.weight"),
                              &format!("{prefix}.resample.1.bias"), dim / 2, &[3, 3], &[1, 1], &[1, 1],
                              &[t as i64, (dim / 2) as i64, oh as i64, ow as i64], out.dev())?;
                out
            }
            "downsample2d" | "downsample3d" => {
                let padded = Buffer::new(t * x.c * (x.h + 1) * (x.w + 1) * 4)?;
                check(unsafe { w21_vae_pad_cat(p(0), 0, p(frames.dev()), 0, 1, p(padded.dev()), (t * x.c) as i64, 0, 1, x.h as i64,
                                               x.w as i64, 0, 1, 0, 1, st) }, "zero pad 2d")?;
                let (oh, ow) = ((x.h + 1 - 3) / 2 + 1, (x.w + 1 - 3) / 2 + 1);
                let out = VTensor { buf: Buffer::new(t * dim * oh * ow * 4)?, c: dim, t, h: oh, w: ow, tc: true };
                self.conv_raw(padded.dev(), &[t as i64, x.c as i64, (x.h + 1) as i64, (x.w + 1) as i64],
                              &format!("{prefix}.resample.1.weight"), &format!("{prefix}.resample.1.bias"), dim, &[3, 3],
                              &[2, 2], &[0, 0], &[t as i64, dim as i64, oh as i64, ow as i64], out.dev())?;
                out
            }
            _ => return Err(format!("unknown resample mode {mode}").into()),
        };
        if y.t == 1 { y.tc = false; }
        if mode == "downsample3d" {
            let idx = cur.slot();
            match &cur.slots[idx] {
                Cache::Empty => {
                    let copy = self.slice(&y, 0, y.t)?;
                    cur.slots[idx] = Cache::Frames(copy);
                }
                Cache::Frames(prev) => {
                    let cache_x = self.slice(&y, y.t - 1, 1)?;
                    let last = self.slice(prev, prev.t - 1, 1)?;
                    let joined = self.cat_time(&last, &y)?;
                    y = self.causal_conv(&joined, &format!("{prefix}.time_conv"), dim, [3, 1, 1], [0, 0, 0], 2, None)?;
                    cur.slots[idx] = Cache::Frames(cache_x);
                }
                Cache::Rep => return Err("downsample cache cannot hold Rep".into()),
            }
            cur.idx += 1;
        }
        Ok(y)
    }

    fn head(&self, cur: &mut Cursor, x: &VTensor, prefix: &str, cout: usize) -> Result<VTensor> {
        let x = self.rms_norm(x, &format!("{prefix}.0.gamma"), true)?;
        self.cached_conv(cur, &x, &format!("{prefix}.2"), cout)
    }

    // ---- encoder / decoder ----------------------------------------------------------------------
    fn encoder(&self, cur: &mut Cursor, x: &VTensor) -> Result<VTensor> {
        let mut x = self.cached_conv(cur, x, "encoder.conv1", DIM)?;
        let dims = [DIM, DIM, 2 * DIM, 4 * DIM, 4 * DIM];
        let downsample = ["downsample2d", "downsample3d", "downsample3d"];
        let mut index = 0;
        for i in 0..4 {
            let (mut cin, cout) = (dims[i], dims[i + 1]);
            for _ in 0..2 {
                x = self.residual_block(cur, &x, &format!("encoder.downsamples.{index}"), cin, cout)?;
                cin = cout;
                index += 1;
            }
            if i != 3 {
                x = self.resample(cur, &x, &format!("encoder.downsamples.{index}"), downsample[i], cout)?;
                index += 1;
            }
        }
        x = self.residual_block(cur, &x, "encoder.middle.0", 4 * DIM, 4 * DIM)?;
        x = self.attention_block(&x, "encoder.middle.1")?;
        x = self.residual_block(cur, &x, "encoder.middle.2", 4 * DIM, 4 * DIM)?;
        self.head(cur, &x, "encoder.head", 2 * Z_DIM)
    }

    fn decoder(&self, cur: &mut Cursor, x: &VTensor) -> Result<VTensor> {
        let mut x = self.cached_conv(cur, x, "decoder.conv1", 4 * DIM)?;
        x = self.residual_block(cur, &x, "decoder.middle.0", 4 * DIM, 4 * DIM)?;
        x = self.attention_block(&x, "decoder.middle.1")?;
        x = self.residual_block(cur, &x, "decoder.middle.2", 4 * DIM, 4 * DIM)?;
        let dims = [4 * DIM, 4 * DIM, 4 * DIM, 2 * DIM, DIM];
        let upsample = ["upsample3d", "upsample3d", "upsample2d"];
        let mut index = 0;
        for i in 0..4 {
            let (mut cin, cout) = (dims[i], dims[i + 1]);
            if i > 0 { cin /= 2; }
            for _ in 0..3 {
                x = self.residual_block(cur, &x, &format!("decoder.upsamples.{index}"), cin, cout)?;
                cin = cout;
                index += 1;
            }
            if i != 3 {
                x = self.resample(cur, &x, &format!("decoder.upsamples.{index}"), upsample[i], cout)?;
                index += 1;
            }
        }
        self.head(cur, &x, "decoder.head", 3)
    }

    /// WanVAE.encode: fp32 video [3, T, H, W] -> fp32 latent [16, 1 + (T-1)/4, H/8, W/8].
    pub fn encode(&self, video: &VTensor) -> Result<Buffer> {
        let mut slots = Vec::new();
        let chunks = 1 + (video.t - 1) / 4;
        let mut out: Option<VTensor> = None;
        for i in 0..chunks {
            let (start, count) = if i == 0 { (0, 1) } else { (1 + 4 * (i - 1), 4) };
            let chunk = self.slice(video, start, count)?;
            let mut cur = Cursor { slots: &mut slots, idx: 0 };
            let o = self.encoder(&mut cur, &chunk)?;
            out = Some(match out { None => o, Some(prev) => self.cat_time(&prev, &o)? });
        }
        let out = out.ok_or("empty video")?;
        let h = self.causal_conv(&out, "conv1", 2 * Z_DIM, [1, 1, 1], [0, 0, 0], 1, None)?;
        // mu = first z_dim channels (chunk(2, dim=1)); (mu - mean) * inv_std.
        let inner = h.t * h.hw();
        let latent = Buffer::new(Z_DIM * inner * 4)?;
        check(unsafe { w21_vae_scale_encode(p(h.dev()), p(self.mean.dev()), p(self.inv_std.dev()), p(latent.dev()), Z_DIM as i64,
                                            inner as i64, self.st()) }, "scale encode")?;
        Ok(latent)
    }

    /// WanVAE.decode: fp32 latent [16, T, h, w] -> fp32 video [3, 1 + 4(T-1), 8h, 8w] clamped to [-1, 1].
    pub fn decode(&self, z: u64, t: usize, h: usize, w: usize) -> Result<(Buffer, usize)> {
        self.decode_with_start(z,t,h,w,|_|Ok(()))
    }
    pub fn decode_with_start(&self,z:u64,t:usize,h:usize,w:usize,start:impl FnOnce(Stream)->Result<()>)->Result<(Buffer,usize)>{
        let st = self.st();
        let inner = t * h * w;
        let scaled = VTensor::new(Z_DIM, t, h, w)?;
        start(st)?;
        check(unsafe { w21_vae_scale_decode(p(z), p(self.mean.dev()), p(self.inv_std.dev()), p(scaled.dev()), Z_DIM as i64,
                                            inner as i64, st) }, "scale decode")?;
        let x = self.causal_conv(&scaled, "conv2", Z_DIM, [1, 1, 1], [0, 0, 0], 1, None)?;
        let mut slots = Vec::new();
        let frames_total = 1 + 4 * (t - 1);
        let video = VTensor::new(3, frames_total, 8 * h, 8 * w)?;
        let mut written = 0;
        for i in 0..t {
            let frame = self.slice(&x, i, 1)?;
            let mut cur = Cursor { slots: &mut slots, idx: 0 };
            let o = self.decoder(&mut cur, &frame)?;
            check(unsafe { w21_vae_place_time(p(o.dev()), c_int::from(o.tc), p(video.dev()), 3, frames_total as i64, written as i64,
                                              o.t as i64, o.hw() as i64, st) }, "decode cat")?;
            written += o.t;
        }
        if written != frames_total { return Err(format!("decode produced {written} frames").into()); }
        check(unsafe { w21_vae_clamp(p(video.dev()), video.elements() as i64, st) }, "clamp")?;
        Ok((video.buf, frames_total))
    }
}
