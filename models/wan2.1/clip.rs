//! CLIPModel.visual (open-clip xlm-roberta ViT-H/14) as the pipeline calls it on the
//! conditioning frame: bicubic resize to 224 in fp32 (outer bf16 autocast), Normalize, then the
//! fp16-autocast vision tower through the first 31 of 32 blocks (use_31_block). Attention is
//! cuDNN SDPA (the reference's FlashAttention replacement); LayerNorms run in fp32.
use crate::conv::{contiguous, Conv, ConvSpec};
use crate::cuda::{check, Buffer, Device, Stream};
use crate::eager::layer_norm_f32;
use crate::weights::Source;
use crate::linear::{Dtype, Linear};
use crate::sdpa::{self, Io, View};
use crate::encoders::gelu_erf;
use crate::Result;
use std::ffi::{c_int, c_void};

extern "C" {
    fn wan_upsample_bicubic2d_f32(x: *const c_void, y: *mut c_void, planes: c_int, in_h: c_int, in_w: c_int, out_h: c_int,
                                  out_w: c_int, stream: Stream) -> c_int;
    fn wan_clip_normalize(x: *const c_void, out: *mut c_void, hw: i64, mean: *const f32, std: *const f32,
                          stream: Stream) -> c_int;
    fn wan_clip_embed(patches: *const c_void, cls: *const c_void, pos: *const c_void, out: *mut c_void, c: i64, p: i64,
                      stream: Stream) -> c_int;
    fn wan_add_f16(a: *const c_void, b: *const c_void, out: *mut c_void, n: i64, stream: Stream) -> c_int;
    fn wan_f16_to_f32(x: *const c_void, y: *mut c_void, n: i64, stream: Stream) -> c_int;
    fn wan_f32_to_f16(x: *const c_void, y: *mut c_void, n: i64, stream: Stream) -> c_int;
}
fn p(a: u64) -> *mut c_void { a as *mut c_void }

const IMAGE: usize = 224;
const PATCH: usize = 14;
const DIM: usize = 1280;
const HEADS: usize = 16;
const HEAD_DIM: usize = 80;
const MLP: usize = 5120;
const BLOCKS: usize = 31;
const TOKENS: usize = (IMAGE / PATCH) * (IMAGE / PATCH) + 1;
const EPS: f32 = 1e-5;
/// open-clip mean / std, as the fp32 tensors torchvision's Normalize builds.
const MEAN: [f64; 3] = [0.48145466, 0.4578275, 0.40821073];
const STD: [f64; 3] = [0.26862954, 0.26130258, 0.27577711];

struct Block { norm1: [Buffer; 2], qkv: [Buffer; 2], proj: [Buffer; 2], norm2: [Buffer; 2], fc1: [Buffer; 2], fc2: [Buffer; 2] }

pub struct ClipVisual<'d> {
    device: &'d Device,
    linear: Linear,
    conv: Conv,
    patch: Buffer,
    cls: Buffer,
    pos: Buffer,
    pre_norm: [Buffer; 2],
    blocks: Vec<Block>,
}

impl<'d> ClipVisual<'d> {
    /// Builds the attention plan in the process-wide cache; may run on another thread while
    /// weights load.
    pub fn prewarm() -> Result<()> {
        let (qv, kv, vv) = Self::qkv_views();
        sdpa::prepare(Io::F16, &qv, &kv, &vv)?;
        Ok(())
    }
    /// `weights`: the clip component (fp16 parameters), as the reference holds it in memory.
    pub fn load(weights: &dyn Source, device: &'d Device) -> Result<Self> {
        let up = |name: &str| -> Result<Buffer> { weights.buffer(name, device) };
        // LayerNorm parameters are upcast to fp32 by autocast's layer_norm policy (exact).
        let ln = |base: &str| -> Result<[Buffer; 2]> {
            let widen = |name: String| -> Result<Buffer> {
                let half = up(&name)?;
                let out = Buffer::new(half.bytes() * 2)?;
                check(unsafe { wan_f16_to_f32(p(half.dev()), p(out.dev()), (half.bytes() / 2) as i64, device.stream) }, "ln widen")?;
                device.sync()?;
                Ok(out)
            };
            Ok([widen(format!("{base}.weight"))?, widen(format!("{base}.bias"))?])
        };
        let pair = |base: &str| -> Result<[Buffer; 2]> { Ok([up(&format!("{base}.weight"))?, up(&format!("{base}.bias"))?]) };
        let blocks = (0..BLOCKS).map(|i| {
            let b = format!("visual.transformer.{i}");
            Ok(Block { norm1: ln(&format!("{b}.norm1"))?, qkv: pair(&format!("{b}.attn.to_qkv"))?,
                       proj: pair(&format!("{b}.attn.proj"))?, norm2: ln(&format!("{b}.norm2"))?,
                       fc1: pair(&format!("{b}.mlp.0"))?, fc2: pair(&format!("{b}.mlp.2"))? })
        }).collect::<Result<Vec<_>>>()?;
        // Build the cuDNN SDPA plan now (a no-op after `prewarm`): its first build compiles the
        // attention kernels (~1 s), which would otherwise land in the first request.
        Self::prewarm()?;
        Ok(Self {
            device, linear: Linear::new(device)?, conv: Conv::new(16 << 30)?, patch: up("visual.patch_embedding.weight")?,
            cls: up("visual.cls_embedding")?, pos: up("visual.pos_embedding")?, pre_norm: ln("visual.pre_norm")?, blocks,
        })
    }

    /// q, k and v as [1, HEADS, TOKENS, HEAD_DIM] views of the packed qkv rows [TOKENS, 3 * DIM].
    fn qkv_views() -> (View, View, View) {
        let (t, d) = (TOKENS as i64, DIM as i64);
        // `q[0, :257]` of the unbind view, `.transpose(0, 1).unsqueeze(0)` (the gold's SDPA call).
        let view = |offset: usize| View { dims: [1, HEADS as i64, t, HEAD_DIM as i64], strides: [d, HEAD_DIM as i64, 3 * d, 1],
                                          offset_bytes: offset * 2 };
        (view(0), view(DIM), view(2 * DIM))
    }

    /// LayerNorm(x.float()).type_as(x) on [TOKENS, DIM] fp16.
    fn norm(&self, x: u64, w: &[Buffer; 2], wide: u64, normed: u64, out: u64) -> Result<()> {
        let (s, n) = (self.device.stream, (TOKENS * DIM) as i64);
        check(unsafe { wan_f16_to_f32(p(x), p(wide), n, s) }, "ln in")?;
        layer_norm_f32(wide, w[0].dev(), w[1].dev(), normed, TOKENS, DIM, EPS, s)?;
        check(unsafe { wan_f32_to_f16(p(normed), p(out), n, s) }, "ln out")
    }

    /// image: the conditioning frame [3, H, W] bf16 in [-1, 1]; out: [257, 1280] fp16.
    /// CLIPModel.visual([img[:, None]]): `image` is the fp32 [3, H, W] tensor in [-1, 1] (to_tensor,
    /// sub_(0.5), div_(0.5)); bicubic to 224 in fp32, mul_(0.5).add_(0.5), Normalize, then the fp16
    /// transformer (31 blocks). `out`: fp16 [257, 1280].
    pub fn forward(&self, image: u64, height: usize, width: usize, out: u64) -> Result<()> {
        let s = self.device.stream;
        let small = Buffer::new(3 * IMAGE * IMAGE * 4)?;
        check(unsafe { wan_upsample_bicubic2d_f32(p(image), p(small.dev()), 3, height as c_int, width as c_int,
                                                  IMAGE as c_int, IMAGE as c_int, s) }, "bicubic")?;
        let mean: Vec<f32> = MEAN.iter().map(|&v| v as f32).collect();
        let std: Vec<f32> = STD.iter().map(|&v| v as f32).collect();
        let pixels = Buffer::new(3 * IMAGE * IMAGE * 2)?;
        check(unsafe { wan_clip_normalize(p(small.dev()), p(pixels.dev()), (IMAGE * IMAGE) as i64, mean.as_ptr(), std.as_ptr(), s) },
              "normalize")?;
        // Patch embedding: fp16 cuDNN conv, kernel = stride = 14, no bias.
        let grid = IMAGE / PATCH;
        let patches = Buffer::new(DIM * grid * grid * 2)?;
        let input_dims = [1, 3, IMAGE as i64, IMAGE as i64];
        let weight_dims = [DIM as i64, 3, PATCH as i64, PATCH as i64];
        let output_dims = [1, DIM as i64, grid as i64, grid as i64];
        let mut spec = ConvSpec { ndim: 4, dtype: 1, allow_tf32: 1, deterministic: 1, ..Default::default() };
        spec.input_dims[..4].copy_from_slice(&input_dims);
        spec.input_strides = contiguous(&input_dims);
        spec.weight_dims[..4].copy_from_slice(&weight_dims);
        spec.weight_strides = contiguous(&weight_dims);
        spec.output_dims[..4].copy_from_slice(&output_dims);
        spec.output_strides = contiguous(&output_dims);
        spec.stride[..2].copy_from_slice(&[PATCH as i64, PATCH as i64]);
        spec.dilation[..2].copy_from_slice(&[1, 1]);
        self.conv.forward(&spec, pixels.dev(), self.patch.dev(), patches.dev(), s)?;
        let n = TOKENS * DIM;
        let x = Buffer::new(n * 2)?;
        check(unsafe { wan_clip_embed(p(patches.dev()), p(self.cls.dev()), p(self.pos.dev()), p(x.dev()), DIM as i64,
                                      (grid * grid) as i64, s) }, "embed")?;
        let (wide, normed, h) = (Buffer::new(n * 4)?, Buffer::new(n * 4)?, Buffer::new(n * 2)?);
        self.norm(x.dev(), &self.pre_norm, wide.dev(), normed.dev(), out)?;

        let qkv = Buffer::new(TOKENS * 3 * DIM * 2)?;
        let attn = Buffer::new(n * 2)?;
        let mlp = Buffer::new(TOKENS * MLP * 2)?;
        let mlp_act = Buffer::new(TOKENS * MLP * 2)?;
        let (t, d) = (TOKENS as i64, DIM as i64);
        let (qv, kv, vv) = Self::qkv_views();
        let prepared = sdpa::prepare(Io::F16, &qv, &kv, &vv)?;
        // [b, h, s, d] with the BSHD memory layout of q (the batch stride is immaterial at b = 1).
        if prepared.o_strides[1..] != [HEAD_DIM as i64, d, 1] {
            return Err(format!("unexpected SDPA output layout {:?}", prepared.o_strides).into());
        }
        let workspace = if prepared.workspace > 0 { Some(Buffer::new(prepared.workspace)?) } else { None };
        let scale = sdpa::default_scale(HEAD_DIM as i64);
        let lin = |x: u64, w: &[Buffer; 2], y: u64, nout: usize, k: usize| {
            self.linear.forward_at(x, w[0].dev(), Some(w[1].dev()), y, TOKENS, nout, k, Dtype::F16)
        };
        for b in &self.blocks {
            // x = x + attn(norm1(x)) ; x = x + mlp(norm2(x))   (x lives in `out`)
            self.norm(out, &b.norm1, wide.dev(), normed.dev(), h.dev())?;
            lin(h.dev(), &b.qkv, qkv.dev(), 3 * DIM, DIM)?;
            sdpa::forward(Io::F16, (qkv.dev(), &qv), (qkv.dev(), &kv), (qkv.dev(), &vv), attn.dev(), workspace.as_ref(), scale, s)?;
            lin(attn.dev(), &b.proj, h.dev(), DIM, DIM)?;
            check(unsafe { wan_add_f16(p(out), p(h.dev()), p(out), n as i64, s) }, "residual")?;
            self.norm(out, &b.norm2, wide.dev(), normed.dev(), h.dev())?;
            lin(h.dev(), &b.fc1, mlp.dev(), MLP, DIM)?;
            gelu_erf(mlp.dev(), mlp_act.dev(), 1, TOKENS * MLP, 0, 1, s)?;
            lin(mlp_act.dev(), &b.fc2, h.dev(), DIM, MLP)?;
            check(unsafe { wan_add_f16(p(out), p(h.dev()), p(out), n as i64, s) }, "residual")?;
        }
        self.device.sync()
    }
}
