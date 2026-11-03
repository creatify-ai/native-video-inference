//! Eager PyTorch elementwise operators of the DiT (kernels/eager.cu, core ops).
use crate::cuda::{check, Stream};
use crate::Result;
use std::ffi::{c_int, c_void};

extern "C" {
    fn wan_gelu_tanh_bf16(x: *const c_void, out: *mut c_void, n: i64, stream: Stream) -> c_int;
    fn wan_add_bf16_f32(a: *const c_void, b: *const c_void, out: *mut c_void, n: i64, stream: Stream) -> c_int;
    fn wan_add_bf16(a: *const c_void, b: *const c_void, out: *mut c_void, n: i64, stream: Stream) -> c_int;
    fn infra_cast(x: *const c_void, out: *mut c_void, n: i64, to_float: c_int, stream: Stream) -> c_int;
    fn wan_silu_f32(x: *const c_void, out: *mut c_void, n: i64, stream: Stream) -> c_int;
    fn wan_relu_bf16(x: *const c_void, out: *mut c_void, n: i64, stream: Stream) -> c_int;
    fn wan_bf16_to_f32(x: *const c_void, out: *mut c_void, n: i64, stream: Stream) -> c_int;
    fn wan_sinusoidal(position: *const c_void, out: *mut c_void, dim: c_int, stream: Stream) -> c_int;
    fn wan_add_channel_bias_bf16(x: *mut c_void, bias: *const c_void, channels: i64, inner: i64, stream: Stream) -> c_int;
    fn wan_layer_norm_f32(x: *const c_void, gamma: *const c_void, beta: *const c_void, out: *mut c_void, rows: i64,
                          cols: c_int, eps: f32, stream: Stream) -> c_int;
    fn wan_cat_latent_y(latent: *const c_void, latent_f32: c_int, y: *const c_void, out: *mut c_void, latent_n: i64,
                        y_n: i64, stream: Stream) -> c_int;
    fn wan_transpose_bf16(x: *const c_void, out: *mut c_void, channels: i64, tokens: i64, stream: Stream) -> c_int;
    fn wan_cat_f32_bf16(a: *const c_void, b: *const c_void, out: *mut c_void, a_n: i64, b_n: i64, stream: Stream) -> c_int;
}

fn p(a: u64) -> *mut c_void { a as *mut c_void }

/// nn.GELU(approximate="tanh") on bf16.
pub fn gelu_tanh_bf16(x: u64, out: u64, n: usize, stream: Stream) -> Result<()> {
    check(unsafe { wan_gelu_tanh_bf16(p(x), p(out), i64::try_from(n)?, stream) }, "gelu_tanh_bf16")
}
/// bf16 + bf16 -> bf16.
pub fn add_bf16(a: u64, b: u64, out: u64, n: usize, stream: Stream) -> Result<()> {
    check(unsafe { wan_add_bf16(p(a), p(b), p(out), i64::try_from(n)?, stream) }, "add_bf16")
}
/// bf16 + fp32 -> fp32 (type promotion).
pub fn add_bf16_f32(a: u64, b: u64, out: u64, n: usize, stream: Stream) -> Result<()> {
    check(unsafe { wan_add_bf16_f32(p(a), p(b), p(out), i64::try_from(n)?, stream) }, "add_bf16_f32")
}
/// fp32 -> bf16 (round to nearest even), as autocast's `.to(bfloat16)`.
pub fn f32_to_bf16(x: u64, out: u64, n: usize, stream: Stream) -> Result<()> {
    check(unsafe { infra_cast(p(x), p(out), i64::try_from(n)?, 0, stream) }, "cast f32->bf16")
}

/// nn.SiLU on fp32.
pub fn silu_f32(x: u64, out: u64, n: usize, stream: Stream) -> Result<()> {
    check(unsafe { wan_silu_f32(p(x), p(out), i64::try_from(n)?, stream) }, "silu_f32")
}
/// torch.relu (clamp_min 0) on bf16.
pub fn relu_bf16(x: u64, out: u64, n: usize, stream: Stream) -> Result<()> {
    check(unsafe { wan_relu_bf16(p(x), p(out), i64::try_from(n)?, stream) }, "relu_bf16")
}
/// bf16 -> fp32 (exact).
pub fn bf16_to_f32(x: u64, out: u64, n: usize, stream: Stream) -> Result<()> {
    check(unsafe { wan_bf16_to_f32(p(x), p(out), i64::try_from(n)?, stream) }, "bf16_to_f32")
}
/// sinusoidal_embedding_1d(dim, position) for one fp32 position.
pub fn sinusoidal(position: u64, out: u64, dim: usize, stream: Stream) -> Result<()> {
    check(unsafe { wan_sinusoidal(p(position), p(out), c_int::try_from(dim)?, stream) }, "sinusoidal")
}
/// In-place per-channel bias add on [channels, inner] bf16 (convolution bias).
pub fn add_channel_bias_bf16(x: u64, bias: u64, channels: usize, inner: usize, stream: Stream) -> Result<()> {
    check(unsafe { wan_add_channel_bias_bf16(p(x), p(bias), i64::try_from(channels)?, i64::try_from(inner)?, stream) },
          "add_channel_bias_bf16")
}
/// torch layer_norm (vectorized path) on fp32 rows with fp32 affine parameters.
pub fn layer_norm_f32(x: u64, gamma: u64, beta: u64, out: u64, rows: usize, cols: usize, eps: f32, stream: Stream) -> Result<()> {
    check(unsafe { wan_layer_norm_f32(p(x), p(gamma), p(beta), p(out), i64::try_from(rows)?, c_int::try_from(cols)?, eps,
                                      stream) }, "layer_norm_f32")
}
/// bf16(cat([latent, y])) with latent bf16 or fp32.
pub fn cat_latent_y(latent: u64, latent_f32: bool, y: u64, out: u64, latent_n: usize, y_n: usize, stream: Stream) -> Result<()> {
    check(unsafe { wan_cat_latent_y(p(latent), c_int::from(latent_f32), p(y), p(out), i64::try_from(latent_n)?,
                                    i64::try_from(y_n)?, stream) }, "cat_latent_y")
}
/// [channels, tokens] -> [tokens, channels] (bf16).
pub fn transpose_bf16(x: u64, out: u64, channels: usize, tokens: usize, stream: Stream) -> Result<()> {
    check(unsafe { wan_transpose_bf16(p(x), p(out), i64::try_from(channels)?, i64::try_from(tokens)?, stream) },
          "transpose_bf16")
}
/// cat([fp32 a, bf16 b]) -> fp32.
pub fn cat_f32_bf16(a: u64, b: u64, out: u64, a_n: usize, b_n: usize, stream: Stream) -> Result<()> {
    check(unsafe { wan_cat_f32_bf16(p(a), p(b), p(out), i64::try_from(a_n)?, i64::try_from(b_n)?, stream) }, "cat_f32_bf16")
}
