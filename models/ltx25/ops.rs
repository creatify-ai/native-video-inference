use crate::{cuda::{check, Stream}, Result};
use std::ffi::{c_int, c_void};
extern "C" {
    fn ltx25_ancestral(x: *const c_void, denoised: *const c_void, noise: *const c_void,
        out: *mut c_void, n: i64, sigmas: *const f32, step: c_int, eta: f32, s_noise: f32,
        noise_bf16: c_int, stream: Stream) -> c_int;
    fn ltx25_gelu_bf16(x: *const c_void, y: *mut c_void, n: i64, stream: Stream) -> c_int;
}
#[allow(clippy::too_many_arguments)]
pub fn ancestral(x: u64, denoised: u64, noise: u64, out: u64, n: usize, sigmas: u64,
                  step: usize, eta: f32, s_noise: f32, noise_bf16: bool, stream: Stream) -> Result<()> {
    check(unsafe { ltx25_ancestral(x as _, denoised as _, noise as _, out as _, n.try_into()?,
        sigmas as _, step.try_into()?, eta, s_noise, noise_bf16.into(), stream) }, "ancestral Euler")
}
pub fn gelu(x: u64, y: u64, n: usize, stream: Stream) -> Result<()> {
    check(unsafe { ltx25_gelu_bf16(x as _, y as _, n.try_into()?, stream) }, "gelu tanh")
}
