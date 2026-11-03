//! Shared encoder operators (T5, CLIP, VAE): torch reductions, softmax, bmm and erf GELU
//! from kernels/encoders.cu, softmax.cu and post.cu (pinned PyTorch 2.13 device code).
use crate::cuda::{check, Buffer, Device, Stream};
use crate::Result;
use std::ffi::{c_int, c_void};

extern "C" {
    fn wan_softmax_f32(x: *const c_void, y: *mut c_void, rows: i64, cols: i64, stream: Stream) -> c_int;
    fn wan_gelu_erf(x: *const c_void, y: *mut c_void, rows: i64, cols: i64, in_stride: i64, dtype: c_int,
                    stream: Stream) -> c_int;
    fn wan_bmm(handle: *mut c_void, dtype: c_int, b1: *const c_void, s1: *const i64, b2: *const c_void, s2: *const i64,
               out: *mut c_void, batch: i64, m: i64, k: i64, n: i64) -> c_int;
    fn wan_blas_create(handle: *mut *mut c_void, workspace: *mut c_void, bytes: usize, stream: Stream) -> c_int;
    fn wan_blas_destroy(handle: *mut c_void);
    fn wan_reduce_f32(op: c_int, x: *const c_void, out: *mut c_void, ndim: c_int, sizes: *const i64, strides: *const i64,
                      reduce_mask: u32, stream: Stream) -> c_int;
}
fn p(a: u64) -> *mut c_void { a as *mut c_void }

pub struct Blas { pub handle: *mut c_void, _workspace: Buffer }
impl Blas {
    pub fn new(device: &Device) -> Result<Self> {
        let workspace = Buffer::new(32 << 20)?;
        let mut handle = std::ptr::null_mut();
        check(unsafe { wan_blas_create(&mut handle, workspace.ptr(), workspace.bytes(), device.stream) }, "cublas handle")?;
        Ok(Self { handle, _workspace: workspace })
    }
    /// at::bmm(b1 [B, M, K] strided, b2 [B, K, N] strided) -> contiguous [B, M, N].
    #[allow(clippy::too_many_arguments)]
    pub fn bmm(&self, bf16: bool, b1: u64, s1: [i64; 3], b2: u64, s2: [i64; 3], out: u64, batch: usize, m: usize, k: usize,
               n: usize) -> Result<()> {
        let code = unsafe { wan_bmm(self.handle, if bf16 { 2 } else { 0 }, p(b1), s1.as_ptr(), p(b2), s2.as_ptr(), p(out),
                                    batch as i64, m as i64, k as i64, n as i64) };
        if code != 0 { return Err(format!("bmm [{batch},{m},{k}]x[{k},{n}]: cuBLAS status {code}").into()); }
        Ok(())
    }
}
impl Drop for Blas {
    fn drop(&mut self) { unsafe { wan_blas_destroy(self.handle) } }
}

pub fn softmax_f32(x: u64, y: u64, rows: usize, cols: usize, stream: Stream) -> Result<()> {
    check(unsafe { wan_softmax_f32(p(x), p(y), rows as i64, cols as i64, stream) }, "softmax_f32")
}
/// GELU (erf) of `rows` rows of `cols` read with row stride `in_stride`; dtype 0 fp32, 1 fp16, 2 bf16.
pub fn gelu_erf(x: u64, y: u64, rows: usize, cols: usize, in_stride: usize, dtype: i32, stream: Stream) -> Result<()> {
    check(unsafe { wan_gelu_erf(p(x), p(y), rows as i64, cols as i64, in_stride as i64, dtype, stream) }, "gelu_erf")
}
/// torch mean over the dims set in `mask` of a strided fp32 tensor (sizes/strides in elements).
pub fn mean_f32(x: u64, out: u64, sizes: &[i64], strides: &[i64], mask: u32, stream: Stream) -> Result<()> {
    assert_eq!(sizes.len(), strides.len());
    check(unsafe { wan_reduce_f32(0, p(x), p(out), sizes.len() as c_int, sizes.as_ptr(), strides.as_ptr(), mask, stream) },
          "reduce_f32")
}
