//! Linear layers through the core port of pinned Torch 2.13 addmm/mm.
use crate::cuda::{Buffer, Device, Stream};
use crate::Result;
use std::ffi::{c_char, c_int, c_void, CStr};

extern "C" {
    fn wan_linear_create(stream: Stream, blas_workspace: *mut c_void, blas_bytes: usize,
                         lt_workspace: *mut c_void, lt_bytes: usize, allow_tf32: c_int, allow_reduced: c_int,
                         out: *mut *mut c_void) -> c_int;
    fn infra_linear_destroy(handle: *mut c_void);
    fn infra_linear_error() -> *const c_char;
    fn infra_linear_forward(handle: *mut c_void, input: *const c_void, weight: *const c_void, bias: *const c_void,
                            output: *mut c_void, rows: c_int, out_features: c_int, in_features: c_int,
                            dtype: c_int) -> c_int;
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Dtype { Bf16 = 0, F32 = 1, F16 = 2 }

/// Torch 2.13 defaults on B200: TF32 matmul off, BF16 reduced-precision reduction
/// allowed; 32 MiB cuBLAS and 1 MiB cuBLASLt workspaces (as the H3 port).
pub struct Linear { handle: *mut c_void, _blas: Buffer, _lt: Buffer }
impl Linear {
    pub fn new(device: &Device) -> Result<Self> {
        let (blas, lt) = (Buffer::new(32 << 20)?, Buffer::new(1 << 20)?);
        let mut handle = std::ptr::null_mut();
        let code = unsafe { wan_linear_create(device.stream, blas.ptr(), 32 << 20, lt.ptr(), 1 << 20, 0, 1, &mut handle) };
        if code != 0 { return Err(format!("wan_linear_create: {}", error()).into()); }
        Ok(Self { handle, _blas: blas, _lt: lt })
    }
    /// Y[M,N] = X[M,K] W[N,K]^T (+ bias[N]), all contiguous row-major, one dtype.
    pub fn forward(&self, x: &Buffer, w: &Buffer, bias: Option<&Buffer>, y: &Buffer,
                   m: usize, n: usize, k: usize, dtype: Dtype) -> Result<()> {
        let width = if dtype == Dtype::F32 { 4 } else { 2 };
        if x.bytes() < m * k * width || w.bytes() < n * k * width || y.bytes() < m * n * width
            || bias.is_some_and(|b| b.bytes() < n * width) {
            return Err("linear operand smaller than its shape".into());
        }
        self.forward_at(x.dev(), w.dev(), bias.map(|b| b.dev()), y.dev(), m, n, k, dtype)
    }
    /// As `forward` on device addresses; the caller guarantees extents.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_at(&self, x: u64, w: u64, bias: Option<u64>, y: u64, m: usize, n: usize, k: usize,
                      dtype: Dtype) -> Result<()> {
        let bias = bias.map_or(std::ptr::null(), |b| b as *const c_void);
        let code = unsafe { infra_linear_forward(self.handle, x as *const c_void, w as *const c_void, bias,
                                                 y as *mut c_void, i32::try_from(m)?, i32::try_from(n)?,
                                                 i32::try_from(k)?, dtype as c_int) };
        if code != 0 { return Err(format!("linear [{m},{k}]x[{n},{k}]: {}", error()).into()); }
        Ok(())
    }
}
impl Drop for Linear {
    fn drop(&mut self) { unsafe { infra_linear_destroy(self.handle) } }
}

fn error() -> String { unsafe { CStr::from_ptr(infra_linear_error()) }.to_string_lossy().into_owned() }

