//! Convolutions through the core port of pinned PyTorch's cuDNN v8 path.
use crate::cuda::Stream;
use crate::Result;
use std::ffi::{c_char, c_int, c_void, CStr};

/// core_infra/kernels/conv.h InfraConvSpec.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct ConvSpec {
    pub ndim: i32, pub dtype: i32, pub transposed: i32, pub allow_tf32: i32, pub deterministic: i32, pub channels_last: i32,
    pub input_dims: [i64; 5], pub input_strides: [i64; 5],
    pub weight_dims: [i64; 5], pub weight_strides: [i64; 5],
    pub output_dims: [i64; 5], pub output_strides: [i64; 5],
    pub padding: [i64; 3], pub stride: [i64; 3], pub dilation: [i64; 3],
}

extern "C" {
    fn infra_conv_create(context: *mut *mut c_void, plan_limit: usize, workspace_limit: usize) -> c_int;
    fn infra_conv_set_oom_handler(handler: extern "C" fn() -> c_int);
    fn infra_conv_forward(context: *mut c_void, spec: *const ConvSpec, input: *const c_void, weight: *const c_void,
                          output: *mut c_void, stream: Stream) -> c_int;
    fn infra_conv_error() -> *const c_char;
    fn infra_conv_destroy(context: *mut c_void) -> c_int;
}

/// Contiguous strides of `dims[..n]`.
pub fn contiguous(dims: &[i64]) -> [i64; 5] {
    let mut strides = [0i64; 5];
    let mut acc = 1;
    for i in (0..dims.len()).rev() { strides[i] = acc; acc *= dims[i]; }
    strides
}

pub struct Conv { context: *mut c_void }
impl Conv {
    pub fn new(workspace_limit: usize) -> Result<Self> {
        // A workspace that does not fit first releases the allocator pool's cached blocks; it never
        // silently falls back to another engine (another algorithm, other numerics).
        unsafe { infra_conv_set_oom_handler(crate::cuda::release_cached_current_device) };
        let mut context = std::ptr::null_mut();
        if unsafe { infra_conv_create(&mut context, 64, workspace_limit) } != 0 { return Err(error().into()); }
        Ok(Self { context })
    }
    /// One convolution without bias (pinned PyTorch adds the bias as a separate op).
    pub fn forward(&self, spec: &ConvSpec, input: u64, weight: u64, output: u64, stream: Stream) -> Result<()> {
        let code = unsafe { infra_conv_forward(self.context, spec, input as *const c_void, weight as *const c_void,
                                               output as *mut c_void, stream) };
        if code != 0 { return Err(format!("cuDNN conv: {}", error()).into()); }
        Ok(())
    }
}
impl Drop for Conv {
    fn drop(&mut self) { unsafe { infra_conv_destroy(self.context); } }
}
fn error() -> String { unsafe { CStr::from_ptr(infra_conv_error()) }.to_string_lossy().into_owned() }
