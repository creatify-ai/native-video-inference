//! cuDNN scaled-dot-product attention as pinned PyTorch 2.13 builds it.
use crate::cuda::{Buffer, Stream};
use crate::Result;
use std::ffi::{c_char, c_int, c_void, CStr};

extern "C" {
    fn wan_sdpa_error() -> *const c_char;
    fn wan_sdpa_prepare(dtype: c_int, q: *const i64, qs: *const i64, k: *const i64, ks: *const i64, v: *const i64,
                        vs: *const i64, o_strides: *mut i64, workspace: *mut usize, plan: *mut c_char,
                        plan_bytes: usize) -> c_int;
    fn wan_sdpa_forward(dtype: c_int, q: *const i64, qs: *const i64, k: *const i64, ks: *const i64, v: *const i64,
                        vs: *const i64, qp: *const c_void, kp: *const c_void, vp: *const c_void, op: *mut c_void,
                        workspace: *mut c_void, workspace_bytes: usize, scale: f32, stream: Stream) -> c_int;
}

fn error() -> String { unsafe { CStr::from_ptr(wan_sdpa_error()) }.to_string_lossy().into_owned() }

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Io { Bf16 = 0, F16 = 1 }

/// One operand: [B,H,S,D] dims, element strides and a byte offset into its buffer.
#[derive(Clone, Copy, Debug)]
pub struct View { pub dims: [i64; 4], pub strides: [i64; 4], pub offset_bytes: usize }

pub struct Prepared { pub o_strides: [i64; 4], pub workspace: usize, pub plan: String }

pub fn prepare(io: Io, q: &View, k: &View, v: &View) -> Result<Prepared> {
    let (mut o_strides, mut workspace, mut plan) = ([0i64; 4], 0usize, vec![0 as c_char; 256]);
    let code = unsafe { wan_sdpa_prepare(io as c_int, q.dims.as_ptr(), q.strides.as_ptr(), k.dims.as_ptr(),
        k.strides.as_ptr(), v.dims.as_ptr(), v.strides.as_ptr(), o_strides.as_mut_ptr(), &mut workspace,
        plan.as_mut_ptr(), plan.len()) };
    if code != 0 { return Err(format!("cuDNN SDPA prepare: {}", error()).into()); }
    let plan = unsafe { CStr::from_ptr(plan.as_ptr()) }.to_string_lossy().into_owned();
    Ok(Prepared { o_strides, workspace, plan })
}

/// Default scale 1/sqrt(D) as sdp::calculate_scale: computed in double, then float.
pub fn default_scale(head_dim: i64) -> f32 { (1.0f64 / (head_dim as f64).sqrt()) as f32 }

#[allow(clippy::too_many_arguments)]
pub fn forward(io: Io, q: (u64, &View), k: (u64, &View), v: (u64, &View), out: u64,
               workspace: Option<&Buffer>, scale: f32, stream: Stream) -> Result<()> {
    let at = |base: u64, view: &View| (base + view.offset_bytes as u64) as *mut c_void;
    let (ws, ws_bytes) = workspace.map_or((std::ptr::null_mut(), 0), |w| (w.ptr(), w.bytes()));
    let code = unsafe { wan_sdpa_forward(io as c_int, q.1.dims.as_ptr(), q.1.strides.as_ptr(), k.1.dims.as_ptr(),
        k.1.strides.as_ptr(), v.1.dims.as_ptr(), v.1.strides.as_ptr(), at(q.0, q.1), at(k.0, k.1), at(v.0, v.1),
        out as *mut c_void, ws, ws_bytes, scale, stream) };
    if code != 0 { return Err(format!("cuDNN SDPA forward: {}", error()).into()); }
    Ok(())
}
