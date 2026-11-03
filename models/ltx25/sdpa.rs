//! cuDNN scaled-dot-product attention as pinned PyTorch 2.13 builds it.
use crate::cuda::{Buffer, Stream};
use crate::Result;
use std::ffi::{c_char, c_int, c_void, CStr};

extern "C" {
    fn ltx25_mem_eff_attention(q:*const c_void,k:*const c_void,v:*const c_void,out:*mut c_void,
        accum:*mut c_void,batch:i64,queries:i64,keys:i64,heads:i64,head_dim:i64,
        q_sb:i64,q_sm:i64,q_sh:i64,k_sb:i64,k_sm:i64,k_sh:i64,v_sb:i64,v_sm:i64,v_sh:i64,
        bias:*const c_void,bias_sb:i64,bias_sh:i64,bias_sm:i64,scale:f32,stream:Stream)->c_int;
    fn ltx25_sdpa_error() -> *const c_char;
    fn ltx25_sdpa_prepare(dtype: c_int, q: *const i64, qs: *const i64, k: *const i64, ks: *const i64, v: *const i64,
                        vs: *const i64, bias: *const i64, bias_s: *const i64, o_strides: *mut i64, workspace: *mut usize, plan: *mut c_char,
                        plan_bytes: usize) -> c_int;
    fn ltx25_sdpa_forward(dtype: c_int, q: *const i64, qs: *const i64, k: *const i64, ks: *const i64, v: *const i64,
                        vs: *const i64, bias_dims: *const i64, bias_strides: *const i64, bias: *const c_void, qp: *const c_void, kp: *const c_void, vp: *const c_void, op: *mut c_void,
                        workspace: *mut c_void, workspace_bytes: usize, scale: f32, stream: Stream) -> c_int;
}

/// PyTorch's aligned CUTLASS 32x128 gmem specialization for BF16 D=256/512.
/// The pinned wheel supplies CUDA headers only; this binary never links ATen.
pub fn efficient(q:&crate::tensor::Tensor,k:&crate::tensor::Tensor,v:&crate::tensor::Tensor,
    mask:Option<&crate::tensor::Tensor>,out:&crate::tensor::Tensor,scale:f32,device:&crate::cuda::Device)->Result<()>{
    use crate::{cuda::check,tensor::{self,Tensor}};
    if q.shape.len()!=4 || k.shape.len()!=4 || v.shape!=k.shape || q.shape[..2]!=k.shape[..2]
        || q.shape[3]!=k.shape[3] || ![256,512].contains(&q.shape[3])
        || [q,k,v,out].iter().any(|t|t.dtype!="torch.bfloat16" || t.strides[3]!=1){
        return Err("efficient SDPA specialization outside ported Gemma layout".into());
    }
    let (b,h,m,d)=(q.shape[0],q.shape[1],q.shape[2],q.shape[3]);
    if out.shape!=q.shape || out.strides!=vec![m*h*d,d,h*d,1]{
        return Err("efficient SDPA output must have BMHK storage".into());
    }
    let converted=if let Some(mask)=mask{
        if mask.dtype=="torch.bool"{
            let value=Tensor::contiguous_shape(mask.shape.clone(),q.dtype.clone(),device)?;
            tensor::elementwise(31,mask,None,None,&value,0.,1.,device.stream)?;Some(value)
        }else{None}
    }else{None};
    let mask=converted.as_ref().or(mask);
    let (bias,sb,sh,sm)=if let Some(mask)=mask{
        if mask.dtype!=q.dtype || mask.shape.len()!=4 || mask.strides[3]!=1
            || mask.shape[3]!=k.shape[2] || ![1,b].contains(&mask.shape[0])
            || ![1,h].contains(&mask.shape[1]) || ![1,m].contains(&mask.shape[2]){
            return Err("efficient SDPA mask layout outside port".into());
        }
        (mask.ptr(),if mask.shape[0]==1{0}else{mask.strides[0]},
            if mask.shape[1]==1{0}else{mask.strides[1]},if mask.shape[2]==1{0}else{mask.strides[2]})
    }else{(0,0,0,0)};
    let accum=Buffer::poisoned(out.elements()?*4,device)?;
    check(unsafe{ltx25_mem_eff_attention(q.ptr() as _,k.ptr() as _,v.ptr() as _,out.ptr() as _,accum.ptr(),
        b.try_into()?,m.try_into()?,k.shape[2].try_into()?,h.try_into()?,d.try_into()?,
        q.strides[0].try_into()?,q.strides[2].try_into()?,q.strides[1].try_into()?,
        k.strides[0].try_into()?,k.strides[2].try_into()?,k.strides[1].try_into()?,
        v.strides[0].try_into()?,v.strides[2].try_into()?,v.strides[1].try_into()?,
        bias as _,sb.try_into()?,sh.try_into()?,sm.try_into()?,scale,device.stream)},"CUTLASS efficient SDPA")
}

fn error() -> String { unsafe { CStr::from_ptr(ltx25_sdpa_error()) }.to_string_lossy().into_owned() }

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Io { Bf16 = 0, F16 = 1 }

/// One operand: [B,H,S,D] dims, element strides and a byte offset into its buffer.
#[derive(Clone, Copy, Debug)]
pub struct View { pub dims: [i64; 4], pub strides: [i64; 4], pub offset_bytes: usize }

pub struct Prepared { pub o_strides: [i64; 4], pub workspace: usize, pub plan: String }

pub fn prepare(io: Io, q: &View, k: &View, v: &View, bias:Option<&View>) -> Result<Prepared> {
    let _trace=native_core_infra::startup_trace::Span::new("backend_plan_prepare","sdpa",0);
    let (mut o_strides, mut workspace, mut plan) = ([0i64; 4], 0usize, vec![0 as c_char; 256]);
    let code = unsafe { ltx25_sdpa_prepare(io as c_int, q.dims.as_ptr(), q.strides.as_ptr(), k.dims.as_ptr(),
        k.strides.as_ptr(), v.dims.as_ptr(), v.strides.as_ptr(),
        bias.map_or(std::ptr::null(),|v|v.dims.as_ptr()),bias.map_or(std::ptr::null(),|v|v.strides.as_ptr()),o_strides.as_mut_ptr(), &mut workspace,
        plan.as_mut_ptr(), plan.len()) };
    if code != 0 { return Err(format!("cuDNN SDPA prepare: {}", error()).into()); }
    let plan = unsafe { CStr::from_ptr(plan.as_ptr()) }.to_string_lossy().into_owned();
    Ok(Prepared { o_strides, workspace, plan })
}

/// Default scale 1/sqrt(D) as sdp::calculate_scale: computed in double, then float.
pub fn default_scale(head_dim: i64) -> f32 { (1.0f64 / (head_dim as f64).sqrt()) as f32 }

#[allow(clippy::too_many_arguments)]
pub fn forward(io: Io, q: (u64, &View), k: (u64, &View), v: (u64, &View), out: u64,
               workspace: Option<&Buffer>, bias:Option<(u64,&View)>, scale: f32, stream: Stream) -> Result<()> {
    let at = |base: u64, view: &View| (base + view.offset_bytes as u64) as *mut c_void;
    let (ws, ws_bytes) = workspace.map_or((std::ptr::null_mut(), 0), |w| (w.ptr(), w.bytes()));
    let code = unsafe { ltx25_sdpa_forward(io as c_int, q.1.dims.as_ptr(), q.1.strides.as_ptr(), k.1.dims.as_ptr(),
        k.1.strides.as_ptr(), v.1.dims.as_ptr(), v.1.strides.as_ptr(),
        bias.map_or(std::ptr::null(),|v|v.1.dims.as_ptr()),bias.map_or(std::ptr::null(),|v|v.1.strides.as_ptr()),
        bias.map_or(std::ptr::null_mut(),|v|at(v.0,v.1)),at(q.0, q.1), at(k.0, k.1), at(v.0, v.1),
        out as *mut c_void, ws, ws_bytes, scale, stream) };
    if code != 0 { return Err(format!("cuDNN SDPA forward: {}", error()).into()); }
    Ok(())
}
