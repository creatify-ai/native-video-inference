//! NATTEN's pinned noncausal cutlass-fna backend, instantiated without ATen.
use crate::{cuda::{check,Buffer,Device,Stream},tensor::{Tensor,contiguous_strides},Result};
use std::ffi::{c_int,c_void};
extern "C" {
    fn ltx25_natten3d(q:*const c_void,k:*const c_void,v:*const c_void,out:*mut c_void,accum:*mut c_void,lse:*mut c_void,
        batch:c_int,t:c_int,h:c_int,w:c_int,heads:c_int,dim:c_int,window:*const c_int,qt:*const c_int,kt:*const c_int,
        scale:f32,stream:Stream)->c_int;
}
pub fn forward(q:&Tensor,k:&Tensor,v:&Tensor,out:&Tensor,window:&[usize],qt:&[usize],kt:&[usize],scale:f32,device:&Device)->Result<()>{
    if q.shape.len()!=6 || q.shape!=k.shape || q.shape!=v.shape || q.shape!=out.shape || q.shape[5]!=64
        || [q,k,v,out].iter().any(|t|t.dtype!="torch.bfloat16") || [window,qt,kt].iter().any(|v|v.len()!=3)
        || out.strides!=contiguous_strides(&out.shape){
        return Err("NATTEN outside the frozen BF16 head64 3D cutlass-fna contract".into());
    }
    let materialize=|t:&Tensor|->Result<Tensor>{if t.strides==contiguous_strides(&t.shape){Ok(t.clone())}else{t.contiguous(device)}};
    let (q,k,v)=(materialize(q)?,materialize(k)?,materialize(v)?);
    let accum=Buffer::poisoned(out.elements()?*4,device)?;
    let lse=Buffer::poisoned(out.elements()?/64*4,device)?;
    let ints=|v:&[usize]|->Result<Vec<c_int>>{v.iter().map(|&v|v.try_into().map_err(Into::into)).collect()};
    let shape=ints(&q.shape)?;let window=ints(window)?;let qt=ints(qt)?;let kt=ints(kt)?;
    check(unsafe{ltx25_natten3d(q.ptr() as _,k.ptr() as _,v.ptr() as _,out.ptr() as _,accum.ptr(),lse.ptr(),
        shape[0],shape[1],shape[2],shape[3],shape[4],shape[5],window.as_ptr(),qt.as_ptr(),kt.as_ptr(),scale,device.stream)},
        "NATTEN cutlass-fna")
}
