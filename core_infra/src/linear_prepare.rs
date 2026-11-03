//! Model-declared GEMM metadata on the serving handle, before uploads start.
//! The shared C++ backend records and discards captures; it launches no graph
//! or inference and reads/writes no dummy tensor contents.
use std::ffi::{c_char,c_void,CStr};
#[repr(C)]
#[derive(Clone,Copy,Debug,PartialEq,Eq,PartialOrd,Ord)]
pub struct Shape {pub rows:i32,pub in_features:i32,pub out_features:i32,pub bias:i32}
unsafe extern "C" {
    fn infra_linear_prepare(handle:*mut c_void,shapes:*const Shape,count:usize,label:*const c_char)->i32;
    fn infra_linear_error()->*const c_char;
}
/// # Safety
/// The handle/context must be current, exclusively owned and idle. Its stream
/// cannot be used by another thread during these metadata-only captures.
pub unsafe fn prepare(handle:*mut c_void,shapes:&[Shape])->Result<(),String>{
    if handle.is_null() || shapes.iter().any(|s|s.rows<=0 || s.in_features<=0 || s.out_features<=0 || ![0,1].contains(&s.bias)){
        return Err("invalid linear metadata declaration".into());
    }
    if shapes.is_empty(){return Ok(());}
    let code=unsafe{infra_linear_prepare(handle,shapes.as_ptr(),shapes.len(),c"NATIVE_LINEAR_METADATA".as_ptr())};
    if code!=0{return Err(format!("linear preparation: {}",unsafe{CStr::from_ptr(infra_linear_error())}.to_string_lossy()));}
    Ok(())
}
