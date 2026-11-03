//! Small read-only reductions queued before marking; reuse existing download sync.
use std::{ffi::c_void,ptr};
use serde_json::{Value,json};
extern "C" {
    fn cudaMalloc(pointer:*mut *mut c_void,bytes:usize)->i32;
    fn cudaFree(pointer:*mut c_void)->i32;
    fn cudaMallocHost(pointer:*mut *mut c_void,bytes:usize)->i32;
    fn cudaFreeHost(pointer:*mut c_void)->i32;
    fn cudaMemcpyAsync(dst:*mut c_void,src:*const c_void,bytes:usize,kind:i32,stream:*mut c_void)->i32;
    fn h3_audio_health(input:*const f32,output:*mut f64,count:i64,stream:*mut c_void)->i32;
}
fn check(code:i32)->Result<(),String>{if code==0{Ok(())}else{Err(format!("audio health CUDA error {code}"))}}
const BYTES:usize=256*2*8;
pub struct Health{device:*mut c_void,host:*mut c_void,count:usize}
impl Health {
    pub fn new()->Result<Self,String>{
        let mut result=Self{device:ptr::null_mut(),host:ptr::null_mut(),count:0};
        check(unsafe{cudaMalloc(&mut result.device,BYTES)})?;
        check(unsafe{cudaMallocHost(&mut result.host,BYTES)})?;Ok(result)
    }
    pub fn enqueue(&mut self,input:*mut c_void,count:usize)->Result<(),String>{
        self.count=count;
        check(unsafe{h3_audio_health(input as _,self.device as _,count as _,ptr::null_mut())})?;
        check(unsafe{cudaMemcpyAsync(self.host,self.device,BYTES,2,ptr::null_mut())})
    }
    /// Caller has completed the existing output tensor download.
    pub fn collect(&self)->Value{
        let parts=unsafe{std::slice::from_raw_parts(self.host as *const f64,512)};
        let(sum,peak)=parts.chunks_exact(2).fold((0.0f64,0.0f64),|(s,p),x|(s+x[0],p.max(x[1])));
        json!({"samples":self.count,"peak_abs":peak,"rms":(sum/self.count as f64).sqrt()})
    }
}
impl Drop for Health{fn drop(&mut self){unsafe{
    if !self.device.is_null(){cudaFree(self.device);}
    if !self.host.is_null(){cudaFreeHost(self.host);}
}}}
