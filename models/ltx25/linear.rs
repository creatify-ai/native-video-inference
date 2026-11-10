use crate::{cuda::{Buffer, Device}, Result};
use std::ffi::{c_char, c_int, c_void, CStr};

extern "C" {
    fn ltx25_bmm_nn_f32(handle:*mut c_void,left:*const f32,right:*const f32,out:*mut f32,
        batch:c_int,rows:c_int,cols:c_int,inner:c_int)->c_int;
    fn infra_linear_create(stream: *mut c_void, blas: *mut c_void, blas_bytes: usize,
        lt: *mut c_void, lt_bytes: usize, tf32: c_int, reduced: c_int,
        out: *mut *mut c_void, policy: Option<extern "C" fn() -> c_int>) -> c_int;
    fn infra_linear_destroy(handle: *mut c_void);
    fn infra_linear_error() -> *const c_char;
    fn infra_linear_forward(handle: *mut c_void, x: *const c_void, w: *const c_void,
        bias: *const c_void, y: *mut c_void, m: c_int, n: c_int, k: c_int, dtype: c_int) -> c_int;
    fn ltx25_linear_column_major(handle:*mut c_void,x:*const c_void,w:*const c_void,y:*mut c_void,
        m:c_int,n:c_int,k:c_int,dtype:c_int)->c_int;
}

pub struct Linear { handle: *mut c_void, stream:crate::cuda::Stream, _blas: Buffer, _lt: Buffer }
impl Linear {
    pub fn bmm_nn(&self,left:&crate::tensor::Tensor,right:&crate::tensor::Tensor,out:&crate::tensor::Tensor)->Result<()>{
        use crate::tensor::contiguous_strides;
        if left.shape.len()!=2 || right.shape.len()!=3 || out.shape.len()!=3
            || left.shape[1]!=right.shape[1] || out.shape!=vec![right.shape[0],left.shape[0],right.shape[2]]
            || [left,right,out].iter().any(|t|t.dtype!="torch.float32" || t.strides!=contiguous_strides(&t.shape)){
            return Err("only contiguous float32 matrix-by-batched-matrix matmul is ported".into());
        }
        let status=unsafe{ltx25_bmm_nn_f32(self.handle,left.ptr() as _,right.ptr() as _,out.ptr() as _,
            right.shape[0].try_into()?,left.shape[0].try_into()?,right.shape[2].try_into()?,left.shape[1].try_into()?)};
        if status!=0{return Err(error().into());}Ok(())
    }
    pub fn new(device: &Device) -> Result<Self> {
        // The PyTorch reference runs with CUBLAS_WORKSPACE_CONFIG=:4096:8.
        // The same usable workspace size must reach the native handle.
        let (blas, lt) = (Buffer::new(32 << 20)?, Buffer::new(1 << 20)?);
        let mut handle = std::ptr::null_mut();
        let status = unsafe { infra_linear_create(device.stream, blas.ptr(), blas.bytes(),
            lt.ptr(), lt.bytes(), 0, 1, &mut handle, None) };
        if status != 0 { return Err(error().into()); }
        Ok(Self { handle, stream:device.stream, _blas: blas, _lt: lt })
    }
    #[allow(clippy::too_many_arguments)]
    pub fn forward(&self, x: u64, w: u64, bias: Option<u64>, y: u64,
                   m: usize, n: usize, k: usize, dtype: &str) -> Result<()> {
        let dtype = match dtype { "torch.bfloat16" => 0, "torch.float32" => 1,
            other => return Err(format!("linear dtype {other} is not ported").into()) };
        native_core_infra::execution_start::enqueue_if_armed(self.stream)?;
        let status = unsafe { infra_linear_forward(self.handle, x as _, w as _,
            bias.unwrap_or(0) as _, y as _, m.try_into()?, n.try_into()?, k.try_into()?, dtype) };
        if status != 0 { return Err(error().into()); }
        Ok(())
    }
    pub fn column_major(&self,x:u64,w:u64,y:u64,m:usize,n:usize,k:usize,dtype:&str)->Result<()>{
        let dtype=match dtype{"torch.bfloat16"=>0,"torch.float32"=>1,_=>return Err("column-major GEMM dtype".into())};
        native_core_infra::execution_start::enqueue_if_armed(self.stream)?;
        let status=unsafe{ltx25_linear_column_major(self.handle,x as _,w as _,y as _,
            m.try_into()?,n.try_into()?,k.try_into()?,dtype)};
        if status!=0{return Err(error().into());}Ok(())
    }
}
impl Drop for Linear { fn drop(&mut self) { unsafe { infra_linear_destroy(self.handle) } } }
fn error() -> String { unsafe { CStr::from_ptr(infra_linear_error()) }.to_string_lossy().into_owned() }
