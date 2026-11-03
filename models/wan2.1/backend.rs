//! Standard ABI glue; request scheduling and copy ownership live in core_infra.
use native_core_infra::{backend_abi::{self,Bytes},backend_service::{Backend,Session}};
use serde_json::{json,Value};
fn open(options:Value)->crate::Result<Box<dyn Session>>{Ok(Box::new(crate::serving::Session::open(options)?))}
fn versions()->crate::Result<Value>{
    unsafe extern "C" {fn cudaRuntimeGetVersion(v:*mut i32)->i32;fn cublasGetProperty(p:i32,v:*mut i32)->i32;fn cudnnGetVersion()->usize;}
    let(mut runtime,mut major,mut minor)=(0,0,0);
    if unsafe{cudaRuntimeGetVersion(&mut runtime)}!=0 || unsafe{cublasGetProperty(0,&mut major)}!=0 || unsafe{cublasGetProperty(1,&mut minor)}!=0{
        return Err("Wan 2.1 runtime version query failed".into());
    }
    Ok(json!({"cuda_runtime":runtime,"cublas_major":major,"cublas_minor":minor,"cudnn":unsafe{cudnnGetVersion()}}))
}
static BACKEND:Backend=Backend::new(open,versions);
#[no_mangle]pub extern "C" fn native_backend_version_v1()->u32{backend_abi::VERSION}
/// # Safety
/// The caller follows native backend ABI v1.
#[no_mangle]pub unsafe extern "C" fn native_backend_call_v1(input:*const u8,len:usize,out:*mut Bytes)->i32{BACKEND.call(input,len,out)}
/// # Safety
/// Must be the allocation returned by this DSO's call function.
#[no_mangle]pub unsafe extern "C" fn native_backend_free_v1(data:*mut u8,len:usize){native_core_infra::backend_service::free(data,len)}
