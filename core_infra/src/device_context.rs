//! A backend's thread-owned CUDA context, sharing the process's driver library.
//! Model runtimes may keep different user-space CUDA versions. Their device
//! drains must not accidentally wait on another backend's decoder stream.
use std::{ffi::{c_char,c_int,c_void,CStr},marker::PhantomData,ptr,rc::Rc,sync::OnceLock,thread::{self,ThreadId}};

type Result<T> = std::result::Result<T,Box<dyn std::error::Error>>;
unsafe extern "C" {
    fn dlopen(name:*const c_char,flags:c_int)->*mut c_void;
    fn dlsym(handle:*mut c_void,name:*const c_char)->*mut c_void;
    fn dlerror()->*const c_char;
}
#[derive(Clone,Copy)]
struct Driver {
    init:unsafe extern "C" fn(u32)->c_int,
    device:unsafe extern "C" fn(*mut c_int,c_int)->c_int,
    create:unsafe extern "C" fn(*mut *mut c_void,u32,c_int)->c_int,
    destroy:unsafe extern "C" fn(*mut c_void)->c_int,
    current:unsafe extern "C" fn(*mut *mut c_void)->c_int,
    set:unsafe extern "C" fn(*mut c_void)->c_int,
    uuid:unsafe extern "C" fn(*mut u8,c_int)->c_int,
    enable_peer:unsafe extern "C" fn(*mut c_void,u32)->c_int,
    copy_peer:unsafe extern "C" fn(u64,*mut c_void,u64,*mut c_void,usize,*mut c_void)->c_int,
    id:Option<unsafe extern "C" fn(*mut c_void,*mut u64)->c_int>,
}
fn driver()->Result<&'static Driver>{
    static DRIVER:OnceLock<std::result::Result<Driver,String>>=OnceLock::new();
    DRIVER.get_or_init(||unsafe{
        // One retained reference for the process lifetime. CUDA installs exit
        // handlers; unloading its driver before those handlers is unsafe.
        let handle=dlopen(c"libcuda.so.1".as_ptr(),2);
        if handle.is_null(){let error=dlerror();return Err(if error.is_null(){"CUDA driver unavailable".into()}
            else{CStr::from_ptr(error).to_string_lossy().into_owned()});}
        macro_rules! symbol {($name:literal,$ty:ty)=>{{
            let name=concat!($name,"\0");let value=dlsym(handle,name.as_ptr().cast());
            if value.is_null(){return Err(format!("CUDA driver lacks {}",$name));}
            std::mem::transmute::<*mut c_void,$ty>(value)
        }}}
        let id=dlsym(handle,c"cuCtxGetId".as_ptr());
        let id=if id.is_null(){None}else{Some(std::mem::transmute::<*mut c_void,unsafe extern "C" fn(*mut c_void,*mut u64)->c_int>(id))};
        Ok(Driver{id,
            init:symbol!("cuInit",unsafe extern "C" fn(u32)->c_int),
            device:symbol!("cuDeviceGet",unsafe extern "C" fn(*mut c_int,c_int)->c_int),
            create:symbol!("cuCtxCreate_v2",unsafe extern "C" fn(*mut *mut c_void,u32,c_int)->c_int),
            destroy:symbol!("cuCtxDestroy_v2",unsafe extern "C" fn(*mut c_void)->c_int),
            current:symbol!("cuCtxGetCurrent",unsafe extern "C" fn(*mut *mut c_void)->c_int),
            set:symbol!("cuCtxSetCurrent",unsafe extern "C" fn(*mut c_void)->c_int),
            uuid:symbol!("cuDeviceGetUuid_v2",unsafe extern "C" fn(*mut u8,c_int)->c_int),
            enable_peer:symbol!("cuCtxEnablePeerAccess",unsafe extern "C" fn(*mut c_void,u32)->c_int),
            copy_peer:symbol!("cuMemcpyPeerAsync",unsafe extern "C" fn(u64,*mut c_void,u64,*mut c_void,usize,*mut c_void)->c_int),
        })
    }).as_ref().map_err(|error|error.clone().into())
}
fn check(code:c_int,operation:&str)->Result<()>{
    if code==0{Ok(())}else{Err(format!("{operation}: CUDA driver error {code}").into())}
}

/// !Send/!Sync: create, use, synchronize and destroy on the backend's worker.
pub struct PrivateContext {
    handle:*mut c_void,previous:*mut c_void,owner:ThreadId,_thread_owned:PhantomData<Rc<()>>,
}
impl PrivateContext {
    pub fn create(ordinal:i32)->Result<Self>{
        let api=driver()?;
        check(unsafe{(api.init)(0)},"cuInit")?;
        let mut device=0;check(unsafe{(api.device)(&mut device,ordinal)},"cuDeviceGet")?;
        let mut previous=ptr::null_mut();check(unsafe{(api.current)(&mut previous)},"cuCtxGetCurrent")?;
        let mut handle=ptr::null_mut();check(unsafe{(api.create)(&mut handle,0,device)},"cuCtxCreate_v2")?;
        Ok(Self{handle,previous,owner:thread::current().id(),_thread_owned:PhantomData})
    }
    pub fn check_current(&self)->Result<()>{
        if self.owner!=thread::current().id(){return Err("CUDA backend used from another thread".into());}
        let mut current=ptr::null_mut();check(unsafe{(driver()?.current)(&mut current)},"cuCtxGetCurrent")?;
        if current!=self.handle{return Err("backend CUDA context changed".into());}Ok(())
    }
    /// This context's identity, for peer calls made on other threads.
    pub fn identity(&self)->usize{self.handle as usize}
    /// Lets this context reach the allocations of context `peer` directly, as
    /// `copy_peer` needs for a direct copy (already enabled is fine).
    pub fn enable_peer_access(&self,peer:usize)->Result<()>{
        self.check_current()?;
        let code=unsafe{(driver()?.enable_peer)(peer as *mut c_void,0)};
        if code==704{return Ok(());} // CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED
        check(code,"cuCtxEnablePeerAccess")
    }
}

/// Stream-ordered copy of `bytes` from `src` in context `source` to `dst` in
/// context `target`, between devices as well.
/// # Safety
/// Both ranges must lie inside live allocations of their contexts, and
/// `stream` must belong to the calling thread's current context.
pub unsafe fn copy_peer(dst:u64,target:usize,src:u64,source:usize,bytes:usize,stream:*mut c_void)->Result<()>{
    check(unsafe{(driver()?.copy_peer)(dst,target as *mut c_void,src,source as *mut c_void,bytes,stream)},"cuMemcpyPeerAsync")
}
impl Drop for PrivateContext {
    fn drop(&mut self){
        if let Ok(api)=driver(){unsafe{(api.destroy)(self.handle);(api.set)(self.previous);}}
    }
}
pub fn device_uuid(ordinal:i32)->Result<String>{
    let api=driver()?;check(unsafe{(api.init)(0)},"cuInit")?;
    let mut device=0;check(unsafe{(api.device)(&mut device,ordinal)},"cuDeviceGet")?;
    let mut uuid=[0u8;16];check(unsafe{(api.uuid)(uuid.as_mut_ptr(),device)},"cuDeviceGetUuid_v2")?;
    let hex=uuid.iter().map(|v|format!("{v:02x}")).collect::<String>();
    Ok(format!("GPU-{}-{}-{}-{}-{}",&hex[..8],&hex[8..12],&hex[12..16],&hex[16..20],&hex[20..]))
}

/// Identity of the calling thread's live context for context-owned resources.
pub fn current_identity()->Result<usize>{
    let mut current=ptr::null_mut();
    check(unsafe{(driver()?.current)(&mut current)},"cuCtxGetCurrent")?;
    if current.is_null(){return Err("CUDA context is not current".into());}
    Ok(current as usize)
}

/// Driver-assigned lifetime identity for caches that can survive a Session.
/// Recreated contexts can reuse their pointer address; their IDs stay distinct.
pub fn current_id()->Result<u64>{
    let context=current_identity()?as *mut c_void;let mut id=0u64;
    let function=driver()?.id.ok_or("CUDA driver does not expose context lifetime IDs")?;
    check(unsafe{function(context,&mut id)},"cuCtxGetId")?;
    Ok(id)
}

/// # Safety
/// The context's owner must join this thread before destroying the context.
pub unsafe fn bind_current(identity:usize)->Result<()> {
    if identity==0{return Err("cannot bind a null CUDA context".into());}
    check(unsafe{(driver()?.set)(identity as *mut c_void)},"cuCtxSetCurrent")
}

/// Restore a shared worker's previous context after one admitted callback.
pub struct CurrentContext {previous:*mut c_void,_thread_owned:PhantomData<Rc<()>>}
impl CurrentContext {
    /// # Safety
    /// The context owner must keep it alive until this guard and its callback
    /// have finished; the guard must stay on this thread.
    pub unsafe fn enter(identity:usize)->Result<Self>{
        let api=driver()?;
        let mut previous=ptr::null_mut();
        check(unsafe{(api.current)(&mut previous)},"cuCtxGetCurrent")?;
        unsafe{bind_current(identity)}?;
        Ok(Self{previous,_thread_owned:PhantomData})
    }
}
impl Drop for CurrentContext {fn drop(&mut self){if let Ok(api)=driver(){unsafe{(api.set)(self.previous);}}}}
