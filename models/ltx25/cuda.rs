//! CUDA runtime boundary: one device, one non-blocking stream, owned allocations.
use crate::Result;
use std::ffi::{c_char, c_int, c_uint, c_void, CStr};

pub type Stream = *mut c_void;

extern "C" {
    fn cudaSetDevice(device: c_int) -> c_int;
    fn cudaMalloc(pointer: *mut *mut c_void, bytes: usize) -> c_int;
    fn cudaFree(pointer: *mut c_void) -> c_int;
    fn cudaMemcpyAsync(dst: *mut c_void, src: *const c_void, bytes: usize, kind: c_int, stream: Stream) -> c_int;
    fn cudaMemsetAsync(pointer: *mut c_void, value: c_int, bytes: usize, stream: Stream) -> c_int;
    fn cudaMemcpy2DAsync(dst: *mut c_void, dpitch: usize, src: *const c_void, spitch: usize, width: usize, height: usize,
                         kind: c_int, stream: Stream) -> c_int;
    fn cudaStreamCreateWithFlags(stream: *mut Stream, flags: c_uint) -> c_int;
    fn cudaStreamSynchronize(stream: Stream) -> c_int;
    fn cudaStreamDestroy(stream: Stream) -> c_int;
    fn cudaDeviceGetAttribute(value: *mut c_int, attribute: c_int, device: c_int) -> c_int;
    fn cudaGetErrorString(code: c_int) -> *const c_char;
    fn cudaDeviceEnablePeerAccess(peer: c_int, flags: c_uint) -> c_int;
    fn cudaMemcpyPeerAsync(dst: *mut c_void, dst_device: c_int, src: *const c_void, src_device: c_int, bytes: usize,
                           stream: Stream) -> c_int;
    fn cudaGetLastError() -> c_int;
    fn cudaMemGetInfo(free: *mut usize, total: *mut usize) -> c_int;
    fn cudaStreamBeginCapture(stream: Stream, mode: c_int) -> c_int;
    fn cudaStreamEndCapture(stream: Stream, graph: *mut *mut c_void) -> c_int;
    fn cudaGraphInstantiate(exec: *mut *mut c_void, graph: *mut c_void, flags: u64) -> c_int;
    fn cudaGraphLaunch(exec: *mut c_void, stream: Stream) -> c_int;
    fn cudaGraphExecDestroy(exec: *mut c_void) -> c_int;
    fn cudaGraphDestroy(graph: *mut c_void) -> c_int;
}

/// Enables access from the calling thread's device to `peer`'s memory (already enabled is fine).
pub fn enable_peer_access(peer: i32) -> Result<()> {
    let code = unsafe { cudaDeviceEnablePeerAccess(peer, 0) };
    if code == 704 { unsafe { cudaGetLastError() }; return Ok(()); } // cudaErrorPeerAccessAlreadyEnabled
    check(code, "cudaDeviceEnablePeerAccess")
}

/// Stream-ordered copy between two devices' allocations.
/// # Safety
/// Both ranges must lie inside live allocations on the named devices.
pub unsafe fn memcpy_peer(dst: u64, dst_device: i32, src: u64, src_device: i32, bytes: usize, stream: Stream) -> c_int {
    cudaMemcpyPeerAsync(dst as *mut c_void, dst_device, src as *const c_void, src_device, bytes, stream)
}

const HOST_TO_DEVICE: c_int = 1;
const DEVICE_TO_HOST: c_int = 2;
const DEVICE_TO_DEVICE: c_int = 3;

/// Stream-ordered device-to-device copy between raw addresses.
/// # Safety
/// Both ranges must lie inside live device allocations.
pub unsafe fn memcpy_d2d(dst: u64, src: u64, bytes: usize, stream: Stream) -> c_int {
    cudaMemcpyAsync(dst as *mut c_void, src as *const c_void, bytes, DEVICE_TO_DEVICE, stream)
}

/// Stream-ordered pitched device-to-device copy of `height` rows of `width` bytes.
/// # Safety
/// Both pitched ranges must lie inside live device allocations.
pub unsafe fn memcpy2d_d2d(dst: u64, dpitch: usize, src: u64, spitch: usize, width: usize, height: usize,
                           stream: Stream) -> c_int {
    cudaMemcpy2DAsync(dst as *mut c_void, dpitch, src as *const c_void, spitch, width, height, DEVICE_TO_DEVICE, stream)
}

/// Uploads `data` into the start of an existing allocation (synchronous).
pub fn upload_into(buffer: &Buffer, data: &[u8], device: &Device) -> Result<()> {
    if data.len() > buffer.bytes() { return Err("upload exceeds the allocation".into()); }
    check(unsafe { cudaMemcpyAsync(buffer.ptr(), data.as_ptr().cast(), data.len(), HOST_TO_DEVICE, device.stream) }, "upload")?;
    device.sync()
}

pub fn check(code: c_int, what: &str) -> Result<()> {
    if code == 0 { return Ok(()); }
    let text = unsafe { CStr::from_ptr(cudaGetErrorString(code)) }.to_string_lossy().into_owned();
    Err(format!("{what}: CUDA error {code}: {text}").into())
}

pub struct Device { pub stream: Stream, pub index: i32,
    weights:std::cell::RefCell<std::collections::BTreeMap<String,crate::tensor::Tensor>>,
    uploads:std::cell::RefCell<Vec<native_core_infra::pinned_weights::CopyLease>>,
    asset_uploads:std::cell::Cell<(usize,usize)>,
    pub(crate) host_assets:Option<std::rc::Rc<native_core_infra::immutable_assets::ImmutableAssets>>,
    kernels:std::cell::RefCell<std::collections::BTreeMap<(std::path::PathBuf,String),std::rc::Rc<crate::triton::Kernel>>>,
    pool:std::cell::RefCell<std::rc::Rc<AllocationPool>>,capturing:std::cell::Cell<bool>,
    /// Bumped whenever cached weights may leave the device (a capture addressing them is then stale).
    pub(crate) weight_generation:std::cell::Cell<u64>,
    context:Option<native_core_infra::device_context::PrivateContext> }
impl Device {
    /// Device 0 must be a B200 (sm_100); the gold kernels are sm_100a binaries.
    pub fn open() -> Result<Self> { Self::open_index(0) }

    /// Makes device `index` current for the calling thread and opens a stream on it.
    pub fn open_index(index: i32) -> Result<Self> {
        check(unsafe { cudaSetDevice(index) }, "cudaSetDevice")?;
        Self::from_current(index,None)
    }
    /// The worker owns its context for its lifetime. Do not call cudaSetDevice
    /// here: preserve the driver context selected before entering this runtime.
    pub fn open_private(index:i32)->Result<Self>{
        let context=native_core_infra::device_context::PrivateContext::create(index)?;
        Self::from_current(index,Some(context))
    }
    fn from_current(index:i32,context:Option<native_core_infra::device_context::PrivateContext>)->Result<Self>{
        let (mut major, mut minor) = (0, 0);
        check(unsafe { cudaDeviceGetAttribute(&mut major, 75, index) }, "compute capability")?;
        check(unsafe { cudaDeviceGetAttribute(&mut minor, 76, index) }, "compute capability")?;
        if (major, minor) != (10, 0) { return Err(format!("expected sm_100 B200, found sm_{major}{minor}").into()); }
        let mut stream = std::ptr::null_mut();
        check(unsafe { cudaStreamCreateWithFlags(&mut stream, 1) }, "cudaStreamCreate")?;
        if let Some(context)=&context{context.check_current()?;}
        Ok(Self { stream, index,weights:std::cell::RefCell::new(std::collections::BTreeMap::new()),
            uploads:std::cell::RefCell::new(Vec::new()),asset_uploads:std::cell::Cell::new((0,0)),host_assets:None,
            kernels:std::cell::RefCell::new(std::collections::BTreeMap::new()),
            pool:std::cell::RefCell::new(AllocationPool::new(TRANSIENT_POOL_BYTES,1<<16)),
            capturing:std::cell::Cell::new(false),weight_generation:std::cell::Cell::new(0),context })
    }
    pub fn sync(&self) -> Result<()> {
        if let Some(context)=&self.context{context.check_current()?;}
        // Captured work only runs when its graph launches; there is nothing to wait for yet.
        if self.capturing.get(){return Ok(());}
        check(unsafe { cudaStreamSynchronize(self.stream) }, "cudaStreamSynchronize")?;
        for lease in self.uploads.borrow().iter(){lease.wait()?;}
        self.uploads.borrow_mut().clear();Ok(())
    }
    pub(crate) fn track_upload(&self,lease:native_core_infra::pinned_weights::CopyLease){self.uploads.borrow_mut().push(lease);}
    pub(crate) fn cached_weight(&self,key:&str)->Option<crate::tensor::Tensor>{self.weights.borrow().get(key).cloned()}
    pub(crate) fn cached_weight_tensors(&self)->usize{self.weights.borrow().len()}
    pub(crate) fn cache_weight(&self,key:String,value:&crate::tensor::Tensor){self.weights.borrow_mut().insert(key,value.clone());}
    pub(crate) fn record_asset_upload(&self,bytes:usize){let(n,b)=self.asset_uploads.get();self.asset_uploads.set((n+1,b+bytes));}
    pub(crate) fn asset_upload_counts(&self)->(usize,usize){self.asset_uploads.get()}
    pub fn clear_weight_cache(&self)->Result<()>{
        self.sync()?;self.weights.borrow_mut().clear();self.weight_generation.set(self.weight_generation.get()+1);Ok(())
    }
    pub(crate) fn evict_checkpoint(&self,path:&std::path::Path)->Result<()>{
        self.sync()?;let prefix=format!("{}\0",path.display());
        self.weights.borrow_mut().retain(|key,_|!key.starts_with(&prefix));
        self.weight_generation.set(self.weight_generation.get()+1);
        self.trim_pool();Ok(())
    }
    /// Frees every cached transient (the stream must be idle: callers synchronize first).
    fn trim_pool(&self){for pointer in self.pool.borrow().free.borrow_mut().drain(){unsafe{cudaFree(pointer);}}}
    pub fn uuid(&self)->Result<String>{native_core_infra::device_context::device_uuid(self.index)}
    /// This device's private context, for peer copies issued on other threads.
    pub(crate) fn context_identity(&self)->Result<usize>{
        Ok(self.context.as_ref().ok_or("peer copies need a private device context")?.identity())
    }
    /// Lets this device reach the allocations of the private context `peer` directly.
    pub(crate) fn enable_peer(&self,peer:usize)->Result<()>{
        self.context.as_ref().ok_or("peer access needs a private device context")?.enable_peer_access(peer)
    }
    pub fn available_bytes(&self)->Result<usize>{
        let(mut free,mut total)=(0,0);
        check(unsafe{cudaMemGetInfo(&mut free,&mut total)},"cudaMemGetInfo")?;Ok(free)
    }
    pub(crate) fn asset(&self,path:&std::path::Path,sha:&str,bytes:usize)->Result<std::rc::Rc<Vec<u8>>>{
        if let Some(cache)=&self.host_assets{return cache.load(path,sha,bytes);}
        native_core_infra::immutable_assets::ImmutableAssets::new(bytes,1).load(path,sha,bytes)
    }
    pub(crate) fn kernel(&self,root:&std::path::Path,sha:&str)->Result<std::rc::Rc<crate::triton::Kernel>>{
        if self.context.is_none(){return Ok(std::rc::Rc::new(crate::triton::Kernel::load(root,sha)?));}
        let key=(root.to_path_buf(),sha.to_owned());
        if let Some(value)=self.kernels.borrow().get(&key){return Ok(value.clone());}
        if self.kernels.borrow().len()>=4096{return Err("resident native kernel cache bound exceeded".into());}
        let value=std::rc::Rc::new(crate::triton::Kernel::load(root,sha)?);
        self.kernels.borrow_mut().insert(key,value.clone());Ok(value)
    }
    pub(crate) fn kernel_count(&self)->usize{self.kernels.borrow().len()}
    pub(crate) fn pool_bytes(&self)->usize{self.pool.borrow().free.borrow().bytes()}
    pub(crate) fn transient(&self,bytes:usize)->Result<Buffer>{
        if self.context.is_none(){return Buffer::new(bytes);}
        // All users run on this device's one private, non-blocking stream.
        // Reuse does not retain tensor contents: poisoned() reinitializes the
        // entire allocation on that same stream before its next use.
        let pool=self.pool.borrow().clone();
        let ptr=pool.free.borrow_mut().take(bytes.max(1));
        let mut value=match ptr{
            Some(ptr)=>Buffer{ptr,bytes,pool:None},
            // Cached blocks of other sizes may hold the memory: return them and retry once.
            None=>match Buffer::new(bytes){
                Ok(value)=>value,
                Err(_) if pool.free.borrow().entries()>0 && !self.capturing.get()=>{
                    unsafe{cudaGetLastError();}  // the failed cudaMalloc's error is not a later launch's
                    self.sync()?;self.trim_pool();Buffer::new(bytes)?
                }
                Err(e)=>return Err(e),
            },
        };
        value.pool=Some(std::rc::Rc::downgrade(&pool));Ok(value)
    }
    /// Records `body`'s work on this device's stream as a CUDA graph instead of running it. Its
    /// transients come from a pool of their own that the graph keeps: the graph's addresses are never
    /// handed to other work, and its tensors may be reused only inside it (stream order, as eager).
    /// Synchronizations inside `body` are skipped (nothing has run); downloads or uploads fail.
    pub(crate) fn capture<T>(&self,body:impl FnOnce()->Result<T>)->Result<(T,Graph)>{
        if self.capturing.get(){return Err("nested CUDA graph capture".into());}
        self.sync()?;
        let own=AllocationPool::new(usize::MAX/2,1<<20);
        let previous=self.pool.replace(own.clone());
        self.capturing.set(true);
        let begun=check(unsafe{cudaStreamBeginCapture(self.stream,2)},"cudaStreamBeginCapture");  // relaxed: pool cudaMalloc stays allowed
        let result=if begun.is_ok(){body()}else{Err("capture did not begin".into())};
        let mut graph=std::ptr::null_mut();
        let ended=if begun.is_ok(){check(unsafe{cudaStreamEndCapture(self.stream,&mut graph)},"cudaStreamEndCapture")}else{begun};
        self.capturing.set(false);
        self.pool.replace(previous);
        let value=result?;ended?;
        let mut exec=std::ptr::null_mut();
        let instantiated=check(unsafe{cudaGraphInstantiate(&mut exec,graph,0)},"cudaGraphInstantiate");
        unsafe{cudaGraphDestroy(graph);}
        instantiated?;
        Ok((value,Graph{exec,stream:self.stream,_pool:own}))
    }
}
/// An instantiated capture and the transients its kernels address.
pub(crate) struct Graph {exec:*mut c_void,stream:Stream,_pool:std::rc::Rc<AllocationPool>}
impl Graph {
    pub(crate) fn launch(&self)->Result<()>{check(unsafe{cudaGraphLaunch(self.exec,self.stream)},"cudaGraphLaunch")}
}
impl Drop for Graph {
    fn drop(&mut self){unsafe{cudaStreamSynchronize(self.stream);cudaGraphExecDestroy(self.exec);}}
}
impl Drop for Device {
    fn drop(&mut self) { unsafe { cudaStreamSynchronize(self.stream); }self.weights.get_mut().clear();
        self.uploads.get_mut().clear();unsafe{cudaStreamDestroy(self.stream);} }
}

/// Freed transients kept for reuse by exact size. A sampling forward allocates the same sizes every
/// time, so after the first forward almost every allocation is a reuse (no cudaMalloc, and no
/// cudaFree, which synchronizes the device). Trimmed when a checkpoint is evicted.
/// The cap holds the decoder's multi-GB temporaries across its components (at 16 GiB they were
/// freed and reallocated each time); a failed cudaMalloc trims the pool and retries.
pub(crate) const TRANSIENT_POOL_BYTES:usize=96<<30;
/// Owned device bytes. Freed after a device synchronization by the owner.
struct AllocationPool {free:std::cell::RefCell<native_core_infra::reuse_pool::ReusePool<*mut c_void>>}
impl AllocationPool {
    fn new(bytes:usize,entries:usize)->std::rc::Rc<Self>{
        std::rc::Rc::new(Self{free:std::cell::RefCell::new(native_core_infra::reuse_pool::ReusePool::new(bytes,bytes,entries))})
    }
}
impl Drop for AllocationPool {
    fn drop(&mut self){for pointer in self.free.get_mut().drain(){unsafe{cudaFree(pointer);}}}
}
pub struct Buffer { ptr: *mut c_void, bytes: usize,pool:Option<std::rc::Weak<AllocationPool>> }
impl Buffer {
    /// # Safety
    /// Takes exclusive ownership of a drained framework CUDA allocation.
    pub(crate) unsafe fn from_allocation(ptr:*mut c_void,bytes:usize)->Self {Self{ptr,bytes,pool:None}}
    pub fn new(bytes: usize) -> Result<Self> {
        let mut ptr = std::ptr::null_mut();
        check(unsafe { cudaMalloc(&mut ptr, bytes.max(1)) }, "cudaMalloc")?;
        Ok(Self { ptr, bytes,pool:None })
    }
    /// Filled with 0xFF so an unwritten output can never match a reference.
    pub fn poisoned(bytes: usize, device: &Device) -> Result<Self> {
        let buffer = device.transient(bytes)?;
        check(unsafe { cudaMemsetAsync(buffer.ptr, 0xFF, bytes, device.stream) }, "cudaMemset")?;
        Ok(buffer)
    }
    pub fn upload(data: &[u8], device: &Device) -> Result<Self> {
        let buffer = Self::new(data.len())?;
        check(unsafe { cudaMemcpyAsync(buffer.ptr, data.as_ptr().cast(), data.len(), HOST_TO_DEVICE, device.stream) },
              "upload")?;
        device.sync()?;
        Ok(buffer)
    }
    pub fn download(&self, device: &Device) -> Result<Vec<u8>> {
        let mut out = vec![0u8; self.bytes];
        device.sync()?;
        check(unsafe { cudaMemcpyAsync(out.as_mut_ptr().cast(), self.ptr, self.bytes, DEVICE_TO_HOST, device.stream) },
              "download")?;
        device.sync()?;
        Ok(out)
    }
    pub fn copy_from(&self, other: &Buffer, bytes: usize, device: &Device) -> Result<()> {
        if bytes > self.bytes || bytes > other.bytes { return Err("device copy out of range".into()); }
        check(unsafe { cudaMemcpyAsync(self.ptr, other.ptr, bytes, DEVICE_TO_DEVICE, device.stream) }, "d2d copy")
    }
    /// Zero-fills the allocation (stream ordered).
    pub fn zero(&self, device: &Device) -> Result<()> {
        check(unsafe { cudaMemsetAsync(self.ptr, 0, self.bytes, device.stream) }, "cudaMemset")
    }
    /// Copies `bytes` from a device address into this allocation (stream ordered).
    pub fn copy_from_device(&self, src: u64, bytes: usize, device: &Device) -> Result<()> {
        if bytes > self.bytes { return Err("device copy out of range".into()); }
        check(unsafe { cudaMemcpyAsync(self.ptr, src as *const c_void, bytes, DEVICE_TO_DEVICE, device.stream) }, "d2d copy")
    }
    pub fn ptr(&self) -> *mut c_void { self.ptr }
    /// Device address, for operators that take raw (offset) addresses.
    pub fn dev(&self) -> u64 { self.ptr as u64 }
    /// Device address `offset_bytes` into this allocation.
    pub fn at(&self, offset_bytes: usize) -> u64 {
        assert!(offset_bytes <= self.bytes, "device offset past the allocation");
        self.ptr as u64 + offset_bytes as u64
    }
    pub fn bytes(&self) -> usize { self.bytes }
}
impl Drop for Buffer {
    fn drop(&mut self) {
        if let Some(pool)=self.pool.as_ref().and_then(std::rc::Weak::upgrade){
            if pool.free.borrow_mut().put(self.bytes.max(1),self.ptr).is_none(){return;}
        }
        unsafe { cudaFree(self.ptr); }
    }
}
