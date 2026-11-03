//! CUDA runtime boundary: one device, one non-blocking stream, owned allocations.
use crate::Result;
use std::ffi::{c_char, c_int, c_uint, c_void, CStr};

pub type Stream = *mut c_void;

extern "C" {
    fn cudaSetDevice(device: c_int) -> c_int;
    fn cudaGetDevice(device: *mut c_int) -> c_int;
    fn cudaDeviceSynchronize() -> c_int;
    fn cudaEventCreateWithFlags(event: *mut *mut c_void, flags: c_uint) -> c_int;
    fn cudaEventRecord(event: *mut c_void, stream: Stream) -> c_int;
    fn cudaStreamWaitEvent(stream: Stream, event: *mut c_void, flags: c_uint) -> c_int;
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
    fn cudaMemGetInfo(free:*mut usize,total:*mut usize)->c_int;
    fn cudaGetErrorString(code: c_int) -> *const c_char;
    fn cudaDeviceEnablePeerAccess(peer: c_int, flags: c_uint) -> c_int;
    fn cudaMemcpyPeerAsync(dst: *mut c_void, dst_device: c_int, src: *const c_void, src_device: c_int, bytes: usize,
                           stream: Stream) -> c_int;
    fn cudaGetLastError() -> c_int;
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
/// A CUDA event (timing disabled) on the current device, for cross-device stream ordering.
/// Held as an address so it can cross threads; never destroyed (a handful per process).
#[derive(Clone, Copy, Debug)]
pub struct Event(pub usize);
impl Event {
    pub fn new() -> Result<Self> {
        let mut event = std::ptr::null_mut();
        check(unsafe { cudaEventCreateWithFlags(&mut event, 2) }, "cudaEventCreate")?;  // cudaEventDisableTiming
        Ok(Self(event as usize))
    }
    /// Records the event at the current end of `stream`.
    pub fn record(self, stream: Stream) -> Result<()> { check(unsafe { cudaEventRecord(self.0 as *mut c_void, stream) }, "cudaEventRecord") }
    /// Makes later work on `stream` wait for the most recent record (any device).
    pub fn wait_on(self, stream: Stream) -> Result<()> {
        check(unsafe { cudaStreamWaitEvent(stream, self.0 as *mut c_void, 0) }, "cudaStreamWaitEvent")
    }
}

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

pub struct Device { pub stream: Stream, pub index: i32 }
impl Device {
    /// Device 0 must be a B200 (sm_100); the kernels are sm_100a binaries.
    pub fn open() -> Result<Self> { Self::open_index(0) }

    /// Makes device `index` current for the calling thread and opens a stream on it.
    pub fn open_index(index: i32) -> Result<Self> {
        check(unsafe { cudaSetDevice(index) }, "cudaSetDevice")?;
        let (mut major, mut minor) = (0, 0);
        check(unsafe { cudaDeviceGetAttribute(&mut major, 75, index) }, "compute capability")?;
        check(unsafe { cudaDeviceGetAttribute(&mut minor, 76, index) }, "compute capability")?;
        if (major, minor) != (10, 0) { return Err(format!("expected sm_100 B200, found sm_{major}{minor}").into()); }
        let mut stream = std::ptr::null_mut();
        check(unsafe { cudaStreamCreateWithFlags(&mut stream, 1) }, "cudaStreamCreate")?;
        STREAMS.lock().unwrap().push((index, stream as usize));
        Ok(Self { stream, index })
    }
    pub fn sync(&self) -> Result<()> { check(unsafe { cudaStreamSynchronize(self.stream) }, "cudaStreamSynchronize") }
    /// A second non-blocking stream for streaming weight uploads, so compute on `stream` never queues
    /// behind a load in progress. Core drains it before publishing each weight stage.
    pub fn upload_stream(&self) -> Result<Stream> {
        let mut stream = std::ptr::null_mut();
        check(unsafe { cudaStreamCreateWithFlags(&mut stream, 1) }, "cudaStreamCreate")?;
        Ok(stream)
    }
    pub fn free_bytes(&self)->Result<usize>{let(mut free,mut total)=(0,0);check(unsafe{cudaMemGetInfo(&mut free,&mut total)},"cudaMemGetInfo")?;Ok(free)}
    pub fn release_idle(&self){release_cached(self.index);}
}
impl Drop for Device {
    fn drop(&mut self) { unsafe { cudaStreamSynchronize(self.stream); cudaStreamDestroy(self.stream); } }
}

/// Owned device bytes. Freed after a device synchronization by the owner.
/// A device allocation. Allocations come from a per-device caching pool (like PyTorch's caching
/// allocator): a dropped buffer returns its block to the pool of its device and size class, and a
/// later `new` on that device reuses it, so steady-state requests make no cudaMalloc/cudaFree
/// calls (cudaFree blocks the host until the device is idle). Reuse is stream ordered: each device
/// has one stream, and cross-device readers finish before the owning stream drops their buffers.
/// `WAN_NO_POOL=1` restores plain cudaMalloc/cudaFree.
pub struct Buffer { ptr: *mut c_void, bytes: usize, device: c_int, class: usize }

type Pool = std::collections::HashMap<(c_int, usize), Vec<usize>>;
static POOL: std::sync::Mutex<Option<Pool>> = std::sync::Mutex::new(None);
thread_local! {static RETIRING:std::cell::Cell<usize>=const{std::cell::Cell::new(0)};}
/// Last-use object destruction releases its private scratch as well as weights.
/// Existing cached blocks (which the decoder may reuse) remain untouched.
pub struct RetirementScope;
impl RetirementScope{pub fn enter()->Self{RETIRING.with(|v|v.set(v.get()+1));Self}}
impl Drop for RetirementScope{fn drop(&mut self){RETIRING.with(|v|v.set(v.get()-1));}}

fn pool_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("WAN_NO_POOL").is_none()) && RETIRING.with(|v|v.get()==0)
}
/// Diagnostics: WAN_POOL_ZERO zero-fills reused blocks (like fresh cudaMalloc pages);
/// WAN_POOL_SYNC synchronizes the device before a block returns to the pool (like cudaFree).
fn pool_flag(name: &str) -> bool { std::env::var_os(name).is_some() }

/// Each device's stream (registered by Device::open_index), for stream-ordered pool work.
static STREAMS: std::sync::Mutex<Vec<(c_int, usize)>> = std::sync::Mutex::new(Vec::new());
fn device_stream(device: c_int) -> Option<Stream> {
    STREAMS.lock().unwrap().iter().find(|(d, _)| *d == device).map(|(_, s)| *s as Stream)
}

/// Size class: 512 B steps below 1 MiB, 2 MiB steps above.
fn size_class(bytes: usize) -> usize {
    let bytes = bytes.max(1);
    if bytes < 1 << 20 { bytes.div_ceil(512) * 512 } else { bytes.div_ceil(2 << 20) * (2 << 20) }
}

impl Buffer {
    pub fn new(bytes: usize) -> Result<Self> {
        let mut device: c_int = 0;
        check(unsafe { cudaGetDevice(&mut device) }, "cudaGetDevice")?;
        let class = size_class(bytes);
        if pool_enabled() {
            let cached = POOL.lock().unwrap().get_or_insert_with(Pool::new).get_mut(&(device, class)).and_then(Vec::pop);
            if let Some(ptr) = cached {
                if pool_flag("WAN_POOL_ZERO") {
                    let stream = device_stream(device).unwrap_or(std::ptr::null_mut());
                    check(unsafe { cudaMemsetAsync(ptr as *mut c_void, 0, class, stream) }, "pool zero")?;
                }
                return Ok(Self { ptr: ptr as *mut c_void, bytes, device, class });
            }
        }
        let mut ptr = std::ptr::null_mut();
        if unsafe { cudaMalloc(&mut ptr, class) } != 0 {
            unsafe { cudaGetLastError() };
            release_cached(device);  // out of memory with blocks cached: free them and retry once
            check(unsafe { cudaMalloc(&mut ptr, class) }, "cudaMalloc")?;
        }
        Ok(Self { ptr, bytes, device, class })
    }
    /// Filled with 0xFF so an unwritten output can never match a reference.
    pub fn poisoned(bytes: usize, device: &Device) -> Result<Self> {
        let buffer = Self::new(bytes)?;
        check(unsafe { cudaMemsetAsync(buffer.ptr, 0xFF, bytes, device.stream) }, "cudaMemset")?;
        Ok(buffer)
    }
    /// A non-owning view of device memory owned elsewhere (a model_load allocation); dropping it frees nothing.
    pub fn view(ptr: u64, bytes: usize) -> Result<Self> {
        let mut device: c_int = 0;
        check(unsafe { cudaGetDevice(&mut device) }, "cudaGetDevice")?;
        Ok(Self { ptr: ptr as *mut c_void, bytes, device, class: 0 })
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
    /// Read an FP32 output directly into its correctly aligned host vector.
    pub fn download_f32(&self,device:&Device)->Result<Vec<f32>> {
        if self.bytes%4!=0 {return Err("FP32 download byte count".into());}
        let mut out=vec![0f32;self.bytes/4];device.sync()?;
        check(unsafe{cudaMemcpyAsync(out.as_mut_ptr().cast(),self.ptr,self.bytes,DEVICE_TO_HOST,device.stream)},"FP32 download")?;
        device.sync()?;
        if cfg!(target_endian="big") {for value in &mut out {*value=f32::from_bits(value.to_bits().swap_bytes());}}
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
        if self.class == 0 { return; }  // a view (Buffer::view)
        if pool_enabled() {
            if pool_flag("WAN_POOL_SYNC") {
                let mut current: c_int = 0;
                unsafe { cudaGetDevice(&mut current); cudaSetDevice(self.device); cudaDeviceSynchronize(); cudaSetDevice(current); }
            }
            POOL.lock().unwrap().get_or_insert_with(Pool::new).entry((self.device, self.class)).or_default().push(self.ptr as usize);
        } else {
            unsafe { cudaFree(self.ptr); }
        }
    }
}

/// Frees the cached blocks of the current device (cuDNN workspace out-of-memory handler).
pub extern "C" fn release_cached_current_device() -> c_int {
    let mut device: c_int = 0;
    unsafe { cudaGetDevice(&mut device) };
    release_cached(device);
    0
}

/// Frees the cached (unused) blocks of `device`.
fn release_cached(device: c_int) {
    let blocks: Vec<usize> = match POOL.lock().unwrap().as_mut() {
        Some(pool) => pool.iter_mut().filter(|((d, _), _)| *d == device).flat_map(|(_, v)| v.drain(..)).collect(),
        None => Vec::new(),
    };
    let mut current: c_int = 0;
    unsafe {
        cudaGetDevice(&mut current);
        cudaSetDevice(device);
        for ptr in blocks { cudaFree(ptr as *mut c_void); }
        cudaSetDevice(current);
    }
}
