//! CUDA driver API: context, streams, device memory, modules and launches (libcuda only).
#![allow(non_camel_case_types)]
use crate::Result;
use std::ffi::{c_char, c_int, c_uint, c_void, CStr, CString};

pub type CUresult = c_int;
pub type CUdevice = c_int;
pub type CUcontext = *mut c_void;
pub type CUmodule = *mut c_void;
pub type CUfunction = *mut c_void;
pub type CUstream = *mut c_void;
pub type CUevent = *mut c_void;
pub type CUdeviceptr = u64;

/// CUlaunchAttribute: 4-byte id, 4 bytes padding, 64-byte value union.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct LaunchAttribute { pub id: c_uint, pub pad: [u8; 4], pub value: [u8; 64] }

#[repr(C)]
struct LaunchConfig {
    grid: [c_uint; 3], block: [c_uint; 3], shared: c_uint, stream: CUstream,
    attrs: *const LaunchAttribute, num_attrs: c_uint,
}

/// CUtensorMap: 128 opaque bytes, 64-byte aligned, passed by value as a kernel parameter.
#[repr(C, align(64))]
#[derive(Clone, Copy)]
pub struct TensorMap(pub [u8; 128]);

#[link(name = "cuda")]
extern "C" {
    fn cuInit(flags: c_uint) -> CUresult;
    fn cuDeviceGet(device: *mut CUdevice, ordinal: c_int) -> CUresult;
    fn cuDevicePrimaryCtxRetain(ctx: *mut CUcontext, device: CUdevice) -> CUresult;
    fn cuCtxSetCurrent(ctx: CUcontext) -> CUresult;
    fn cuCtxSynchronize() -> CUresult;
    fn cuDeviceGetAttribute(value: *mut c_int, attribute: c_int, device: CUdevice) -> CUresult;
    fn cuMemAlloc_v2(ptr: *mut CUdeviceptr, bytes: usize) -> CUresult;
    fn cuMemFree_v2(ptr: CUdeviceptr) -> CUresult;
    fn cuMemcpyHtoD_v2(dst: CUdeviceptr, src: *const c_void, bytes: usize) -> CUresult;
    fn cuMemcpyDtoH_v2(dst: *mut c_void, src: CUdeviceptr, bytes: usize) -> CUresult;
    fn cuMemcpyHtoDAsync_v2(dst: CUdeviceptr, src: *const c_void, bytes: usize, stream: CUstream) -> CUresult;
    fn cuMemcpyDtoDAsync_v2(dst: CUdeviceptr, src: CUdeviceptr, bytes: usize, stream: CUstream) -> CUresult;
    fn cuMemsetD8Async(dst: CUdeviceptr, value: u8, count: usize, stream: CUstream) -> CUresult;
    fn cuMemsetD8_v2(dst: CUdeviceptr, value: u8, count: usize) -> CUresult;
    fn cuMemsetD32Async(dst: CUdeviceptr, value: c_uint, count: usize, stream: CUstream) -> CUresult;
    fn cuStreamCreate(stream: *mut CUstream, flags: c_uint) -> CUresult;
    fn cuEventCreate(event: *mut *mut c_void, flags: c_uint) -> CUresult;
    fn cuEventRecord(event: *mut c_void, stream: CUstream) -> CUresult;
    fn cuStreamWaitEvent(stream: CUstream, event: *mut c_void, flags: c_uint) -> CUresult;
    fn cuStreamBeginCapture_v2(stream: CUstream, mode: c_uint) -> CUresult;
    fn cuStreamEndCapture(stream: CUstream, graph: *mut *mut c_void) -> CUresult;
    fn cuGraphInstantiateWithFlags(exec: *mut *mut c_void, graph: *mut c_void, flags: u64) -> CUresult;
    fn cuGraphLaunch(exec: *mut c_void, stream: CUstream) -> CUresult;
    fn cuGraphExecDestroy(exec: *mut c_void) -> CUresult;
    fn cuGraphDestroy(graph: *mut c_void) -> CUresult;
    fn cuStreamSynchronize(stream: CUstream) -> CUresult;
    fn cuModuleLoadData(module: *mut CUmodule, image: *const c_void) -> CUresult;
    fn cuModuleGetFunction(function: *mut CUfunction, module: CUmodule, name: *const c_char) -> CUresult;
    fn cuFuncSetAttribute(function: CUfunction, attribute: c_int, value: c_int) -> CUresult;
    fn cuFuncGetAttribute(value: *mut c_int, attribute: c_int, function: CUfunction) -> CUresult;
    fn cuFuncGetParamInfo(function: CUfunction, index: usize, offset: *mut usize, size: *mut usize) -> CUresult;
    fn cuLaunchKernel(f: CUfunction, gx: c_uint, gy: c_uint, gz: c_uint, bx: c_uint, by: c_uint, bz: c_uint,
                      shared: c_uint, stream: CUstream, params: *mut *mut c_void, extra: *mut *mut c_void) -> CUresult;
    fn cuLaunchKernelEx(config: *const LaunchConfig, f: CUfunction, params: *mut *mut c_void,
                        extra: *mut *mut c_void) -> CUresult;
    fn cuTensorMapEncodeTiled(map: *mut TensorMap, dtype: c_uint, rank: c_uint, address: *mut c_void,
                              dims: *const u64, strides: *const u64, box_dims: *const u32, element_strides: *const u32,
                              interleave: c_uint, swizzle: c_uint, l2: c_uint, oob: c_uint) -> CUresult;
    fn cuGetErrorString(code: CUresult, text: *mut *const c_char) -> CUresult;
}

pub fn check(code: CUresult, what: &str) -> Result<()> {
    if code == 0 { return Ok(()); }
    let mut text = std::ptr::null();
    unsafe { cuGetErrorString(code, &mut text) };
    let text = if text.is_null() { "unknown".into() } else { unsafe { CStr::from_ptr(text) }.to_string_lossy() };
    Err(format!("{what}: CUDA driver error {code}: {text}"))
}

pub struct Device { pub ordinal: c_int, pub ctx: CUcontext, pub sm_count: u32, pub stream: Stream }
// The primary context is a process-wide handle: a device opened on a helper thread is used on
// another after `set_current(ctx)`.
unsafe impl Send for Device {}

impl Device {
    pub fn open(ordinal: c_int) -> Result<Self> {
        unsafe {
            check(cuInit(0), "cuInit")?;
            let mut device = 0;
            check(cuDeviceGet(&mut device, ordinal), "cuDeviceGet")?;
            let mut ctx = std::ptr::null_mut();
            check(cuDevicePrimaryCtxRetain(&mut ctx, device), "cuDevicePrimaryCtxRetain")?;
            check(cuCtxSetCurrent(ctx), "cuCtxSetCurrent")?;
            let mut sms = 0;
            check(cuDeviceGetAttribute(&mut sms, 16 /* MULTIPROCESSOR_COUNT */, device), "sm count")?;
            Ok(Self { ordinal, ctx, sm_count: sms as u32, stream: Stream::new()? })
        }
    }
    pub fn synchronize(&self) -> Result<()> { check(unsafe { cuCtxSynchronize() }, "cuCtxSynchronize") }
}

#[derive(Clone, Copy, Debug)]
pub struct Stream(pub CUstream);
unsafe impl Send for Stream {}

impl Stream {
    pub fn new() -> Result<Self> {
        let mut s = std::ptr::null_mut();
        check(unsafe { cuStreamCreate(&mut s, 1 /* NON_BLOCKING */) }, "cuStreamCreate")?;
        Ok(Self(s))
    }
    pub fn synchronize(&self) -> Result<()> { check(unsafe { cuStreamSynchronize(self.0) }, "cuStreamSynchronize") }
}

/// A synchronization event (no timing).
pub struct Event(pub *mut c_void);
unsafe impl Send for Event {}

impl Event {
    pub fn new() -> Result<Self> {
        let mut e = std::ptr::null_mut();
        check(unsafe { cuEventCreate(&mut e, 2 /* DISABLE_TIMING */) }, "cuEventCreate")?;
        Ok(Self(e))
    }
}

/// Make `to` wait for everything enqueued on `from` so far (a fork or join; inside a stream
/// capture it becomes a graph edge).
pub fn stream_wait(from: Stream, to: Stream, ev: &Event) -> Result<()> {
    check(unsafe { cuEventRecord(ev.0, from.0) }, "cuEventRecord")?;
    check(unsafe { cuStreamWaitEvent(to.0, ev.0, 0) }, "cuStreamWaitEvent")
}

/// An owned device allocation.
pub struct Buffer { pub ptr: CUdeviceptr, pub bytes: usize }
unsafe impl Send for Buffer {}

impl Buffer {
    pub fn alloc(bytes: usize) -> Result<Self> {
        let mut ptr = 0;
        check(unsafe { cuMemAlloc_v2(&mut ptr, bytes.max(1)) }, &format!("cuMemAlloc({bytes})"))?;
        Ok(Self { ptr, bytes })
    }
    pub fn from_host(data: &[u8]) -> Result<Self> {
        let b = Self::alloc(data.len())?;
        b.upload(data)?;
        Ok(b)
    }
    /// Copy `data` in and wait for it to land: a pageable cuMemcpyHtoD returns once the data is staged,
    /// before the DMA completes, and kernels on other (non-blocking) streams are not ordered after it.
    pub fn upload(&self, data: &[u8]) -> Result<()> {
        if data.len() > self.bytes { return Err("upload larger than buffer".into()); }
        check(unsafe { cuMemcpyHtoD_v2(self.ptr, data.as_ptr().cast(), data.len()) }, "cuMemcpyHtoD")?;
        ctx_sync()
    }
    /// Zero the buffer and wait for it.
    pub fn zero(&self) -> Result<()> {
        check(unsafe { cuMemsetD8_v2(self.ptr, 0, self.bytes) }, "cuMemsetD8")?;
        ctx_sync()
    }
    pub fn download(&self) -> Result<Vec<u8>> { download(self.ptr, self.bytes) }
}

impl Drop for Buffer {
    fn drop(&mut self) { unsafe { cuMemFree_v2(self.ptr) }; }
}

/// Wait for all work in the context (every stream).
/// A captured CUDA graph: one stream's launches recorded once and replayed with one call (how
/// SGLang runs decode). Everything that changes per replay must come from device memory.
pub struct Graph { graph: *mut c_void, exec: *mut c_void }

impl Graph {
    /// Record what `f` launches on `stream` (relaxed capture mode: other threads may keep using CUDA).
    pub fn capture<T>(stream: Stream, f: impl FnOnce() -> Result<T>) -> Result<(Self, T)> {
        check(unsafe { cuStreamBeginCapture_v2(stream.0, 2) }, "cuStreamBeginCapture")?;
        let result = f();
        let mut graph = std::ptr::null_mut();
        let ended = check(unsafe { cuStreamEndCapture(stream.0, &mut graph) }, "cuStreamEndCapture");
        let value = result?;
        ended?;
        let mut exec = std::ptr::null_mut();
        check(unsafe { cuGraphInstantiateWithFlags(&mut exec, graph, 0) }, "cuGraphInstantiate")?;
        Ok((Self { graph, exec }, value))
    }
    pub fn launch(&self, stream: Stream) -> Result<()> { check(unsafe { cuGraphLaunch(self.exec, stream.0) }, "cuGraphLaunch") }
}

impl Drop for Graph {
    fn drop(&mut self) { unsafe { cuGraphExecDestroy(self.exec); cuGraphDestroy(self.graph); } }
}

/// Make `ctx` current on the calling thread (helper threads working in the device's context).
pub fn set_current(ctx: CUcontext) -> Result<()> { check(unsafe { cuCtxSetCurrent(ctx) }, "cuCtxSetCurrent") }

pub fn ctx_sync() -> Result<()> { check(unsafe { cuCtxSynchronize() }, "cuCtxSynchronize") }

pub fn download(ptr: CUdeviceptr, bytes: usize) -> Result<Vec<u8>> {
    let mut out = vec![0u8; bytes];
    check(unsafe { cuMemcpyDtoH_v2(out.as_mut_ptr().cast(), ptr, bytes) }, "cuMemcpyDtoH")?;
    Ok(out)
}

pub fn upload_async(dst: CUdeviceptr, data: &[u8], stream: Stream) -> Result<()> {
    check(unsafe { cuMemcpyHtoDAsync_v2(dst, data.as_ptr().cast(), data.len(), stream.0) }, "cuMemcpyHtoDAsync")
}

pub fn copy_async(dst: CUdeviceptr, src: CUdeviceptr, bytes: usize, stream: Stream) -> Result<()> {
    check(unsafe { cuMemcpyDtoDAsync_v2(dst, src, bytes, stream.0) }, "cuMemcpyDtoDAsync")
}

pub fn memset_async(dst: CUdeviceptr, value: u8, bytes: usize, stream: Stream) -> Result<()> {
    check(unsafe { cuMemsetD8Async(dst, value, bytes, stream.0) }, "cuMemsetD8Async")
}

pub fn memset32_async(dst: CUdeviceptr, value: u32, count: usize, stream: Stream) -> Result<()> {
    check(unsafe { cuMemsetD32Async(dst, value, count, stream.0) }, "cuMemsetD32Async")
}

/// A loaded module image (CUBIN); keeps the module alive for its functions.
pub struct Module { pub handle: CUmodule }
unsafe impl Send for Module {}
unsafe impl Sync for Module {}

impl Module {
    pub fn load(image: &[u8]) -> Result<Self> {
        let mut handle = std::ptr::null_mut();
        check(unsafe { cuModuleLoadData(&mut handle, image.as_ptr().cast()) }, "cuModuleLoadData")?;
        Ok(Self { handle })
    }
    pub fn function(&self, name: &str) -> Result<Function> {
        let entry = CString::new(name).map_err(|e| e.to_string())?;
        let mut f = std::ptr::null_mut();
        check(unsafe { cuModuleGetFunction(&mut f, self.handle, entry.as_ptr()) }, &format!("cuModuleGetFunction {name}"))?;
        Ok(Function { handle: f, name: name.into() })
    }
}

#[derive(Clone)]
pub struct Function { pub handle: CUfunction, pub name: String }
unsafe impl Send for Function {}
unsafe impl Sync for Function {}

impl Function {
    pub fn set_max_dynamic_shared(&self, bytes: u32) -> Result<()> {
        check(unsafe { cuFuncSetAttribute(self.handle, 8, bytes as c_int) }, "cuFuncSetAttribute(max dynamic smem)")
    }
    /// cuFuncSetAttribute (e.g. 14: CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED).
    pub fn set_attribute(&self, attribute: c_int, value: c_int) -> Result<()> {
        check(unsafe { cuFuncSetAttribute(self.handle, attribute, value) }, "cuFuncSetAttribute")
    }
    pub fn attribute(&self, attribute: c_int) -> Result<i32> {
        let mut v = 0;
        check(unsafe { cuFuncGetAttribute(&mut v, attribute, self.handle) }, "cuFuncGetAttribute")?;
        Ok(v)
    }
    /// (offset, size) of each declared parameter.
    pub fn params(&self) -> Vec<(usize, usize)> {
        let mut out = Vec::new();
        for i in 0.. {
            let (mut offset, mut size) = (0usize, 0usize);
            if unsafe { cuFuncGetParamInfo(self.handle, i, &mut offset, &mut size) } != 0 { break; }
            out.push((offset, size));
        }
        out
    }
    /// Launch with each parameter's bytes (in declaration order) and optional launch attributes.
    pub fn launch(&self, grid: [u32; 3], block: [u32; 3], shared: u32, stream: Stream, params: &[Vec<u8>],
                  attrs: &[LaunchAttribute]) -> Result<()> {
        // Each parameter gets its own 64-byte-aligned slot (TMA descriptors need the alignment).
        let mut storage: Vec<TensorMap> = params.iter().map(|_| TensorMap([0; 128])).collect();
        let mut big: Vec<Vec<u8>> = Vec::new();
        let mut pointers: Vec<*mut c_void> = Vec::with_capacity(params.len());
        for (i, p) in params.iter().enumerate() {
            if p.len() <= 128 {
                storage[i].0[..p.len()].copy_from_slice(p);
                pointers.push(storage[i].0.as_mut_ptr().cast());
            } else {
                big.push(p.clone());
                pointers.push(big.last_mut().unwrap().as_mut_ptr().cast());
            }
        }
        let code = if attrs.is_empty() {
            unsafe { cuLaunchKernel(self.handle, grid[0], grid[1], grid[2], block[0], block[1], block[2], shared, stream.0,
                                    pointers.as_mut_ptr(), std::ptr::null_mut()) }
        } else {
            let config = LaunchConfig { grid, block, shared, stream: stream.0, attrs: attrs.as_ptr(), num_attrs: attrs.len() as c_uint };
            unsafe { cuLaunchKernelEx(&config, self.handle, pointers.as_mut_ptr(), std::ptr::null_mut()) }
        };
        check(code, &format!("launch {}", self.name))
    }
}

#[allow(clippy::too_many_arguments)]
pub fn encode_tiled(dtype: u32, address: CUdeviceptr, dims: &[u64], strides: &[u64], box_dims: &[u32],
                    element_strides: &[u32], interleave: u32, swizzle: u32, l2: u32, oob: u32) -> Result<TensorMap> {
    let mut map = TensorMap([0; 128]);
    check(unsafe {
        cuTensorMapEncodeTiled(&mut map, dtype, dims.len() as c_uint, address as *mut c_void, dims.as_ptr(), strides.as_ptr(),
                               box_dims.as_ptr(), element_strides.as_ptr(), interleave, swizzle, l2, oob)
    }, "cuTensorMapEncodeTiled")?;
    Ok(map)
}
