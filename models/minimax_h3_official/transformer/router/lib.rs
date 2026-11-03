//! Native CUDA-driver execution of the unchanged Triton router CUBINs exported from the reference model.
//! Compilation/export belongs to the offline harness. This library has no Python ABI.
use std::{collections::HashMap, ffi::{c_char, c_int, c_uint, c_void, CStr}, fs,
          path::Path, ptr};

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
pub type Stream = *mut c_void;
pub type DevicePtr = u64;
type Module = *mut c_void;
type Function = *mut c_void;

unsafe extern "C" {
    fn cuInit(flags: c_uint) -> c_int;
    fn cuDeviceGet(device: *mut c_int, ordinal: c_int) -> c_int;
    fn cuDeviceGetAttribute(value: *mut c_int, attribute: c_int, device: c_int) -> c_int;
    fn cuDevicePrimaryCtxRetain(context: *mut *mut c_void, device: c_int) -> c_int;
    fn cuDevicePrimaryCtxRelease_v2(device: c_int) -> c_int;
    fn cuCtxSetCurrent(context: *mut c_void) -> c_int;
    fn cuCtxSynchronize() -> c_int;
    fn cuStreamCreate(stream: *mut Stream, flags: c_uint) -> c_int;
    fn cuStreamDestroy_v2(stream: Stream) -> c_int;
    fn cuStreamSynchronize(stream: Stream) -> c_int;
    fn cuMemAlloc_v2(pointer: *mut DevicePtr, bytes: usize) -> c_int;
    fn cuMemFree_v2(pointer: DevicePtr) -> c_int;
    fn cuMemcpyHtoD_v2(dst: DevicePtr, src: *const c_void, bytes: usize) -> c_int;
    fn cuMemcpyDtoH_v2(dst: *mut c_void, src: DevicePtr, bytes: usize) -> c_int;
    fn cuModuleLoadData(module: *mut Module, image: *const c_void) -> c_int;
    fn cuModuleUnload(module: Module) -> c_int;
    fn cuModuleGetFunction(function: *mut Function, module: Module, name: *const c_char) -> c_int;
    fn cuFuncSetAttribute(function: Function, attribute: c_int, value: c_int) -> c_int;
    fn cuLaunchKernel(function: Function, gx: c_uint, gy: c_uint, gz: c_uint,
        bx: c_uint, by: c_uint, bz: c_uint, shared: c_uint, stream: Stream,
        params: *mut *mut c_void, extra: *mut *mut c_void) -> c_int;
    fn cuGetErrorString(code: c_int, text: *mut *const c_char) -> c_int;
}

pub fn check(code: c_int) -> Result<()> {
    if code == 0 { return Ok(()); }
    let mut p = ptr::null();
    unsafe { cuGetErrorString(code, &mut p); }
    let message = if p.is_null() { "unknown CUDA driver error".into() }
        else { unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned() };
    Err(format!("CUDA driver {code}: {message}").into())
}

/// A single stream, owned allocations, and a retained primary context for standalone replay.
/// An integrated Rust runtime can instead construct Program under its own current context.
pub struct Arena {
    pub stream: Stream,
    device: c_int,
    allocations: Vec<(DevicePtr, usize)>,
}
impl Arena {
    pub fn new() -> Result<Self> {
        check(unsafe { cuInit(0) })?;
        let mut device = 0;
        check(unsafe { cuDeviceGet(&mut device, 0) })?;
        let (mut major, mut minor) = (0, 0);
        check(unsafe { cuDeviceGetAttribute(&mut major, 75, device) })?;
        check(unsafe { cuDeviceGetAttribute(&mut minor, 76, device) })?;
        if (major, minor) != (10, 0) { return Err("exported router requires sm_100 B200".into()); }
        let mut context = ptr::null_mut();
        check(unsafe { cuDevicePrimaryCtxRetain(&mut context, device) })?;
        check(unsafe { cuCtxSetCurrent(context) })?;
        let mut stream = ptr::null_mut();
        check(unsafe { cuStreamCreate(&mut stream, 1) })?;
        Ok(Self { stream, device, allocations: Vec::new() })
    }
    pub fn alloc(&mut self, bytes: usize) -> Result<DevicePtr> {
        if bytes == 0 { return Err("zero-sized native router allocation".into()); }
        let mut p = 0;
        check(unsafe { cuMemAlloc_v2(&mut p, bytes) })?;
        self.allocations.push((p, bytes));
        Ok(p)
    }
    pub fn upload(&self, pointer: DevicePtr, bytes: &[u8]) -> Result<()> {
        if !self.allocations.iter().any(|&(p, n)| p == pointer && bytes.len() <= n) {
            return Err("upload exceeds an owned allocation".into());
        }
        check(unsafe { cuMemcpyHtoD_v2(pointer, bytes.as_ptr().cast(), bytes.len()) })?;
        // Pageable H2D can return after staging, before device DMA completes.
        // The compute stream is NON_BLOCKING, so explicitly finish the default-stream copy.
        check(unsafe { cuCtxSynchronize() })
    }
    pub fn download(&self, pointer: DevicePtr, bytes: usize) -> Result<Vec<u8>> {
        if !self.allocations.iter().any(|&(p, n)| p == pointer && bytes <= n) {
            return Err("download exceeds an owned allocation".into());
        }
        self.sync()?;
        let mut output = vec![0; bytes];
        check(unsafe { cuMemcpyDtoH_v2(output.as_mut_ptr().cast(), pointer, bytes) })?;
        Ok(output)
    }
    pub fn sync(&self) -> Result<()> { synchronize(self.stream) }
}
impl Drop for Arena {
    fn drop(&mut self) {
        let _ = self.sync();
        for &(p, _) in &self.allocations { let _ = unsafe { cuMemFree_v2(p) }; }
        unsafe { cuStreamDestroy_v2(self.stream); cuDevicePrimaryCtxRelease_v2(self.device); }
    }
}
pub fn synchronize(stream: Stream) -> Result<()> { check(unsafe { cuStreamSynchronize(stream) }) }

#[derive(Debug)]
pub struct BufferSpec {
    pub name: String, pub input: bool, pub file: String, pub bytes: usize,
}
enum Arg { Pointer(String), Bits(u64) }
struct Kernel {
    name: String, module: Module, function: Function,
    grid: [u32; 3], threads: u32, shared: u32, args: Vec<Arg>,
}
impl Drop for Kernel {
    fn drop(&mut self) { unsafe { cuModuleUnload(self.module); } }
}

pub struct Program { pub buffers: Vec<BufferSpec>, kernels: Vec<Kernel> }
impl Program {
    /// Loads a shape-specialized, pre-exported manifest into the current CUDA context.
    pub fn load(root: &Path) -> Result<Self> {
        let manifest = fs::read_to_string(root.join("launch.tsv"))?;
        let mut lines = manifest.lines();
        if lines.next() != Some("H3_ROUTER_AOT_V1") { return Err("invalid router manifest version".into()); }
        let mut program = Self { buffers: Vec::new(), kernels: Vec::new() };
        for line in lines {
            if line.is_empty() { continue; }
            let fields: Vec<_> = line.split('\t').collect();
            match fields[0] {
                "buffer" if fields.len() == 6 => {
                    let bytes: usize = fields[4].parse()?;
                    if bytes == 0 || !["input", "output"].contains(&fields[2]) { return Err("invalid buffer".into()); }
                    if fields[5] != "contiguous" { return Err("only contiguous fixture files are supported".into()); }
                    program.buffers.push(BufferSpec { name: fields[1].into(), input: fields[2] == "input",
                        file: fields[3].into(), bytes });
                },
                "launch" if fields.len() == 9 => {
                    let cubin = fs::read(root.join(fields[2]))?;
                    let mut module = ptr::null_mut();
                    check(unsafe { cuModuleLoadData(&mut module, cubin.as_ptr().cast()) })?;
                    let name = std::ffi::CString::new(fields[3])?;
                    let mut kernel = Kernel { name: fields[1].into(), module,
                        function: ptr::null_mut(), grid: [fields[4].parse()?, fields[5].parse()?, fields[6].parse()?],
                        threads: fields[7].parse()?, shared: fields[8].parse()?, args: Vec::new() };
                    check(unsafe { cuModuleGetFunction(&mut kernel.function, module, name.as_ptr()) })?;
                    if kernel.shared > 0 {
                        check(unsafe { cuFuncSetAttribute(kernel.function, 8, kernel.shared as i32) })?;
                    }
                    if kernel.grid.contains(&0) || kernel.threads == 0 || kernel.threads > 1024 {
                        return Err("invalid CUDA launch dimensions".into());
                    }
                    program.kernels.push(kernel);
                },
                "arg" if fields.len() == 3 => {
                    let kernel = program.kernels.last_mut().ok_or("argument without kernel")?;
                    kernel.args.push(match fields[1] {
                        "ptr" => Arg::Pointer(fields[2].into()),
                        "bits" => Arg::Bits(fields[2].parse()?),
                        _ => return Err("unknown parameter encoding".into()),
                    });
                },
                _ => return Err(format!("malformed manifest line: {line}").into()),
            }
        }
        if program.kernels.len() != 4 { return Err("router requires pool Q, pool K, scores, top-k".into()); }
        Ok(program)
    }
    /// Buffer pointers and stream belong to the caller and must outlive stream completion.
    /// Shape, dtype and strides must match the frozen launch manifest exactly.
    pub fn launch(&self, pointers: &HashMap<String, DevicePtr>, stream: Stream) -> Result<()> {
        for buffer in &self.buffers {
            let pointer = pointers.get(&buffer.name).copied().unwrap_or(0);
            if pointer == 0 || pointer % 16 != 0 {
                return Err(format!("router buffer {} must be nonnull and 16-byte aligned", buffer.name).into());
            }
        }
        for kernel in &self.kernels {
            let mut values = Vec::<u64>::with_capacity(kernel.args.len());
            for arg in &kernel.args {
                values.push(match arg {
                    Arg::Pointer(name) => *pointers.get(name).ok_or_else(|| format!("missing {name}"))?,
                    Arg::Bits(value) => *value,
                });
            }
            let mut args: Vec<*mut c_void> = values.iter_mut().map(|value| (value as *mut u64).cast()).collect();
            check(unsafe { cuLaunchKernel(kernel.function, kernel.grid[0], kernel.grid[1], kernel.grid[2],
                kernel.threads, 1, 1, kernel.shared, stream, args.as_mut_ptr(), ptr::null_mut()) })
                .map_err(|e| format!("{}: {e}", kernel.name))?;
        }
        Ok(())
    }
}
impl Drop for Program {
    fn drop(&mut self) {
        // Modules must outlive queued kernels even if the caller drops after an error.
        unsafe { cuCtxSynchronize(); }
    }
}

/// C integration API. Caller sets a CUDA context; all six pointers are device pointers.
/// Failure returns -1 and writes details to stderr. A successfully loaded handle owns modules.
#[no_mangle]
pub unsafe extern "C" fn h3_router_load(path: *const c_char, out: *mut *mut Program) -> c_int {
    if path.is_null() || out.is_null() { return -1; }
    match CStr::from_ptr(path).to_str().map_err(|e| e.into()).and_then(|p| Program::load(Path::new(p))) {
        Ok(program) => { *out = Box::into_raw(Box::new(program)); 0 },
        Err(error) => { eprintln!("h3_router_load: {error}"); -1 },
    }
}
#[no_mangle]
pub unsafe extern "C" fn h3_router_forward(program: *const Program, q: DevicePtr, k: DevicePtr,
    qp: DevicePtr, kp: DevicePtr, scores: DevicePtr, indices: DevicePtr, stream: Stream) -> c_int {
    if program.is_null() { return -1; }
    let pointers = [ ("q", q), ("k", k), ("qp", qp), ("kp", kp), ("scores", scores), ("indices", indices) ]
        .into_iter().map(|(name, p)| (name.to_owned(), p)).collect();
    match (*program).launch(&pointers, stream) {
        Ok(()) => 0,
        Err(error) => { eprintln!("h3_router_forward: {error}"); -1 },
    }
}
#[no_mangle]
pub unsafe extern "C" fn h3_router_buffer_bytes(program: *const Program, sizes: *mut usize) -> c_int {
    if program.is_null() || sizes.is_null() { return -1; }
    for (index, name) in ["q", "k", "qp", "kp", "scores", "indices"].into_iter().enumerate() {
        let Some(buffer) = (*program).buffers.iter().find(|buffer| buffer.name == name) else { return -1; };
        *sizes.add(index) = buffer.bytes;
    }
    0
}
#[no_mangle]
pub unsafe extern "C" fn h3_router_destroy(program: *mut Program) {
    if !program.is_null() { drop(Box::from_raw(program)); }
}
