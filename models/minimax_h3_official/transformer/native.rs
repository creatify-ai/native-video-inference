//! Explicit H3 host execution. All device work stays in the validated CUDA/library ABIs.
//! The only inference data read from steps after zero are fixed schedule/index tables.
use crate::{bytes_i64, compare, finite_f32, f32_values, i64_values, row_timestep_indices,
    validate_packed_indices, Result, TensorSpec};
use sha2::{Digest,Sha256};
use crate::dynamic_router::DynamicProgram as Program;
use serde_json::{json, Value};
use std::{collections::HashMap, ffi::{c_char, c_int, c_void, CStr, CString},
          fs, path::{Path, PathBuf}, ptr, time::Instant};

use crate::profile::{self,HostSpan,GpuSpan};

type Stream = *mut c_void;
fn fusion_enabled(name:&str)->bool {
    std::env::var(name).as_deref()!=Ok("0")
}
fn qkv_inplace_enabled()->bool {
    static FLAG:std::sync::OnceLock<bool>=std::sync::OnceLock::new();
    *FLAG.get_or_init(||fusion_enabled("H3_QKV_INPLACE"))
}
fn cross_layer_fusion_enabled()->bool {
    static FLAG:std::sync::OnceLock<bool>=std::sync::OnceLock::new();
    *FLAG.get_or_init(||fusion_enabled("H3_CROSS_LAYER_FUSION"))
}
fn final_norm_fusion_enabled()->bool {
    static FLAG:std::sync::OnceLock<bool>=std::sync::OnceLock::new();
    *FLAG.get_or_init(||fusion_enabled("H3_FINAL_NORM_FUSION"))
}
const HIDDEN: usize = 5376;
const HEADS: usize = 56;
const INNER: usize = HEADS * 128;
const FF: usize = 14336;
const VIDEO: usize = 96;
const AUDIO: usize = 32;

unsafe extern "C" {
    fn cudaGetErrorString(code: c_int) -> *const c_char;
    fn cudaMalloc(pointer: *mut *mut c_void, bytes: usize) -> c_int;
    fn cudaGetLastError() -> c_int;
    fn cudaFree(pointer: *mut c_void) -> c_int;
    fn cudaStreamCreateWithFlags(stream: *mut Stream, flags: u32) -> c_int;
    fn cudaStreamSynchronize(stream: Stream) -> c_int;
    fn cudaStreamDestroy(stream: Stream) -> c_int;
    fn cudaMemcpyAsync(dst: *mut c_void, src: *const c_void, bytes: usize, kind: c_int, stream: Stream) -> c_int;
    fn cudaMemsetAsync(dst: *mut c_void, value: c_int, bytes: usize, stream: Stream) -> c_int;
    fn h3_check_device() -> c_int;
    fn h3_bsa_workspace_bytes(tokens: c_int, heads: c_int, kept: c_int, bytes: *mut usize) -> c_int;
    fn h3_bsa_masked_workspace_bytes(tokens: c_int, heads: c_int, bytes: *mut usize) -> c_int;
    fn h3_attention_copy_heads(input: *const c_void, output: *mut c_void, rows: i64,
        first: c_int, heads: c_int, scatter: c_int, stream: Stream) -> c_int;
    fn h3_bsa_forward_masked(q: *const c_void, k: *const c_void, v: *const c_void,
        index: *const i32, block_sizes: *const i32, sparse_mask: *const i32, out: *mut c_void,
        workspace: *mut c_void, bytes: usize, tokens: c_int, heads: c_int,
        kept: c_int, scale: f32, stream: Stream) -> c_int;
    fn h3_linear_create(stream: Stream, workspace: *mut c_void, bytes: usize,
        lt_workspace: *mut c_void, lt_bytes: usize, tf32: c_int, reduced: c_int,
        out: *mut *mut c_void) -> c_int;
    fn h3_linear_forward(handle: *mut c_void, input: *const c_void, weight: *const c_void,
        bias: *const c_void, out: *mut c_void, rows: c_int, n: c_int, k: c_int, dtype: c_int) -> c_int;
    fn h3_sglang_qkv_rope_inplace(qkv:*mut c_void,qw:*const c_void,kw:*const c_void,
        cos:*const c_void,sin:*const c_void,tokens:i64,heads:c_int,qe:f32,ke:f32,stream:Stream)->c_int;
    fn h3_pw_final_gated_norm_mod_cast(hidden:*mut c_void,branch:*const c_void,gate:*const c_void,
        weight:*const c_void,scale:*const c_void,shift:*const c_void,gate_index:*const i64,
        mod_index:*const i64,out:*mut f32,rows:i64,cols:c_int,gate_stride:c_int,
        mod_stride:c_int,eps:f32,stream:Stream)->c_int;
    fn h3_linear_destroy(handle: *mut c_void);
    fn h3_linear_error() -> *const c_char;
    fn h3_prod_rms_mod_bf16(x: *const c_void, weight: *const c_void, scale: *const c_void,
        shift: *const c_void, index: *const i64, out: *mut c_void, rows: i64, cols: c_int,
        table_stride: c_int, eps: f32, stream: Stream) -> c_int;
    fn h3_prod_gated_rms_mod_bf16(hidden: *mut c_void, branch: *const c_void, gate: *const c_void,
        weight: *const c_void, scale: *const c_void, shift: *const c_void, index: *const i64,
        out: *mut c_void, rows: i64, cols: c_int, table_stride: c_int, eps: f32, stream: Stream) -> c_int;
    fn h3_prod_swiglu_bf16(x: *const c_void, out: *mut c_void, rows: i64, cols: c_int, stream: Stream) -> c_int;
    fn h3_prod_add_bf16(x:*const c_void,y:*const c_void,out:*mut c_void,n:i64,stream:Stream)->c_int;
    fn h3_prod_qk_rope(q: *const c_void,k: *const c_void,qw: *const c_void,kw: *const c_void,
        cos: *const c_void,sin: *const c_void,qn: *mut c_void,kn: *mut c_void,qo: *mut c_void,ko: *mut c_void,
        tokens: i64,heads: c_int,qeps: f32,keps: f32,stream: Stream) -> c_int;
    fn h3_pw_rms_mod_bf16(x: *const c_void, weight: *const c_void, scale: *const c_void,
        shift: *const c_void, index: *const i64, out: *mut c_void, rows: i64, cols: c_int,
        table_stride: c_int, eps: f32, stream: Stream) -> c_int;
    fn h3_pw_gated_rms_mod_bf16(hidden: *mut c_void, branch: *const c_void, gate: *const c_void,
        weight: *const c_void, scale: *const c_void, shift: *const c_void, index: *const i64,
        out: *mut c_void, rows: i64, cols: c_int, table_stride: c_int, eps: f32, stream: Stream) -> c_int;
    fn h3_pw_gated_bf16(hidden: *const c_void, branch: *const c_void, gate: *const c_void,
        index: *const i64, out: *mut c_void, rows: i64, cols: c_int, table_stride: c_int,
        stream: Stream) -> c_int;
    fn h3_rms_bf16(x: *const c_void, weight: *const c_void, out: *mut c_void,
        rows: i64, cols: c_int, eps: f32, stream: Stream) -> c_int;
    fn h3_modulate_bf16(x: *mut c_void, scale: *const c_void, shift: *const c_void,
        index: *const i64, rows: i64, cols: c_int, table_stride: c_int, stream: Stream) -> c_int;
    fn h3_gated_bf16(x: *const c_void, branch: *const c_void, gate: *const c_void,
        index: *const i64, out: *mut c_void, rows: i64, cols: c_int, table_stride: c_int, stream: Stream) -> c_int;
    fn h3_swiglu_bf16(x: *const c_void, out: *mut c_void, rows: i64, cols: c_int, stream: Stream) -> c_int;
    fn h3_opt_swiglu_bf16(x: *const c_void, out: *mut c_void, rows: i64, cols: c_int, stream: Stream) -> c_int;
    fn h3_qk_rope_opt_dispatch_v1(q: *const c_void,k: *const c_void,qw: *const c_void,kw: *const c_void,
        cos: *const c_void,sin: *const c_void,qn: *mut c_void,kn: *mut c_void,qo: *mut c_void,ko: *mut c_void,
        tokens: i64,heads: c_int,qeps: f32,keps: f32,stream: Stream) -> c_int;
    fn h3_sglang_qk_rope_v1(q: *const c_void,k: *const c_void,qw: *const c_void,kw: *const c_void,
        cos: *const c_void,sin: *const c_void,qn: *mut c_void,kn: *mut c_void,qo: *mut c_void,ko: *mut c_void,
        tokens: i64,heads: c_int,qeps: f32,keps: f32,stream: Stream) -> c_int;
    fn h3_sglang_qk_rope_v2(q: *const c_void,k: *const c_void,qw: *const c_void,kw: *const c_void,
        cos: *const c_void,sin: *const c_void,qn: *mut c_void,kn: *mut c_void,qo: *mut c_void,ko: *mut c_void,
        tokens: i64,heads: c_int,qeps: f32,keps: f32,stream: Stream) -> c_int;
    fn h3_sglang_qkv_rope_v2(qkv: *const c_void,qw: *const c_void,kw: *const c_void,
        cos: *const c_void,sin: *const c_void,qn: *mut c_void,kn: *mut c_void,qo: *mut c_void,ko: *mut c_void,vo: *mut c_void,
        tokens: i64,heads: c_int,qeps: f32,keps: f32,stream: Stream) -> c_int;
    fn h3_rope_bf16(x: *const c_void, cos: *const c_void, sin: *const c_void, out: *mut c_void,
        tokens: i64, heads: c_int, rotary: c_int, stream: Stream) -> c_int;
    fn h3_add_bf16(x: *const c_void, y: *const c_void, out: *mut c_void, n: i64, stream: Stream) -> c_int;
    fn h3_rows_copy(x: *const c_void, index: *const i64, out: *mut c_void, rows: i64,
        cols: c_int, dtype_bytes: c_int, scatter: c_int, stream: Stream) -> c_int;
    fn h3_cast(x: *const c_void, out: *mut c_void, n: i64, to_float: c_int, stream: Stream) -> c_int;
    fn h3_silu_f32(x: *const f32, out: *mut c_void, n: i64, to_bf16: c_int, stream: Stream) -> c_int;
    fn h3_euler_f32(sample: *const f32, velocity: *const f32, out: *mut f32,
        n: i64, sigma_from_timestep: f32, ratio: f32, stream: Stream) -> c_int;
    fn h3_qkv_split_bf16(x: *const c_void, q: *mut c_void, k: *mut c_void, v: *mut c_void,
        tokens: i64, inner: c_int, stream: Stream) -> c_int;
    fn h3_rotary_tables(positions: *const f64, inv_freq: *const f32, cos_out: *mut f32,
        sin_out: *mut f32, tokens: i64, stream: Stream) -> c_int;
    fn dlopen(name: *const c_char, flags: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    fn dlclose(handle: *mut c_void) -> c_int;
    fn dlerror() -> *const c_char;
}

use crate::bsa_pack::CPP_CALLS as BSA_CPP_CALLS;
/// The kernel that actually ran each sparse call, from the bridge's own counters. `cpp_calls`
/// counts the served FlashInfer kernel and `exact_calls` the exact CuTe kernel; the `cute_*`
/// fields describe the retired inexact CuTe path and stay off.
fn sparse_attention_report() -> Result<Value> {
    let [served, exact] = crate::bsa_pack::kernel_counts()?;
    let calls = BSA_CPP_CALLS.load(std::sync::atomic::Ordering::Relaxed);
    if served.checked_add(exact) != Some(calls) {
        return Err(format!("BSA kernel counters served={served} exact={exact} differ from {calls} sparse calls").into());
    }
    let exact_enabled = crate::bsa_pack::exact_selected();
    Ok(json!({"cute_available": false, "cute_enabled": false, "cute_calls": 0,
        "cpp_calls": served, "exact_available": crate::bsa_pack::exact_available(),
        "exact_enabled": exact_enabled, "exact_calls": exact,
        "kernel": if exact_enabled {"cute-dsl-blk64-exact"} else {"flashinfer-0.7.0-sm100-blk64-aot"}}))
}
fn cuda(code: c_int) -> Result<()> {
    if code == 0 { return Ok(()); }
    Err(format!("CUDA {code}: {}", unsafe { CStr::from_ptr(cudaGetErrorString(code)) }.to_string_lossy()).into())
}
fn blas(code: c_int) -> Result<()> {
    if code == 0 { return Ok(()); }
    Err(format!("BLAS {code}: {}", unsafe { CStr::from_ptr(h3_linear_error()) }.to_string_lossy()).into())
}

#[derive(Clone, Copy, Debug)]
struct Buffer { ptr: *mut c_void, bytes: usize }
impl Buffer {
    fn prefix(self, bytes: usize) -> Result<Self> { self.slice(0, bytes) }
    fn slice(self, offset: usize, bytes: usize) -> Result<Self> {
        if offset.checked_add(bytes).ok_or("pointer overflow")? > self.bytes { return Err("device view exceeds allocation".into()); }
        Ok(Self { ptr: (self.ptr as usize + offset) as *mut c_void, bytes })
    }
}
include!("request_pool.rs");
struct Arena { stream: Stream, allocations: Vec<Buffer>, pool:Option<std::rc::Rc<RequestPool>>, cp_watchdog:Option<std::rc::Rc<CpGroup>> }
impl Arena {
    fn new() -> Result<Self> {Self::new_with_pool(None)}
    fn new_with_pool(pool:Option<std::rc::Rc<RequestPool>>)->Result<Self>{
        let _trace=crate::startup_trace::Span::new("cuda_context","dit-arena",0);
        cuda(unsafe { h3_check_device() })?;
        let mut stream = ptr::null_mut();
        crate::startup_trace::measure("dit_cuda_stage","stream",0,||cuda(unsafe { cudaStreamCreateWithFlags(&mut stream, 1) }))?;
        Ok(Self { stream, allocations: vec![],pool,cp_watchdog:None })
    }
    fn alloc(&mut self, bytes: usize) -> Result<Buffer> {
        if bytes == 0 { return Err("zero-sized device allocation".into()); }
        if let Some(buffer)=self.pool.as_ref().and_then(|pool|pool.take(bytes)){
            // Track the actual allocation for release/accounting; kernels and
            // copies see only the requested extent, never stale capacity bytes.
            self.allocations.push(buffer);return buffer.slice(0,bytes);
        }
        let mut ptr = ptr::null_mut();
        let mut result=unsafe { cudaMalloc(&mut ptr, bytes) };
        if result==2 && self.pool.as_ref().is_some_and(|p|p.cached_bytes.get()>0){
            // Reclaim inactive activation storage before reporting real OOM.
            // Do not retry other CUDA errors or retain reconstructed weights.
            unsafe{cudaGetLastError();}self.pool.as_ref().unwrap().clear();
            result=unsafe{cudaMalloc(&mut ptr,bytes)};
        }
        if result!=0 {
            let(mut free,mut total)=(0usize,0usize);
            unsafe extern "C" {fn cudaMemGetInfo(free:*mut usize,total:*mut usize)->c_int;}
            let memory_status=unsafe{cudaMemGetInfo(&mut free,&mut total)};
            let owned=self.allocations.iter().map(|b|b.bytes).sum::<usize>();
            return Err(format!("device allocation failed: requested={bytes} arena={owned} free={free} total={total} memory_status={memory_status}: {}",cuda(result).unwrap_err()).into());
        }
        let buffer = Buffer { ptr, bytes };
        self.allocations.push(buffer);
        Ok(buffer)
    }
    /// blk64 TMA reads a complete final KV block. Storage covers that block;
    /// the logical length and block_sizes still exclude its zero guard rows.
    fn sparse_input(&mut self,rows:usize,width:usize)->Result<Buffer> {
        let padded=crate::subblock::storage_rows(rows)?;
        let logical=rows.checked_mul(width).and_then(|n|n.checked_mul(2)).ok_or("sparse input size overflow")?;
        let bytes=padded.checked_mul(width).and_then(|n|n.checked_mul(2)).ok_or("sparse input size overflow")?;
        let buffer=self.alloc(bytes)?;
        if bytes>logical {
            let guard=buffer.slice(logical,bytes-logical)?;
            cuda(unsafe{cudaMemsetAsync(guard.ptr,0,guard.bytes,self.stream)})?;
        }
        Ok(buffer)
    }
    fn alloc_model_tensors(&mut self,sizes:&[usize])->Result<Vec<Buffer>> {
        if sizes.is_empty() || !crate::weight_loader::flag("H3_MODEL_LOAD_SLAB")? {
            return sizes.iter().map(|&bytes|self.alloc(bytes)).collect();
        }
        if self.pool.is_some(){return Err("model allocation must not use the request pool".into());}
        let (offsets,bytes)=crate::weight_loader::tensor_allocation_layout(sizes)?;
        let owned=self.alloc(bytes)?;
        // The arena owns one allocation; tensor views retain their original
        // lengths and cannot free individual portions of the model storage.
        offsets.iter().zip(sizes).map(|(&offset,&bytes)|owned.slice(offset,bytes)).collect()
    }
    fn sync(&self) -> Result<()> { if let Some(group)=&self.cp_watchdog {cp_check(unsafe{h3_comm_synchronize(group.handle)})}else{cuda(unsafe { cudaStreamSynchronize(self.stream) })} }
    fn upload(&mut self, data: &[u8]) -> Result<Buffer> {
        let buffer = self.alloc(data.len())?;
        self.write(buffer, data)?;
        Ok(buffer)
    }
    fn write(&self, buffer: Buffer, data: &[u8]) -> Result<()> {
        if data.len() != buffer.bytes { return Err("H2D byte count mismatch".into()); }
        let result = cuda(unsafe { cudaMemcpyAsync(buffer.ptr, data.as_ptr().cast(), data.len(), 1, self.stream) });
        // Even a launch/copy failure must not let pageable host memory die early.
        let synchronized = self.sync();
        result?; synchronized
    }
    fn download(&self, buffer: Buffer) -> Result<Vec<u8>> {
        let mut data = vec![0; buffer.bytes];
        let result = cuda(unsafe { cudaMemcpyAsync(data.as_mut_ptr().cast(), buffer.ptr, buffer.bytes, 2, self.stream) });
        let synchronized = self.sync();
        result?; synchronized?; Ok(data)
    }
    fn copy(&self, dst: Buffer, src: Buffer) -> Result<()> {
        if dst.bytes != src.bytes { return Err("D2D byte count mismatch".into()); }
        cuda(unsafe { cudaMemcpyAsync(dst.ptr, src.ptr, dst.bytes, 3, self.stream) })
    }
}
impl Drop for Arena {
    fn drop(&mut self) {
        let synchronized=self.sync().is_ok();
        for &buffer in &self.allocations {
            if let Some(pool)=&self.pool {if synchronized {pool.release(buffer);continue;}}
            unsafe { cudaFree(buffer.ptr); }
        }
        unsafe { cudaStreamDestroy(self.stream); }
    }
}

type DenseWorkspace = unsafe extern "C" fn(c_int, c_int, *mut usize) -> c_int;
type DenseForward = unsafe extern "C" fn(*const c_void, *const c_void, *const c_void, *mut c_void,
    *mut c_void, usize, c_int, c_int, f32, Stream) -> c_int;
type DenseWorkspaceV2 = unsafe extern "C" fn(c_int, c_int, c_int, *mut usize) -> c_int;
type DenseForwardV2 = unsafe extern "C" fn(*const c_void, *const c_void, *const c_void, *mut c_void,
    *mut c_void, usize, c_int, c_int, c_int, f32, Stream) -> c_int;
type DenseError = unsafe extern "C" fn() -> *const c_char;
type DenseCleanup = unsafe extern "C" fn() -> c_int;
struct Dense {
    library: *mut c_void, workspace: DenseWorkspace, forward: DenseForward,
    workspace_v2: Option<DenseWorkspaceV2>, forward_v2: Option<DenseForwardV2>,
    error: Option<DenseError>, cleanup: DenseCleanup, cleaned: bool,
}
impl Dense {
    fn load(path: &Path) -> Result<Self> {
        let filename = CString::new(path.to_str().ok_or("non-UTF8 dense library path")?)?;
        let library = unsafe { dlopen(filename.as_ptr(), 2) };
        if library.is_null() {
            let error = unsafe { dlerror() };
            return Err(format!("load dense library: {}", if error.is_null() { "unknown".into() }
                else { unsafe { CStr::from_ptr(error) }.to_string_lossy() }).into());
        }
        let symbol = |name: &str| unsafe { dlsym(library, CString::new(name).unwrap().as_ptr()) };
        let (workspace, forward, error, cleanup) = (symbol("h3_dense_workspace_bytes"), symbol("h3_dense_forward"),
            symbol("h3_dense_error"), symbol("h3_dense_cleanup"));
        if workspace.is_null() || forward.is_null() || cleanup.is_null() {
            unsafe { dlclose(library); }
            return Err("dense library does not implement the frozen C ABI".into());
        }
        let (workspace_v2,forward_v2)=(symbol("h3_dense_workspace_bytes_v2"),symbol("h3_dense_forward_v2"));
        if workspace_v2.is_null()!=forward_v2.is_null() {
            unsafe{dlclose(library);}return Err("incomplete dense v2 ABI".into());
        }
        Ok(Self { library, workspace: unsafe { std::mem::transmute::<*mut c_void, DenseWorkspace>(workspace) },
            forward: unsafe { std::mem::transmute::<*mut c_void, DenseForward>(forward) },
            workspace_v2: if workspace_v2.is_null(){None}else{Some(unsafe{std::mem::transmute::<*mut c_void,DenseWorkspaceV2>(workspace_v2)})},
            forward_v2: if forward_v2.is_null(){None}else{Some(unsafe{std::mem::transmute::<*mut c_void,DenseForwardV2>(forward_v2)})},
            error: if error.is_null() { None } else { Some(unsafe { std::mem::transmute::<*mut c_void, DenseError>(error) }) },
            cleanup: unsafe { std::mem::transmute::<*mut c_void, DenseCleanup>(cleanup) }, cleaned: false })
    }
    fn check(&self, code: c_int) -> Result<()> {
        if code == 0 { return Ok(()); }
        let detail = self.error.map(|function| {
            let text = unsafe { function() };
            if text.is_null() { "unknown".to_owned() } else { unsafe { CStr::from_ptr(text) }.to_string_lossy().into_owned() }
        }).unwrap_or_default();
        Err(format!("dense backend {code}: {detail}").into())
    }
    fn size(&self, tokens: usize, layout: c_int) -> Result<usize> {
        let mut bytes = 0;
        if let Some(function)=self.workspace_v2 {
            self.check(unsafe { function(tokens as c_int, HEADS as c_int, layout, &mut bytes) })?;
        } else {
            if ![(63510,0),(2456,1)].contains(&(tokens,layout)) {
                return Err("new shapes require dense explicit-layout v2 ABI".into());
            }
            self.check(unsafe { (self.workspace)(tokens as c_int, HEADS as c_int, &mut bytes) })?;
        }
        Ok(bytes.max(1))
    }
    fn run(&self,q:Buffer,k:Buffer,v:Buffer,out:Buffer,workspace:Buffer,tokens:usize,
           layout:c_int,scale:f32,stream:Stream)->Result<()> {
        let code=if let Some(function)=self.forward_v2 {
            unsafe{function(q.ptr,k.ptr,v.ptr,out.ptr,workspace.ptr,workspace.bytes,
                tokens as c_int,HEADS as c_int,layout,scale,stream)}
        } else {
            if ![(63510,0),(2456,1)].contains(&(tokens,layout)) {
                return Err("new shapes require dense explicit-layout v2 ABI".into());
            }
            unsafe{(self.forward)(q.ptr,k.ptr,v.ptr,out.ptr,workspace.ptr,workspace.bytes,
                tokens as c_int,HEADS as c_int,scale,stream)}
        };
        self.check(code)
    }
    fn cleanup(&mut self) -> Result<()> {
        if !self.cleaned {
            self.check(unsafe { (self.cleanup)() })?;
            self.cleaned = true;
        }
        Ok(())
    }
}
impl Drop for Dense {
    fn drop(&mut self) {
        let _ = self.cleanup();
        unsafe { dlclose(self.library); }
    }
}

#[derive(Clone)]
struct Tensor { spec: TensorSpec, buffer: Buffer }
include!("sharded_native.rs");
include!("timestep_native.rs");
#[derive(Clone, Copy)]
struct Step {
    indices: Buffer, adaln_indices: Buffer, final_mod: Buffer, cursor: usize,
    video_t: f32, audio_t: f32, video_ratio: f32, audio_ratio: f32, next_video_sigma: f32,
}
struct TraceReference {
    root: PathBuf, layer_crc32: Vec<u32>,
    gold_manifest_sha256: Value,
}

include!("router_cache.rs");
pub struct WarmLoadCache {dense:std::rc::Rc<Dense>,pool:std::rc::Rc<RequestPool>}
pub struct ResidentModel {
    // A recipe view borrows tensor pointers; keep their allocating arena alive.
    weight_owner:Option<std::rc::Rc<ResidentModel>>,
    cp_group:Option<std::rc::Rc<CpGroup>>,
    routers:RouterCache,
    dense:std::rc::Rc<Dense>,
    pool:std::rc::Rc<RequestPool>,
    arena: Arena,
    root: PathBuf,
    weights: HashMap<String,Tensor>,
    buffers: HashMap<String,Tensor>,
    final_mods: Vec<Tensor>,
    plan_final_mods: HashMap<usize,Vec<Tensor>>,
    zero_final_mod: Option<Buffer>,
    sharded: Option<std::rc::Rc<ShardedWeights>>,
    load_seconds: f64,
}
impl ResidentModel {
    pub fn load_seconds(&self)->f64{self.load_seconds}
    pub fn verify_loaded_payload(&self)->Result<Value>{
        // Sharded tensor views point into a reusable gather workspace, not persistent
        // full weights. Audit the resident rank's immutable shards instead.
        if let Some(sharded)=&self.sharded{return sharded.verify_resident_payload(self.arena.stream);}
        let config=crate::weight_loader::Config::from_env()?.ok_or("GPU verification requires pipeline")?;
        let verify=crate::weight_loader::device::Upload::new(self.arena.stream,config)?;
        for tensor in self.weights.values(){unsafe{verify.verify(tensor.buffer.ptr,tensor.spec.bytes,
            tensor.spec.sha256.as_deref().ok_or("fused weight SHA missing")?)}?;}
        Ok(json!({"expected_fused_payload_sha_verified":true,"tensors":self.weights.len()}))
    }
}

impl ResidentModel {
    pub fn warm_load_cache(&self)->WarmLoadCache{WarmLoadCache{dense:self.dense.clone(),pool:self.pool.clone()}}
    pub fn has_request_cache(&self)->bool{
        self.sharded.as_ref().is_some_and(|s|s.workspace.mode.cache() && s.cache.borrow().is_some())
    }
    pub fn with_compact_layout<T>(&self,rebuild:impl FnOnce()->Result<T>)->Result<T>{
        let _pause=crate::allocation_policy::Pause::new(&self.pool.enabled);
        self.pool.clear();rebuild()
    }
}

impl ResidentModel {
    fn load(root:&Path,manifest:&Value,manifest_bytes:&[u8],dense:std::rc::Rc<Dense>,
            request_pool:std::rc::Rc<RequestPool>,shared:Option<&Self>,gather_capacity:usize,
            files:&mut Vec<String>,prefetched:Option<crate::weight_loader::ReaderPool>,
            combined:Option<bool>,
            weight_owner:Option<std::rc::Rc<Self>>,shard_readers:Option<ShardReaders>,
            pending:Option<PendingGroup>)->Result<std::rc::Rc<Self>> {
        // Unsharded, `prefetched` reads the weights; sharded, the shards have their own readers
        // and `prefetched` reads the buffers every rank keeps whole.
        let mut prefetched=prefetched;
        let recipe=crate::recipe::Recipe::from_model(manifest)?;
        crate::integrity::admit_base(manifest)?;
        let combined_payload=if let Some(value)=combined {value}else{combined_payload_enabled(manifest)?};
        let model_started = Instant::now();
        let owner=shared.or(weight_owner.as_deref());
        if owner.is_some() && pending.is_some(){return Err("a shared model already has its communicator".into());}
        let mut cp_group=if let Some(other)=owner{other.cp_group.clone()}else if pending.is_some(){None}else{CpGroup::from_environment()?};
        let mut model_arena=Arena::new()?;
        let weights_root=if let Some(shared)=manifest.get("shared_weights") {
            let path=PathBuf::from(shared["root"].as_str().ok_or("shared weights root missing")?);
            let bytes=crate::startup_trace::read(path.join("manifest.json"))?;
            if format!("{:x}",crate::startup_trace::measure("sha256","",(&bytes).len(),||Sha256::digest(&bytes)))!=shared["manifest_sha256"].as_str().ok_or("shared manifest SHA missing")? {
                return Err("shared weights manifest identity changed".into());
            }
            let source:Value=serde_json::from_slice(&bytes)?;
            if source["status"]!="complete" || source["weights"]!=manifest["weights"] {
                return Err("shared model weight specifications changed".into());
            }
            path
        }else{root.to_path_buf()};
        let (sharded,weights,preloaded_buffers)=if let Some(owner)=&weight_owner {
            (owner.sharded.clone(),owner.weights.clone(),None)
        }else if manifest.get("weight_sharding").is_some() {
            let group=match pending {Some(pending)=>GroupSource::Pending(pending),
                None=>GroupSource::Ready(cp_group.clone().ok_or("weight sharding requires a communicator")?)};
            let (sharded,weights)=ShardedWeights::load(&mut model_arena,manifest,manifest_bytes,group,
                shared.and_then(|m|m.sharded.as_ref().map(|s|s.workspace.clone())),gather_capacity,files,shard_readers)?;
            cp_group=Some(sharded.group.clone());
            (Some(std::rc::Rc::new(sharded)),weights,None)
        }else if combined_payload {
            let (weights,buffers)=load_weights_and_buffers(&mut model_arena,&weights_root,root,
                &manifest["weights"],&manifest["buffers"],files,prefetched.take())?;
            (None,weights,Some(buffers))
        }else {
            (None,load_model_tensors(&mut model_arena,&weights_root,&manifest["weights"],files,"weights",prefetched.take())?,None)
        };
        let buffers=if let Some(buffers)=preloaded_buffers{buffers}else{
            load_model_tensors(&mut model_arena,root,&manifest["buffers"],files,"buffers",prefetched.take())?};
        let mut final_mods=Vec::new();
        for entry in manifest["steps"].as_array().ok_or("missing steps")? {
            final_mods.push(load_model_tensor(&mut model_arena,root,&entry["final_modulation"],files)?);
        }
        let mut plan_final_mods=HashMap::new();
        if let Some(entries)=manifest.get("final_modulation_plans") {
            for entry in entries.as_array().ok_or("final modulation plans must be an array")? {
                let index=entry["plan_index"].as_u64().ok_or("final modulation plan index missing")? as usize;
                let fingerprint=manifest["adaln"]["plans"].as_array().ok_or("AdaLN plans missing")?.iter()
                    .find(|p|p["plan_index"].as_u64()==Some(index as u64)).ok_or("final modulation plan is not installed")?;
                let values=entry["steps"].as_array().ok_or("final modulation plan steps missing")?;
                if values.len()!=recipe.nfe{return Err("final modulation plan length mismatch".into());}
                let mut tensors=Vec::new();
                for (i,spec) in values.iter().enumerate() {
                    let tensor=load_model_tensor(&mut model_arena,root,spec,files)?;
                    let rows=fingerprint["fingerprint"][i].as_array().ok_or("AdaLN fingerprint missing")?.len();
                    if tensor.spec.dtype!="torch.bfloat16" || tensor.spec.shape!=[rows,2*HIDDEN] {
                        return Err("plan-specific final modulation shape mismatch".into());
                    }
                    tensors.push(tensor);
                }
                if plan_final_mods.insert(index,tensors).is_some(){return Err("duplicate final modulation plan".into());}
            }
        }
        let zero_final_mod=if recipe.workflow=="fl2va" {
            if let Some(sharded)=&sharded{sharded.materialize("global",model_arena.stream)?;}
            Some(zero_timestep_final_mod(&mut model_arena,&weights)?)
        }else{None};
        if let Some(sharded)=&sharded{sharded.finish_request(model_arena.stream)?;}
        model_arena.sync()?;
        if sharded.is_some(){cp_group.as_ref().unwrap().bind(ptr::null_mut())?;}
        Ok(std::rc::Rc::new(ResidentModel{weight_owner,cp_group,routers:RouterCache::new(),dense:dense.clone(),pool:request_pool,arena:model_arena,root:root.to_owned(),weights,buffers,final_mods,plan_final_mods,zero_final_mod,sharded,
            load_seconds:model_started.elapsed().as_secs_f64()}))
    }
    pub fn preload(root:&Path,dense_path:&Path,shared:Option<std::rc::Rc<Self>>,gather_capacity:usize)
        ->Result<std::rc::Rc<Self>> {Self::preload_cached(root,dense_path,shared,gather_capacity,None)}
    pub fn preload_cached(root:&Path,dense_path:&Path,shared:Option<std::rc::Rc<Self>>,gather_capacity:usize,
        warm:Option<WarmLoadCache>)->Result<std::rc::Rc<Self>> {
        let _trace=crate::startup_trace::Span::new("dit_model_preload",root.to_string_lossy(),0);
        let bytes=crate::startup_trace::read(root.join("manifest.json"))?;
        let mut manifest:Value=serde_json::from_slice(&bytes)?;
        // A CP rank's shards come from its package, not the model manifest.
        if let Some(sharding)=crate::weight_shards::configured()? {
            if manifest.get("weight_sharding").is_some(){return Err("the model manifest already names weight shards".into());}
            manifest["weight_sharding"]=sharding;
        }
        if manifest["schema"]!="h3-native-model-v1" || manifest["status"]!="complete" ||
            manifest["linear_backend"]["backend"]["allow_tf32"]!=false ||
            manifest["linear_backend"]["backend"]["allow_bf16_reduced_precision_reduction"]!=true {
            return Err("catalog requires complete static models with accepted GEMM flags".into());
        }
        crate::integrity::admit_base(&manifest)?;
        let combined=combined_payload_enabled(&manifest)?;
        let source=manifest.get("shared_weights").and_then(|v|v["root"].as_str()).map(PathBuf::from).unwrap_or_else(||root.to_path_buf());
        let init=crate::weight_loader::flag("H3_MODEL_LOAD_INIT_PREFETCH")?;
        let sharded=manifest.get("weight_sharding").is_some();
        // A rank streaming its shards forms its communicator on a helper thread from here on,
        // through the reads and the upload; it does not depend on either.
        let pending=if init && sharded && shared.is_none() {PendingGroup::start()?}else{None};
        let shard_readers=if init && sharded {ShardReaders::start(&manifest,&source)?}else{None};
        let prefetched=if init && sharded {
            if let Some(config)=crate::weight_loader::Config::from_env()? {
                let jobs=manifest["buffers"].as_object().ok_or("model buffer map missing")?.values().map(|v|{
                    let spec=TensorSpec::parse(v).map_err(|e|e.to_string())?;
                    crate::weight_loader::Job::new(root,&spec.file,spec.bytes,spec.sha256)
                }).collect::<std::result::Result<Vec<_>,String>>()?;
                Some(crate::weight_loader::ReaderPool::start(jobs,config)?)
            }else{None}
        }else if init {
            if let Some(config)=crate::weight_loader::Config::from_env()? {
                let jobs=if combined {
                    crate::model_payload::Groups::parse(&manifest["weights"],&manifest["buffers"])?.jobs(&source,root)?
                }else{manifest["weights"].as_object().ok_or("model weight map missing")?.values().map(|v|{
                    let spec=TensorSpec::parse(v).map_err(|e|e.to_string())?;
                    crate::weight_loader::Job::new(&source,&spec.file,spec.bytes,spec.sha256)
                }).collect::<std::result::Result<Vec<_>,String>>()?};
                let cache_hit=!combined && crate::host_cache::enabled()? && crate::host_cache::can_upload(&jobs)?;
                if cache_hit {None}else{Some(crate::weight_loader::ReaderPool::start(jobs,config)?)}
            }else{None}
        }else{None};
        cp_select_device_from_environment()?;
        let dense=if let Some(other)=&shared{other.dense.clone()}else if let Some(warm)=&warm{warm.dense.clone()}else{std::rc::Rc::new(Dense::load(dense_path)?)};
        let pool=shared.as_ref().map(|m|m.pool.clone()).or_else(||warm.as_ref().map(|w|w.pool.clone())).unwrap_or_else(||std::rc::Rc::new(RequestPool::new()));
        Self::load(root,&manifest,&bytes,dense,pool,shared.as_deref(),gather_capacity,&mut Vec::new(),prefetched,Some(combined),None,
            shard_readers,pending)
    }
    pub fn weight_identity(&self)->String {
        let mut entries=self.weights.iter().map(|(name,t)|(name,t.buffer.ptr as usize,t.buffer.bytes)).collect::<Vec<_>>();
        entries.sort();format!("{:x}",Sha256::digest(serde_json::to_vec(&entries).unwrap()))
    }
    pub fn ownership_report(&self)->Result<Value> {
        Ok(json!({"weights_identity_sha256":self.identity()?,"model_load_count":1,"sparse_attention":sparse_attention_report()?,
            "shared_weight_view":self.weight_owner.is_some(),"weight_tensor_identity_sha256":self.weight_identity(),
            "model_arena_bytes":self.arena.allocations.iter().map(|b|b.bytes).sum::<usize>(),
            "weight_sharding":self.sharded.as_ref().map_or_else(||json!({"enabled":false}),|s|s.report())}))
    }
}

impl ResidentModel {
    pub fn identity(&self)->Result<String> {
        let mut entries=Vec::new();
        if let Some(sharded)=&self.sharded {
            for (name,buffer) in &sharded.shards {
                entries.push((format!("shards/{}/{name}",sharded.manifest_sha256),buffer.ptr as usize,buffer.bytes));
            }
        }
        for (kind,tensors) in [("weights",&self.weights),("buffers",&self.buffers)] {
            for (name,tensor) in tensors{entries.push((format!("{kind}/{name}"),tensor.buffer.ptr as usize,tensor.buffer.bytes));}
        }
        for (i,tensor) in self.final_mods.iter().enumerate(){entries.push((format!("final_mod/{i}"),tensor.buffer.ptr as usize,tensor.buffer.bytes));}
        for (plan,tensors) in &self.plan_final_mods {
            for (i,tensor) in tensors.iter().enumerate(){entries.push((format!("final_mod/plan{plan}/{i}"),tensor.buffer.ptr as usize,tensor.buffer.bytes));}
        }
        if let Some(zero)=self.zero_final_mod {
            entries.push(("final_mod/zero_single_row".into(),zero.ptr as usize,zero.bytes));
        }
        entries.sort();Ok(format!("{:x}",Sha256::digest(serde_json::to_vec(&entries)?)))
    }
}

pub struct Runtime {
    cp:Option<CpState>,
    resident_model: std::rc::Rc<ResidentModel>,
    arena: Arena,
    linear: *mut c_void,
    dense: std::rc::Rc<Dense>,
    router: Option<std::rc::Rc<Program>>,
    route_buffers: HashMap<String, u64>,
    weights: HashMap<String, Tensor>,
    buffers: HashMap<String, Tensor>,
    scratch: HashMap<&'static str, Buffer>,
    steps: Vec<Step>,
    execution: crate::execution::Execution,
    long_request: Option<Box<PreparedLong>>,
    long_sparse_calls: Option<u64>,
    router_root: PathBuf,
    dense_path: PathBuf,
    manifest: Value,
    root: PathBuf,
    output: PathBuf,
    seq: usize, nv: usize, na: usize, nt: usize,
    condition_video: usize, condition_audio: usize,
    dense_audio_range: Option<(usize,usize)>,
    numerics: crate::numerics::Numerics,
    latent_video: Buffer, latent_audio: Buffer, conditioning: Buffer,
    video_indices: Buffer, audio_indices: Buffer, text_indices: Buffer,
    text_indices_host: Vec<i64>,
    cos: Buffer, sin: Buffer,
    block_sizes: Buffer,
    query_block_mask: Buffer, kept: usize,
    native_input_files: Vec<String>,
    native_request: Value,
    validate_reference: bool,
    resident_layout_inputs: HashMap<String,Vec<u8>>,
    load_seconds: f64, model_load_seconds: f64,
}

fn load(arena: &mut Arena, root: &Path, value: &Value, files: &mut Vec<String>) -> Result<Tensor> {
    let spec = TensorSpec::parse(value)?;
    let data = spec.read(root)?;
    let buffer = arena.upload(&data)?;
    files.push(spec.file.clone());
    Ok(Tensor { spec, buffer })
}

fn load_model_tensor(arena:&mut Arena,root:&Path,value:&Value,files:&mut Vec<String>)->Result<Tensor> {
    if !crate::host_cache::enabled()? {return load_model_tensor_uncached(arena,root,value,files);}
    if crate::host_cache::deferred()? {
        let spec=TensorSpec::parse(value)?;
        let job=crate::weight_loader::Job::new(root,&spec.file,spec.bytes,spec.sha256.clone())?;
        if !crate::host_cache::can_upload(&[job])? {return load_model_tensor_uncached(arena,root,value,files);}
    }
    let spec=TensorSpec::parse(value)?;
    let job=crate::weight_loader::Job::new(root,&spec.file,spec.bytes,spec.sha256.clone())?;
    let buffer=arena.alloc(spec.bytes)?;
    unsafe {crate::host_cache::upload(&[job],&[buffer.ptr],arena.stream,false)}?;
    files.push(spec.file.clone());
    Ok(Tensor {spec,buffer})
}

fn load_model_tensor_uncached(arena:&mut Arena,root:&Path,value:&Value,files:&mut Vec<String>)->Result<Tensor>{
    let spec=TensorSpec::parse(value)?;
    let data=if crate::weight_loader::light::async_enabled()? {
        let data=crate::startup_trace::read(root.join(&spec.file))?;
        if data.len()!=spec.bytes{return Err("model payload length mismatch".into());}
        crate::weight_loader::light::verify_model_bytes(&data,spec.sha256.as_deref().ok_or("model payload SHA missing")?)?;
        data
    }else{spec.read(root)?};
    let buffer=arena.upload(&data)?;files.push(spec.file.clone());Ok(Tensor{spec,buffer})
}


fn combined_payload_enabled(manifest:&Value)->Result<bool> {
    if !crate::weight_loader::flag("H3_DIT_LOAD_AUX_WITH_WEIGHTS")? {return Ok(false);}
    // A rank's shards and its buffers stream from separate readers.
    if manifest.get("weight_sharding").is_some() {return Ok(false);}
    if crate::weight_loader::Config::from_env()?.is_none() {
        return Err("combined DiT payload requires streaming".into());
    }
    // Cached weights use their existing bounded upload path.
    Ok(!crate::host_cache::enabled()?)
}

fn load_weights_and_buffers(arena:&mut Arena,weights_root:&Path,buffers_root:&Path,
    weights:&Value,buffers:&Value,files:&mut Vec<String>,prefetched:Option<crate::weight_loader::ReaderPool>)
    ->Result<(HashMap<String,Tensor>,HashMap<String,Tensor>)> {
    let config=crate::weight_loader::Config::from_env()?.ok_or("combined DiT payload requires streaming")?;
    let groups=crate::model_payload::Groups::parse(weights,buffers)?;
    let mut pool=if let Some(pool)=prefetched{pool}else{
        let jobs=groups.jobs(weights_root,buffers_root)?;
        crate::weight_loader::ReaderPool::start(jobs,config)?};
    let mut upload=crate::weight_loader::device::Upload::new(arena.stream,config)?;
    // Preserve two allocation groups and the original 256-byte tensor alignment.
    // Only IO scheduling changes; compute begins after the complete stream drain.
    let weight_buffers=arena.alloc_model_tensors(&groups.weights.iter().map(|(_,s)|s.bytes).collect::<Vec<_>>())?;
    let auxiliary_buffers=arena.alloc_model_tensors(&groups.buffers.iter().map(|(_,s)|s.bytes).collect::<Vec<_>>())?;
    let destinations=weight_buffers.iter().chain(&auxiliary_buffers).collect::<Vec<_>>();
    unsafe{pool.consume_upload(&mut upload,|i,offset|destinations[i].ptr.cast::<u8>().add(offset).cast())}?;
    upload.finish()?;
    let verify=std::env::var("H3_MODEL_LOAD_VERIFY_GPU").as_deref()==Ok("1");
    let specs=groups.weights.iter().chain(&groups.buffers).map(|(_,s)|s);
    if verify {for (buffer,spec) in destinations.iter().zip(specs) {
        unsafe{upload.verify(buffer.ptr,buffer.bytes,spec.sha256.as_deref().ok_or("GPU verification requires SHA256")?)}?;
    }}
    let weights=groups.weights.into_iter().zip(weight_buffers).map(|((name,spec),buffer)|{
        files.push(spec.file.clone());(name,Tensor{spec,buffer})}).collect::<HashMap<_,_>>();
    let buffers=groups.buffers.into_iter().zip(auxiliary_buffers).map(|((name,spec),buffer)|{
        files.push(spec.file.clone());(name,Tensor{spec,buffer})}).collect::<HashMap<_,_>>();
    eprintln!("native weights+buffers streaming load: {} weight tensors, {} buffers, config={config:?}, GPU_SHA256_verified={verify}",weights.len(),buffers.len());
    Ok((weights,buffers))
}

fn load_model_tensors(arena: &mut Arena, root: &Path, values: &Value,
    files: &mut Vec<String>, label: &str,prefetched:Option<crate::weight_loader::ReaderPool>) -> Result<HashMap<String, Tensor>> {
    let entries: Vec<(String, TensorSpec)> = values.as_object().ok_or("model tensor map missing")?
        .iter().map(|(name, value)| Ok((name.clone(), TensorSpec::parse(value)?)))
        .collect::<Result<_>>()?;
    let mut loaded = HashMap::new();
    let mut bytes = 0usize;
    let started = Instant::now();
    let readers=model_read_parallelism()?;
    let cache_jobs=if crate::host_cache::enabled()? {
        Some(entries.iter().map(|(_,spec)|crate::weight_loader::Job::new(root,&spec.file,spec.bytes,spec.sha256.clone()))
            .collect::<std::result::Result<Vec<_>,String>>()?)
    }else{None};
    if cache_jobs.as_ref().map(|jobs|crate::host_cache::can_upload(jobs)).transpose()?.unwrap_or(false) {
        let jobs=entries.iter().map(|(_,spec)|crate::weight_loader::Job::new(root,&spec.file,spec.bytes,spec.sha256.clone()))
            .collect::<std::result::Result<Vec<_>,String>>()?;
        let mut buffers=Vec::with_capacity(entries.len());
        for (name,spec) in &entries {
            let buffer=arena.alloc(spec.bytes)?;
            buffers.push(buffer.ptr);files.push(spec.file.clone());bytes+=spec.bytes;
            loaded.insert(name.clone(),Tensor {spec:spec.clone(),buffer});
        }
        let report=unsafe {crate::host_cache::upload(&jobs,&buffers,arena.stream,false)}?;
        eprintln!("native {label} pinned cache load: {report}");
        if std::env::var("H3_MODEL_LOAD_VERIFY_GPU").as_deref()==Ok("1") {
            let config=crate::weight_loader::Config::from_env()?.ok_or("GPU verification requires pipeline")?;
            let verify=crate::weight_loader::device::Upload::new(arena.stream,config)?;
            for (pointer,(_,spec)) in buffers.iter().zip(&entries) {
                unsafe {verify.verify(*pointer,spec.bytes,spec.sha256.as_deref().ok_or("GPU verification requires SHA256")?)}?;
            }
        }
        return Ok(loaded);
    }
    if let Some(config)=crate::weight_loader::Config::from_env()? {
        let jobs=entries.iter().map(|(_,spec)|crate::weight_loader::Job::new(root,&spec.file,spec.bytes,spec.sha256.clone()))
            .collect::<std::result::Result<Vec<_>,String>>()?;
        let mut pool=if let Some(pool)=prefetched{pool}else{crate::weight_loader::ReaderPool::start(jobs,config)?};
        let mut upload=crate::weight_loader::device::Upload::new(arena.stream,config)?;
        let verify=std::env::var("H3_MODEL_LOAD_VERIFY_GPU").as_deref()==Ok("1");
        let buffers=arena.alloc_model_tensors(&entries.iter().map(|(_,spec)|spec.bytes).collect::<Vec<_>>())?;
        for ((name,spec),&buffer) in entries.iter().zip(&buffers) {
            files.push(spec.file.clone());
            bytes+=spec.bytes;
            loaded.insert(name.clone(),Tensor {spec:spec.clone(),buffer});
        }
        unsafe{pool.consume_upload(&mut upload,|i,offset|buffers[i].ptr.cast::<u8>().add(offset).cast())}?;
        upload.finish()?;
        if verify {for (buffer,(_,spec)) in buffers.iter().zip(&entries) {
            unsafe{upload.verify(buffer.ptr,buffer.bytes,spec.sha256.as_deref().ok_or("GPU verification requires SHA256")?)}?;
        }}
        eprintln!("native {label} streaming load: {} tensors, {:.2} GiB, {:.1}s, config={config:?}, GPU_SHA256_verified={verify}",
            loaded.len(),bytes as f64/(1u64<<30) as f64,started.elapsed().as_secs_f64());
        return Ok(loaded);
    }
    // Only host file reads are parallel. Bound read-ahead by the trusted setting;
    // preserve manifest order and the original single-stream CUDA uploads.
    for (batch, chunk) in entries.chunks(readers).enumerate() {
        let data = std::thread::scope(|scope| {
            let handles: Vec<_> = chunk.iter().map(|(_, spec)|
                scope.spawn(move || spec.read(root).map_err(|error| error.to_string()))).collect();
            handles.into_iter().map(|handle| handle.join()
                .map_err(|_| "model tensor reader panicked".to_owned())?)
                .collect::<std::result::Result<Vec<_>, String>>()
        }).map_err(|error| format!("{label} read: {error}"))?;
        for ((name, spec), data) in chunk.iter().zip(data) {
            let buffer = arena.upload(&data)?;
            bytes += data.len();
            files.push(spec.file.clone());
            loaded.insert(name.clone(), Tensor { spec: spec.clone(), buffer });
        }
        if (batch + 1) % 8 == 0 || loaded.len() == entries.len() {
            eprintln!("native {label} load: {}/{} tensors, {:.2} GiB, {:.1}s",
                loaded.len(), entries.len(), bytes as f64 / (1u64 << 30) as f64,
                started.elapsed().as_secs_f64());
        }
    }
    Ok(loaded)
}

fn model_read_parallelism()->Result<usize> {
    match std::env::var("H3_MODEL_READ_PARALLELISM").as_deref() {
        Err(_)|Ok("8")=>Ok(8),Ok("32")=>Ok(32),_=>Err("model read parallelism must be 8 or 32".into()),
    }
}

impl Runtime {
    fn log_step(&self,step:usize,started:Instant)->Result<()> {
        let _profile=HostSpan::new("host.step_log",0);
        if self.cp.as_ref().is_none_or(|cp|cp.group.rank==0) {
            let sparse_layers=self.execution.sparse_layers(step,self.seq)?;
            // Value's Display implementation streams many small JSON fragments.
            // Buffer the complete line before touching stderr (which may be a
            // remote log file), so logging cannot stall sampling per fragment.
            let line=json!({"event":"native.dit.step_completed","rank":0,
                "schedule_index":step,"attention_evaluation":self.execution.attention_index(step)?,
                "evaluations":self.execution.evaluations(),"sequence":self.seq,
                "sparse":sparse_layers>0,"sparse_layers":sparse_layers,
                "dense_layers":crate::execution::MAIN_LAYERS-sparse_layers,"audio_frozen":self.condition_audio==self.na,
                "wall_seconds":started.elapsed().as_secs_f64()}).to_string();
            eprintln!("{line}");
        }
        Ok(())
    }
    pub fn new(root: &Path, router_root: &Path, dense_path: &Path, output: &Path) -> Result<Self> {
        Self::new_with_request(root, router_root, dense_path, output, None)
    }
    pub fn new_with_request(root: &Path, router_root: &Path, dense_path: &Path, output: &Path,
        request_root: Option<&Path>) -> Result<Self> {
        Self::build(root,router_root,dense_path,output,request_root,true)
    }
    pub fn new_for_generation(root:&Path,router_root:&Path,dense_path:&Path,output:&Path,
        request_root:&Path)->Result<Self> {
        Self::build(root,router_root,dense_path,output,Some(request_root),false)
    }
    fn build(root:&Path,router_root:&Path,dense_path:&Path,output:&Path,
        request_root:Option<&Path>,validate_reference:bool)->Result<Self> {
        Self::build_shared(root,router_root,dense_path,output,request_root,validate_reference,None)
    }
    fn build_shared(root:&Path,router_root:&Path,dense_path:&Path,output:&Path,
        request_root:Option<&Path>,validate_reference:bool,
        shared:Option<std::rc::Rc<ResidentModel>>)->Result<Self> {
        cp_select_device_from_environment()?;
        let load_started = Instant::now();
        let manifest_bytes = crate::startup_trace::read(root.join("manifest.json"))?;
        let manifest: Value = serde_json::from_slice(&manifest_bytes)?;
        let recipe=crate::recipe::Recipe::from_model(&manifest)?;
        let numerics=crate::numerics::Numerics::from_model(&manifest)?;
        crate::integrity::admit_base(&manifest)?;
        let valid_schema=manifest["schema"]=="h3-native-export-v1" ||
            (!validate_reference && manifest["schema"]=="h3-native-model-v1");
        if !valid_schema || manifest["status"] != "complete" {
            return Err("requires a completed native export with recipe-bound steps".into());
        }
        fs::create_dir_all(output)?;
        let backend: Value = if let Some(value)=manifest.get("linear_backend") {value.clone()}else{
            serde_json::from_slice(&crate::startup_trace::read(root.parent().ok_or("no export parent")?.join("linears/manifest.json"))?)?};
        if backend["backend"]["allow_tf32"] != false || backend["backend"]["allow_bf16_reduced_precision_reduction"] != true {
            return Err("GEMM backend flags differ from validated native library contract".into());
        }
        let request_pool=shared.as_ref().map(|m|m.pool.clone()).unwrap_or_else(||std::rc::Rc::new(RequestPool::new()));
        let mut arena = Arena::new_with_pool(Some(request_pool.clone()))?;
        let dense=if let Some(model)=&shared{model.dense.clone()}else{std::rc::Rc::new(Dense::load(dense_path)?)};
        let mut files = vec![];
        let request_manifest: Option<Value> = if let Some(path)=request_root {
            Some(serde_json::from_slice(&crate::startup_trace::read(path.join("manifest.json"))?)?)
        } else {None};
        let input_root = request_root.unwrap_or(root);
        let execution=if let Some(request)=&request_manifest {crate::execution::Execution::from_request(request)?}
            else{crate::execution::Execution::full(recipe.nfe)};
        let initial = request_manifest.as_ref().map_or(&manifest["steps"][0]["inputs"], |request|&request["inputs"]);
        let mut native_request = json!({"enabled":false});
        if let Some(request) = &request_manifest {
            if crate::recipe::Recipe::from_request(request)?!=recipe{return Err("native request and model recipes differ".into());}
            if request["schema"] != "h3-native-request-v1" || request["status"] != "complete" ||
                request["producer"] != "h3-native-pipeline" || request["gold_activations_used"] != false {
                return Err("requires a complete native request preparation manifest".into());
            }
            let mut checks=serde_json::Map::new();
            for name in ["hidden_states", "audio_hidden_states", "encoder_hidden_states", "position_ids",
                         "token_tags", "video_indices", "audio_indices", "text_indices"] {
                let actual=TensorSpec::parse(&initial[name])?;
                if !validate_reference {continue;}
                let expected=TensorSpec::parse(&manifest["steps"][0]["inputs"][name])?;
                if actual.shape!=expected.shape || actual.dtype!=expected.dtype {
                    return Err(format!("native request changes fixed case shape/dtype: {name}").into());
                }
                let actual_bytes=actual.read(input_root)?;
                let expected_bytes=expected.read(root)?;
                // Gold is comparison-only here. CUDA input allocations below are
                // populated directly from request_root, with no gold fallback.
                let exact=actual_bytes==expected_bytes;
                checks.insert(name.to_string(),json!({"passed":exact,"bytes":actual_bytes.len(),
                    "actual_file":input_root.join(&actual.file),"comparison_file":root.join(&expected.file)}));
                if !exact { return Err(format!("native request differs from fixed reference: {name}").into()); }
            }
            native_request=json!({"enabled":true,"input_root":input_root,"preparation":request,
                "input_checks":checks,"reference_inputs_used_for_comparison_only":validate_reference,
                "gold_activations_read":validate_reference,"rotary_checks":[]});
        }
        let get = |name: &str| TensorSpec::parse(&initial[name]);
        let (v_spec, a_spec, t_spec) = (get("hidden_states")?, get("audio_hidden_states")?, get("encoder_hidden_states")?);
        if v_spec.shape.len() != 3 || a_spec.shape.len() != 3 || t_spec.shape.len() != 3 ||
            v_spec.shape[0] != 1 || a_spec.shape[0] != 1 || t_spec.shape[0] != 1 ||
            v_spec.shape[2] != VIDEO || a_spec.shape[2] != AUDIO || t_spec.shape[2] != 5120 ||
            v_spec.dtype != "torch.float32" || a_spec.dtype != "torch.float32" || t_spec.dtype != "torch.bfloat16" {
            return Err("unsupported initial modality layout/dtype".into());
        }
        let (nv, na, nt) = (v_spec.shape[1], a_spec.shape[1], t_spec.shape[1]);
        let seq = nv + na + nt;
        if seq > i32::MAX as usize || seq != get("token_tags")?.shape[0] { return Err("packed sequence mismatch".into()); }
        let latent_video = load(&mut arena, input_root, &initial["hidden_states"], &mut files)?.buffer;
        let latent_audio = load(&mut arena, input_root, &initial["audio_hidden_states"], &mut files)?.buffer;
        let conditioning = load(&mut arena, input_root, &initial["encoder_hidden_states"], &mut files)?.buffer;
        let tags_spec = get("token_tags")?;
        if tags_spec.dtype != "torch.int64" { return Err("token_tags must be int64".into()); }
        let tags = i64_values(&tags_spec.read(input_root)?)?;
        files.push(tags_spec.file.clone());
        let mut modality_indices = vec![];
        for (name, count) in [("video_indices", nv), ("audio_indices", na), ("text_indices", nt)] {
            let spec = get(name)?;
            if spec.dtype != "torch.int64" { return Err("indices must be int64".into()); }
            let values = i64_values(&spec.read(input_root)?)?;
            if values.len() != count { return Err("index count mismatch".into()); }
            modality_indices.push(values);
        }
        validate_packed_indices(&tags, &modality_indices[0], &modality_indices[1], &modality_indices[2])?;
        let dense_audio_range=None;

        // Reject cheap structural mismatches before reading the 38 GiB model.
        let reused_model=shared.is_some();
        let resident_model=if let Some(model)=shared {
            if model.root!=root{return Err("resident model root changed".into());}
            model
        }else{
        ResidentModel::load(root,&manifest,&manifest_bytes,dense.clone(),request_pool,None,0,&mut files,None,None,None,None,None)?

        };
        let weights=resident_model.weights.clone();let buffers=resident_model.buffers.clone();
        let model_load_seconds=if reused_model{0.0}else{resident_model.load_seconds};
        let video_indices = load(&mut arena, input_root, &initial["video_indices"], &mut files)?.buffer;
        let audio_indices = load(&mut arena, input_root, &initial["audio_indices"], &mut files)?.buffer;
        let text_indices = load(&mut arena, input_root, &initial["text_indices"], &mut files)?.buffer;
        let text_indices_host = i64_values(&get("text_indices")?.read(input_root)?)?;
        let condition_video = request_manifest.as_ref().map_or(&manifest["steps"][0]["sampler"]["num_condition_video_rows"],
            |request|&request["condition_video_rows"]).as_u64().ok_or("conditioning video count")? as usize;
        let condition_audio = request_manifest.as_ref().map_or(&manifest["steps"][0]["sampler"]["num_condition_audio_rows"],
            |request|&request["condition_audio_rows"]).as_u64().ok_or("conditioning audio count")? as usize;
        if condition_video > nv || condition_audio > na { return Err("conditioning row count exceeds modality".into()); }
        if validate_reference && (condition_video!=manifest["steps"][0]["sampler"]["num_condition_video_rows"].as_u64().unwrap() as usize ||
            condition_audio!=manifest["steps"][0]["sampler"]["num_condition_audio_rows"].as_u64().unwrap() as usize) {
            return Err("request condition counts differ from reference".into());
        }
        let read_schedule = |name: &str| -> Result<Vec<f32>> {
            let spec = TensorSpec::parse(&manifest["schedule"][name])?;
            if spec.dtype != "torch.float32" { return Err("schedule must be float32".into()); }
            f32_values(&spec.read(root)?)
        };
        let vs = read_schedule("video_sigmas")?;
        let ats = read_schedule("audio_sigmas")?;
        let vt = read_schedule("video_timesteps")?;
        let at = read_schedule("audio_timesteps")?;
        if vs.len() != recipe.nfe+1 || ats.len() != recipe.nfe+1 || vt.len() != recipe.nfe || at.len() != recipe.nfe { return Err("recipe schedule length mismatch".into()); }
        let request_steps=if let Some(request)=&request_manifest {
            let aug=request["keyframe_noise_aug"].as_f64().ok_or("keyframe_noise_aug missing")? as f32;
            Some((0..recipe.nfe).map(|i|row_timestep_indices(seq,&modality_indices[0],&modality_indices[1],
                condition_video,condition_audio,vt[i],at[i],aug)).collect::<Result<Vec<_>>>()?)
        }else{None};
        let plan=match &request_steps {Some(rows)=>crate::timestep_plan::select(&manifest["adaln"],rows)?,None=>None};
        let plan_finals=plan.as_ref().and_then(|p|resident_model.plan_final_mods.get(&p.plan));
        let mut steps = vec![];
        for i in 0..recipe.nfe {
            let entry = &manifest["steps"][i];
            let host_indices=if let Some(rows)=&request_steps {
                let (unique,indices)=&rows[i];
                let spec=TensorSpec::parse(&entry["inputs"]["timestep"])?;
                if spec.dtype!="torch.float32" {return Err("fixed timestep values must be FP32".into());}
                let expected=f32_values(&spec.read(root)?)?;
                let indices=if plan_finals.is_some(){indices.clone()}else{
                    crate::resident_timestep_indices(unique,indices,&expected)
                        .map_err(|error|format!("step {i}: {error}"))?};
                if validate_reference {
                    let spec=TensorSpec::parse(&entry["inputs"]["timestep_indices"])?;
                    if indices!=i64_values(&spec.read(root)?)? {return Err(format!("step {i}: fresh timestep rows differ").into());}
                }
                indices
            }else{
                let spec=TensorSpec::parse(&entry["inputs"]["timestep_indices"])?;
                if spec.dtype!="torch.int64" {return Err("timestep indices must be int64".into());}
                i64_values(&spec.read(root)?)?
            };
            if host_indices.len()!=seq {return Err("timestep index shape mismatch".into());}
            let modulation_indices=plan.as_ref().map_or(&host_indices,|p|&p.indices[i]);
            let combined: Vec<i64> = modulation_indices.iter().zip(&tags).map(|(index, tag)| index * 3 + tag).collect();
            let indices = arena.upload(&bytes_i64(&host_indices))?;
            let adaln_indices = arena.upload(&bytes_i64(&combined))?;
            let final_mod = &plan_finals.unwrap_or(&resident_model.final_mods)[i];
            if final_mod.spec.dtype != "torch.bfloat16" || final_mod.spec.shape.len() != 2 || final_mod.spec.shape[1] != 2 * HIDDEN {
                return Err("final modulation table layout mismatch".into());
            }
            // T2V's initial M=1 time embedding must be computed at M=1. A
            // slice of FL2VA's M=2 GEMM has different rounding. Later M=2
            // rows match the resident M=3 tables (checked against original).
            let zero_only=plan_finals.is_none() && recipe.workflow=="fl2va" && i==0 && request_steps.as_ref().is_some_and(|rows|rows[0].0==[0.0]);
            if zero_only && plan.is_none(){return Err("text-only AdaLN plan is not installed".into());}
            let nrows = if zero_only{1}else{final_mod.spec.shape[0]};
            let final_buffer=if zero_only{resident_model.zero_final_mod.ok_or("text-only final modulation missing")?}else{final_mod.buffer};
            if host_indices.iter().any(|&x| x < 0 || x as usize >= nrows) { return Err("timestep index out of range".into()); }
            let cursor = match &plan{Some(p)=>p.cursors[i],None=>entry["adaln_cursor"].as_u64().ok_or("AdaLN cursor missing")? as usize};
            for layer in 0..50 {
                let name = format!("transformer_blocks.{layer}.adaln_proj.table");
                let table = buffers.get(&name).ok_or("missing main AdaLN table")?;
                if table.spec.dtype != "torch.bfloat16" || table.spec.shape.len() != 3 || table.spec.shape[2] != 6 * HIDDEN ||
                    cursor >= table.spec.shape[0] || combined.iter().any(|&x| x < 0 || x as usize >= table.spec.shape[1]) {
                    return Err(format!("AdaLN table/cursor/index mismatch: {name}").into());
                }
            }
            if entry["attention"]["dense_prefix_steps"] != 10 || entry["attention"]["sparsity"] != 0.75 {
                return Err("attention schedule differs from frozen contract".into());
            }
            steps.push(Step { indices, adaln_indices, final_mod: final_buffer, cursor,
                video_t: vt[i], audio_t: at[i], video_ratio: vs[i + 1] / vs[i], audio_ratio: ats[i + 1] / ats[i],
                next_video_sigma:vs[i+1] });
        }
        let rotary = &manifest["native_inputs"]["rotary_emb"];
        let mut rope_buffers = vec![];
        let fresh_rotary = if request_root.is_some() {
            let position=load(&mut arena,input_root,&initial["position_ids"],&mut files)?;
            let inv=buffers.get("rope.inv_freq").ok_or("model rotary frequency buffer missing")?;
            if position.spec.dtype!="torch.float64" || position.spec.shape!=[seq,3] ||
                inv.spec.dtype!="torch.float32" || inv.spec.shape!=[16] {return Err("rotary request input mismatch".into());}
            let cos=arena.alloc(seq*96*4)?;let sin=arena.alloc(seq*96*4)?;
            cuda(unsafe{h3_rotary_tables(position.buffer.ptr.cast(),inv.buffer.ptr.cast(),cos.ptr.cast(),sin.ptr.cast(),seq as i64,arena.stream)})?;
            Some([cos,sin])
        } else {None};
        for j in 0..2 {
            let value=if let Some(tables)=fresh_rotary {
                if validate_reference {
                    let spec=TensorSpec::parse(&rotary[j])?;
                    if spec.dtype!="torch.float32" || spec.shape!=[seq,96] {return Err("reference rotary shape mismatch".into());}
                    let actual=arena.download(tables[j])?;
                    let check=compare(&actual,&spec.read(root)?,4)?;
                    if check["passed"]!=true {return Err(format!("fresh native rotary table differs: {check}").into());}
                    native_request["rotary_checks"].as_array_mut().unwrap().push(check);
                }
                tables[j]
            } else {
                let spec=TensorSpec::parse(&rotary[j])?;
                if spec.dtype!="torch.float32" || spec.shape!=[seq,96] {return Err("rotary table shape/dtype mismatch".into());}
                load(&mut arena,root,&rotary[j],&mut files)?.buffer
            };
            if numerics.fused() {
                rope_buffers.push(value);
            } else {
                let converted = arena.alloc(seq * 96 * 2)?;
                cuda(unsafe { h3_cast(value.ptr, converted.ptr, (seq * 96) as i64, 0, arena.stream) })?;
                rope_buffers.push(converted);
            }
        }
        let (world,rank)=resident_model.cp_group.as_ref().map_or((1,0),|g|(g.world,g.rank));
        let plan=crate::scratch_plan::ScratchPlan::new(seq,nt,world,rank)?;
        let projection_aliases=plan.projection_aliases(world,validate_reference)?;
        let rows=plan.rows;
        let head_plan=crate::attention_heads::HeadPlan::new(seq)?;
        let dense_bytes=if world>1 {dense.size(nt,1)?}else{
            dense.bytes_for_heads(seq,head_plan.heads,0)?.max(dense.size(nt,1)?).max(if qkv_inplace_enabled(){dense.size(seq,2)?}else{1})};
        let mut scratch = HashMap::new();
        for (name, bytes) in [
            ("hidden", seq * HIDDEN * 2), ("norm", seq * HIDDEN * 2), ("branch", rows * HIDDEN * 2),
            ("q", rows * INNER * 2), ("k", rows * INNER * 2), ("v", rows * INNER * 2),
            ("qn", rows * INNER * 2), ("kn", rows * INNER * 2), ("attn", rows * INNER * 2),
            ("ff2", plan.ff2), ("ff", plan.ff), ("fp_hidden", plan.fp_hidden),
            ("text", nt * HIDDEN * 2), ("all_video", seq * VIDEO * 4), ("all_audio", seq * AUDIO * 4),
            ("velocity_video", nv * VIDEO * 4), ("velocity_audio", na * AUDIO * 4),
            ("blas_ws", 32 << 20), ("lt_ws", 1 << 20),
            ("dense_ws", dense_bytes.max(1)),
        ] {
            if projection_aliases.is_some() && matches!(name,"qn"|"kn"|"fp_hidden") {continue;}
            let buffer=if world==1 && matches!(name,"q"|"k"|"v") {
                arena.sparse_input(rows,INNER)
            }else{arena.alloc(bytes)};
            scratch.insert(name, buffer.map_err(|error|format!("DiT scratch {name} (S={seq}): {error}"))?);
        }
        if let Some(views)=projection_aliases {
            // Keep all kernel types, GEMM dimensions and operation order. On
            // one device the input is not staged in fp_hidden. FFN
            // projection storage is dead during Q/K normalization and FP32
            // modality projections; qn/kn have separate slices while live.
            // Validation layouts retain Q/K for later diagnostic comparisons.
            let storage=scratch["ff2"];
            scratch.insert("qn",storage.prefix(views.qk_bytes)?);
            scratch.insert("kn",storage.slice(views.kn_offset,views.qk_bytes)?);
            scratch.insert("fp_hidden",storage.prefix(views.fp_bytes)?);
        }
        if plan.small_input>0 {
            scratch.insert("cp_small_input",arena.alloc(plan.small_input)?);
            scratch.insert("cp_small_output",arena.alloc(plan.small_output)?);
        }
        let blocks = seq.div_ceil(64);
        let kept = (((blocks as f64 * 0.25).ceil() as usize).div_ceil(8) * 8).min(blocks);
        let mut route_buffers = HashMap::new();
        // CP owns a separate local-head router/workspace. Allocating this full-
        // head copy as well wastes quadratic scratch and it is never executed.
        let router = if world==1 {
            let heads=head_plan.heads;
            if head_plan.groups>1 {
                for name in ["head_q","head_k","head_v","head_out"] {
                    let buffer=if name=="head_out"{arena.alloc(seq*heads*128*2)?}else{arena.sparse_input(seq,heads*128)?};
                    scratch.insert(name,buffer);
                }
            }
            let router=resident_model.routers.get(router_root,seq,heads)?;
            for buffer in &router.buffers {
                if buffer.name=="q" || buffer.name=="k" {
                    if buffer.bytes!=seq*heads*128*2 {return Err("router fixture shape does not match DiT sequence".into());}
                    let name=match (head_plan.groups>1,buffer.name.as_str()) {
                        (true,"q")=>"head_q",(true,_)=>"head_k",(false,"q")=>"q",_=>"k"};
                    route_buffers.insert(buffer.name.clone(),scratch[name].ptr as u64);
                } else {route_buffers.insert(buffer.name.clone(),arena.alloc(buffer.bytes)?.ptr as u64);}
            }
            let index_bytes=router.buffers.iter().find(|x|x.name=="indices").ok_or("router index buffer missing")?.bytes;
            if index_bytes!=heads*blocks*4*kept {return Err("router indices layout mismatch".into());}
            // The large-layout head plan bounds the C++ pack/LSE workspace.
            {
                let mut bsa_bytes=0;
                cuda(unsafe{h3_bsa_masked_workspace_bytes(seq as c_int,heads as c_int,&mut bsa_bytes)})?;
                scratch.insert("bsa_ws",arena.alloc(bsa_bytes)?);
            }
            Some(router)
        } else {None};
        let sizes: Vec<u8> = (0..blocks).flat_map(|i| ((seq - i * 64).min(64) as i32).to_le_bytes()).collect();
        let block_sizes = arena.upload(&sizes)?;
        let reference_geometry=if recipe.workflow=="ref2va" {
            Some(serde_json::from_value::<Vec<h3_native_pipeline::ref2va::ReferenceGeometry>>(
                request_manifest.as_ref().ok_or("Ref2VA sparse queries need reference geometry")?["references"].clone())?)
        }else{None};
        let mut image_queries=crate::subblock::image_condition_positions(&modality_indices[0],condition_video,reference_geometry.as_deref())?;
        let qwen_video_tokens=request_manifest.as_ref().and_then(|r|r.get("text_video_token_indices"))
            .map(|v|serde_json::from_value::<Vec<usize>>(v.clone())).transpose()?.unwrap_or_default();
        image_queries.extend(crate::subblock::text_image_positions(&tags,&modality_indices[2],&qwen_video_tokens)?);
        let query_mask=crate::subblock::sparse_query_blocks(&tags,&image_queries)?;
        let query_bytes=query_mask.iter().flat_map(|value|value.to_le_bytes()).collect::<Vec<_>>();
        let query_block_mask=arena.upload(&query_bytes)?;
        if let Some((lo,hi))=dense_audio_range {
            let local_heads=resident_model.cp_group.as_ref().map_or(head_plan.heads,|g|HEADS/g.world);
            let bytes=dense.rect_bytes(hi-lo,seq,local_heads)?;
            scratch.insert("dense_audio_ws",arena.alloc(bytes)?);
        }

        arena.sync()?;
        let mut resident_layout_inputs=HashMap::new();
        if request_root.is_some(){
            for name in ["position_ids","token_tags","video_indices","audio_indices","text_indices"] {
                let spec=TensorSpec::parse(&initial[name])?;
                resident_layout_inputs.insert(name.to_owned(),spec.read(input_root)?);
            }
        }
        let cp=CpState::new(&mut arena,&resident_model,router_root,seq)?;
        let mut linear = ptr::null_mut();
        blas(unsafe { h3_linear_create(arena.stream, scratch["blas_ws"].ptr, 32 << 20,
            scratch["lt_ws"].ptr, 1 << 20, 0, 1, &mut linear) })?;
        let result = Self { cp, resident_model, arena, linear, dense, router, route_buffers, weights, buffers, scratch,
            steps, execution, long_request:None,long_sparse_calls:None,router_root:router_root.to_owned(),dense_path:dense_path.to_owned(), manifest, root: root.to_owned(), output: output.to_owned(), seq, nv, na, nt,
            condition_video, condition_audio, dense_audio_range, numerics, latent_video, latent_audio, conditioning,
            video_indices, audio_indices, text_indices, text_indices_host,
            cos: rope_buffers[0], sin: rope_buffers[1], block_sizes, query_block_mask, kept, native_input_files: files, native_request,
            validate_reference,resident_layout_inputs,load_seconds: load_started.elapsed().as_secs_f64(), model_load_seconds };
        result.audit("loaded")?;
        eprintln!("native loaded: S={seq} video={nv} audio={na} text={nt}, {} resident weight tensors", result.weights.len());
        Ok(result)
    }

    fn gather_weights(&self,group:&str)->Result<()> {
        if let Some(sharded)=&self.resident_model.sharded{sharded.materialize(group,self.arena.stream)?;}
        Ok(())
    }
    fn w(&self, name: &str) -> Result<Tensor> {
        let tensor=self.weights.get(name).ok_or_else(||format!("missing effective weight {name}"))?;
        if let Some(sharded)=&self.resident_model.sharded{return sharded.resolve(&format!("weights/{name}"),tensor);}
        Ok(tensor.clone())
    }
    fn s(&self, name: &'static str) -> Buffer { self.scratch[name] }
    fn eps(&self, name: &str) -> Result<f32> {
        self.manifest["modules"][name]["eps"].as_f64().map(|x| x as f32).ok_or_else(|| format!("missing epsilon {name}").into())
    }
    fn linear_qkv(&self,name:&str,input:Buffer,q:Buffer,k:Buffer,v:Buffer,rows:usize)->Result<()> {
        for (suffix,out) in [("to_q",q),("to_k",k),("to_v",v)] {
            self.linear(&format!("{name}.{suffix}"),input,out,rows)?;
        }
        Ok(())
    }
    fn linear(&self, name: &str, input: Buffer, out: Buffer, rows: usize) -> Result<()> {
        let fine_main=name.starts_with("transformer_blocks.");
        let fine_label=match name {
            "proj_in"=>"projection.input_video", "audio_proj_in"=>"projection.input_audio",
            "context_embedder"=>"projection.context",
            "proj_out"=>"projection.output_video", "audio_proj_out"=>"projection.output_audio",
            _ if name.ends_with(".net.0.proj")=>if fine_main{"ffn.main.up_gemm"}else{"ffn.refiner.up_gemm"},
            _ if name.ends_with(".net.2")=>if fine_main{"ffn.main.down_gemm"}else{"ffn.refiner.down_gemm"},
            _ if name.ends_with(".to_q")=>if fine_main{"projection.main.q"}else{"projection.refiner.q"},
            _ if name.ends_with(".to_k")=>if fine_main{"projection.main.k"}else{"projection.refiner.k"},
            _ if name.ends_with(".to_v")=>if fine_main{"projection.main.v"}else{"projection.refiner.v"},
            _ if name.ends_with(".to_qkv")=>if fine_main{"projection.main.qkv"}else{"projection.refiner.qkv"},
            _ if name.ends_with(".to_out.0")=>if fine_main{"projection.main.attention_out"}else{"projection.refiner.attention_out"},
            _=>"projection.unclassified",
        };
        let _fine_host=HostSpan::new(fine_label,0);
        let _fine_gpu=GpuSpan::new(fine_label,self.arena.stream)?;
        let weight = self.w(&format!("{name}.weight"))?;
        if weight.spec.shape.len() != 2 { return Err("Linear weight is not a matrix".into()); }
        let (n, k) = (weight.spec.shape[0], weight.spec.shape[1]);
        let dtype = match weight.spec.dtype.as_str() { "torch.bfloat16" => 0, "torch.float32" => 1, _ => return Err("Linear weight dtype".into()) };
        let width = if dtype == 0 { 2 } else { 4 };
        if input.bytes < rows * k * width || out.bytes < rows * n * width { return Err(format!("Linear buffer too small: {name}").into()); }
        let bias_name=format!("{name}.bias");
        let bias=self.weights.contains_key(&bias_name).then(||self.w(&bias_name)).transpose()?;
        if let Some(bias) = &bias { if bias.spec.dtype != weight.spec.dtype || bias.spec.shape != [n] { return Err("Linear bias shape/dtype mismatch".into()); } }
        let full_m=self.cp.as_ref().filter(|_|self.seq<4096 && fine_main);
        if name=="proj_in"{native_core_infra::execution_start::enqueue_if_armed(self.arena.stream)?;}
        if let Some(cp)=full_m {
            // S595 diagnosis: Q/K/V, RoPE and merged attention are exact, but
            // the local M74/75 output projection differs from original M595.
            // Preserve all base GEMM dimensions for these small layouts: the
            // same vendor selection issue can occur at later projections.
            // The row-independent projection needs no activation collective;
            // non-owned rows are zero and their outputs are discarded.
            if rows!=cp.rows || dtype!=0{return Err("small CP output projection geometry mismatch".into());}
            let padded=self.s("cp_small_input").prefix(self.seq*k*width)?;
            let projected=self.s("cp_small_output").prefix(self.seq*n*width)?;
            if [input.ptr,out.ptr].contains(&padded.ptr) || [input.ptr,out.ptr,padded.ptr].contains(&projected.ptr) {
                return Err("small CP projection scratch aliases live tensors".into());
            }
            cuda(unsafe{cudaMemsetAsync(padded.ptr,0,padded.bytes,self.arena.stream)})?;
            self.arena.copy(padded.slice(cp.start*k*width,rows*k*width)?,input.prefix(rows*k*width)?)?;
            blas(unsafe{h3_linear_forward(self.linear,padded.ptr,weight.buffer.ptr,
                bias.as_ref().map_or(ptr::null(),|x|x.buffer.ptr),projected.ptr,
                self.seq as c_int,n as c_int,k as c_int,dtype)})
                .map_err(|error|format!("{name}: {error}"))?;
            self.arena.copy(out.prefix(rows*n*width)?,projected.slice(cp.start*n*width,rows*n*width)?)?;
        } else {
            blas(unsafe { h3_linear_forward(self.linear, input.ptr, weight.buffer.ptr,
                bias.as_ref().map_or(ptr::null(), |x| x.buffer.ptr), out.ptr, rows as c_int, n as c_int, k as c_int, dtype) })
                .map_err(|error| format!("{name}: {error}"))?;
        }
        Ok(())
    }
    fn norm(&self, name: &str, input: Buffer, output: Buffer, rows: usize, width: usize) -> Result<()> {
        let weight = self.w(&format!("{name}.weight"))?;
        if weight.spec.dtype != "torch.bfloat16" || weight.spec.shape != [width] { return Err(format!("norm weight mismatch: {name}").into()); }
        if input.bytes < rows * width * 2 || output.bytes < rows * width * 2 { return Err("norm buffer too small".into()); }
        cuda(unsafe { h3_rms_bf16(input.ptr, weight.buffer.ptr, output.ptr, rows as i64,
            width as c_int, self.eps(name)?, self.arena.stream) })
    }
    fn cast(&self, input: Buffer, out: Buffer, n: usize, to_float: bool) -> Result<()> {
        let _fine_host=HostSpan::new("layout.cast",0);
        let _fine_gpu=GpuSpan::new("layout.cast",self.arena.stream)?;
        if input.bytes < n * if to_float { 2 } else { 4 } || out.bytes < n * if to_float { 4 } else { 2 } { return Err("cast buffer too small".into()); }
        cuda(unsafe { h3_cast(input.ptr, out.ptr, n as i64, to_float as c_int, self.arena.stream) })
    }
    fn scatter(&self, input: Buffer, indices: Buffer, out: Buffer, rows: usize, cols: usize, bytes: usize, scatter: bool) -> Result<()> {
        let _fine_host=HostSpan::new("layout.rows_copy",0);
        let _fine_gpu=GpuSpan::new("layout.rows_copy",self.arena.stream)?;
        cuda(unsafe { h3_rows_copy(input.ptr, indices.ptr.cast(), out.ptr, rows as i64,
            cols as c_int, bytes as c_int, scatter as c_int, self.arena.stream) })
    }

    fn attention(&self, name: &str, input: Buffer, rows: usize, main: bool, sparse: bool) -> Result<Buffer> {
        self.attention_trace(name,input,rows,main,sparse,None,&mut Vec::new())
    }
    fn attention_trace(&self, name: &str, input: Buffer, rows: usize, main: bool, sparse: bool,
        directory:Option<&Path>,trace:&mut Vec<Value>) -> Result<Buffer> {
        if main && self.cp.is_some(){return Err("CP main attention requires local-row path".into());}
        let (mut q, mut k, mut v, qn, kn) = (self.s("q"), self.s("k"), self.s("v"), self.s("qn"), self.s("kn"));
        let packed_rope=main && self.numerics.sglang()
            && self.manifest["modules"][name]["fused_projections"]==true && directory.is_none()
            && !crate::debug_audit::enabled();
        let inplace_qkv=packed_rope && qkv_inplace_enabled() && crate::pointwise_opt::enabled() && !sparse && self.cp.is_none()
            && self.router.as_ref().is_none_or(|r|r.plan.heads==HEADS);
        if inplace_qkv {
            let packed=self.s("ff2");let width=INNER*2;
            q=packed;k=packed.slice(width,packed.bytes-width)?;
            v=packed.slice(2*width,packed.bytes-2*width)?;
        }
        if self.manifest["modules"][name]["fused_projections"] == true {
            self.linear(&format!("{name}.to_qkv"), input, self.s("ff2"), rows)?;
            if !packed_rope {
                cuda(unsafe { h3_qkv_split_bf16(self.s("ff2").ptr, q.ptr, k.ptr, v.ptr, rows as i64, INNER as c_int, self.arena.stream) })?;
            }
        } else {
            self.linear_qkv(name,input,q,k,v,rows)?;
        }
        for (label,buffer) in [("q",q),("k",k),("v",v)] {
            self.phase_boundary(directory,trace,label,buffer,rows,INNER)?;
        }
        let fine_qk_label=if main{"pointwise.main.qk_norm_rope"}else{"pointwise.refiner.qk_norm"};
        let fine_qk_host=HostSpan::new(fine_qk_label,0);
        let fine_qk_gpu=GpuSpan::new(fine_qk_label,self.arena.stream)?;
        let (actual_q, actual_k) = if main {
            let q_name=format!("{name}.norm_q");
            let k_name=format!("{name}.norm_k");
            let qw=self.w(&format!("{q_name}.weight"))?;
            let kw=self.w(&format!("{k_name}.weight"))?;
            for weight in [&qw,&kw] {
                if weight.spec.dtype!="torch.bfloat16" || weight.spec.shape!=[128] {
                    return Err("fused QK norm weight mismatch".into());
                }
            }
            let bytes=rows*HEADS*128*2;
            if [q,k,qn,kn].iter().any(|buffer|buffer.bytes<bytes) {
                return Err("fused QK norm buffer too small".into());
            }
            if self.numerics.sglang() {
                let keep_norm=crate::debug_audit::enabled() || directory.is_some();
                let (qn_ptr,kn_ptr)=if keep_norm{(qn.ptr,kn.ptr)}else{(ptr::null_mut(),ptr::null_mut())};
                if inplace_qkv {
                    cuda(unsafe {h3_sglang_qkv_rope_inplace(self.s("ff2").ptr,qw.buffer.ptr,kw.buffer.ptr,
                        self.cos.ptr,self.sin.ptr,rows as i64,HEADS as c_int,
                        self.eps(&q_name)?,self.eps(&k_name)?,self.arena.stream)})?;
                }else if packed_rope {
                    cuda(unsafe {h3_sglang_qkv_rope_v2(self.s("ff2").ptr,qw.buffer.ptr,kw.buffer.ptr,
                        self.cos.ptr,self.sin.ptr,qn_ptr,kn_ptr,q.ptr,k.ptr,v.ptr,rows as i64,HEADS as c_int,
                        self.eps(&q_name)?,self.eps(&k_name)?,self.arena.stream)})?;
                }else{
                    cuda(unsafe {h3_sglang_qk_rope_v2(q.ptr,k.ptr,qw.buffer.ptr,kw.buffer.ptr,
                        self.cos.ptr,self.sin.ptr,qn_ptr,kn_ptr,q.ptr,k.ptr,rows as i64,HEADS as c_int,
                        self.eps(&q_name)?,self.eps(&k_name)?,self.arena.stream)})?;
                }
            }else{
                cuda(unsafe { (if self.numerics.fused(){h3_prod_qk_rope}else{h3_qk_rope_opt_dispatch_v1})(q.ptr,k.ptr,qw.buffer.ptr,kw.buffer.ptr,
                    self.cos.ptr,self.sin.ptr,qn.ptr,kn.ptr,q.ptr,k.ptr,rows as i64,HEADS as c_int,
                    self.eps(&q_name)?,self.eps(&k_name)?,self.arena.stream) })?;
            }
            (q,k)
        } else {
            self.norm(&format!("{name}.norm_q"), q, qn, rows * HEADS, 128)?;
            self.norm(&format!("{name}.norm_k"), k, kn, rows * HEADS, 128)?;
            (qn,kn)
        };
        drop(fine_qk_gpu);drop(fine_qk_host);
        self.phase_boundary(directory,trace,"q_rope",actual_q,rows,INNER)?;
        self.phase_boundary(directory,trace,"k_rope",actual_k,rows,INNER)?;
        let out = self.s("attn");
        let scale = 1.0f32 / (128.0f32).sqrt();
        if main && self.router.as_ref().is_some_and(|r|r.plan.heads<HEADS) {
            if rows!=self.seq {return Err("grouped attention requires full main sequence".into());}
            self.grouped_attention(actual_q,actual_k,v,out,sparse)?;
        } else if sparse {
            if !main || rows != self.seq { return Err("router only supports full main packed sequence".into()); }
            {
                let _fine_host=HostSpan::new("router",0);
                let _fine_gpu=GpuSpan::new("router",self.arena.stream)?;
            self.router.as_ref().ok_or("full-head router unavailable in CP")?.launch(&self.route_buffers, self.arena.stream)?;
            }
            if let Some(directory)=directory {
                // Read-only diagnosis of the first sparse call, before a bad
                // routing index can hide its producer behind a vendor fault.
                for spec in &self.router.as_ref().ok_or("router unavailable")?.buffers {
                    if spec.input {continue;}
                    let buffer=Buffer{ptr:(self.route_buffers[&spec.name] as usize as *mut c_void),bytes:spec.bytes};
                    let data=self.arena.download(buffer)?;
                    fs::write(directory.join(format!("router_{}.raw",spec.name)),&data)?;
                    if spec.name=="indices" {
                        let invalid=data.chunks_exact(4).filter(|v|{
                            let index=i32::from_le_bytes((*v).try_into().unwrap());
                            index<0 || index as usize>=rows.div_ceil(64)
                        }).count();
                        eprintln!("sparse diagnostic: invalid_indices={invalid}");
                        if invalid>0 {return Err("sparse router produced an out-of-range block index".into());}
                    }
                }
            }
            let workspace = self.s("bsa_ws");
            let profile_host=HostSpan::new("sparse_attention",0);
            let profile_gpu=GpuSpan::new("sparse_attention",self.arena.stream)?;
            cuda(unsafe { h3_bsa_forward_masked(actual_q.ptr, actual_k.ptr, v.ptr,
                self.route_buffers["indices"] as *const i32, self.block_sizes.ptr.cast(), self.query_block_mask.ptr.cast(), out.ptr,
                workspace.ptr, workspace.bytes, rows as c_int, HEADS as c_int, self.kept as c_int, scale, self.arena.stream) })?;
            BSA_CPP_CALLS.fetch_add(1,std::sync::atomic::Ordering::Relaxed);
            self.correct_dense_audio(actual_q,actual_k,v,out,HEADS)?;
            drop(profile_gpu);drop(profile_host);
        } else {
            let workspace = self.s("dense_ws");
            let profile_label=if main{"dense_attention.main"}else{"dense_attention.refiner"};
            let _profile_host=HostSpan::new(profile_label,0);
            let _profile_gpu=GpuSpan::new(profile_label,self.arena.stream)?;
            self.dense.run(actual_q,actual_k,v,out,workspace,rows,if inplace_qkv{2}else if main{0}else{1},scale,self.arena.stream)?;
        }
        self.phase_boundary(directory,trace,"attention",out,rows,INNER)?;
        self.linear(&format!("{name}.to_out.0"), out, self.s("branch"), rows)?;
        Ok(self.s("branch"))
    }
    fn feedforward(&self, name: &str, input: Buffer, rows: usize,
        trace: &mut Option<Vec<Value>>) -> Result<Buffer> {
        let profile_label=if name.starts_with("transformer_blocks."){"ffn.main"}else{"ffn.refiner"};
        let _profile_host=HostSpan::new(profile_label,0);
        let _profile_gpu=GpuSpan::new(profile_label,self.arena.stream)?;
        self.linear(&format!("{name}.net.0.proj"), input, self.s("ff2"), rows)?;
        self.cp_diagnostic_boundary(trace,"ff.net.0.proj",self.s("ff2"),&[rows,2*FF])?;
        {
            let _fine_host=HostSpan::new(if name.starts_with("transformer_blocks."){"ffn.main.swiglu"}else{"ffn.refiner.swiglu"},0);
            let _fine_gpu=GpuSpan::new(if name.starts_with("transformer_blocks."){"ffn.main.swiglu"}else{"ffn.refiner.swiglu"},self.arena.stream)?;
        cuda(unsafe { (if self.numerics.fused() && name.starts_with("transformer_blocks."){h3_prod_swiglu_bf16}else{h3_opt_swiglu_bf16})(self.s("ff2").ptr, self.s("ff").ptr, rows as i64, FF as c_int, self.arena.stream) })?;
        }
        self.cp_diagnostic_boundary(trace,"swiglu",self.s("ff"),&[rows,FF])?;
        self.linear(&format!("{name}.net.2"), self.s("ff"), self.s("branch"), rows)?;
        self.cp_diagnostic_boundary(trace,"ff.net.2",self.s("branch"),&[rows,HIDDEN])?;
        Ok(self.s("branch"))
    }
    fn refiner(&self, trace: bool) -> Result<Buffer> {
        self.gather_weights("global")?;
        let text = self.s("text");
        let directory = self.output.join("diagnostics/refiner");
        let mut boundaries = Vec::new();
        if trace { fs::create_dir_all(&directory)?; }
        self.linear("context_embedder", self.conditioning, text, self.nt)?;
        if trace { self.diagnostic_boundary(&directory, "context_embedder", text, &mut boundaries)?; }
        for layer in 0..2 {
            let name = format!("token_refiner.refiner_blocks.{layer}");
            self.gather_weights(&name)?;
            {
                let _fine_host=HostSpan::new("pointwise.refiner.norm",0);
                let _fine_gpu=GpuSpan::new("pointwise.refiner.norm",self.arena.stream)?;
            self.norm(&format!("{name}.norm1"), text, self.s("norm"), self.nt, HIDDEN)?;
            }
            let branch = self.attention(&format!("{name}.attn"), self.s("norm"), self.nt, false, false)?;
            {
                let _fine_host=HostSpan::new("pointwise.refiner.residual",0);
                let _fine_gpu=GpuSpan::new("pointwise.refiner.residual",self.arena.stream)?;
            cuda(unsafe { h3_add_bf16(text.ptr, branch.ptr, text.ptr, (self.nt * HIDDEN) as i64, self.arena.stream) })?;
            }
            {
                let _fine_host=HostSpan::new("pointwise.refiner.norm",0);
                let _fine_gpu=GpuSpan::new("pointwise.refiner.norm",self.arena.stream)?;
            self.norm(&format!("{name}.norm2"), text, self.s("norm"), self.nt, HIDDEN)?;
            }
            let branch = self.feedforward(&format!("{name}.ff"), self.s("norm"), self.nt, &mut None)?;
            {
                let _fine_host=HostSpan::new("pointwise.refiner.residual",0);
                let _fine_gpu=GpuSpan::new("pointwise.refiner.residual",self.arena.stream)?;
            cuda(unsafe { h3_add_bf16(text.ptr, branch.ptr, text.ptr, (self.nt * HIDDEN) as i64, self.arena.stream) })?;
            }
            if trace { self.diagnostic_boundary(&directory, &format!("refiner{layer:02}_output"), text, &mut boundaries)?; }
        }
        self.gather_weights("global")?;
        {
            let _fine_host=HostSpan::new("pointwise.refiner.norm",0);
            let _fine_gpu=GpuSpan::new("pointwise.refiner.norm",self.arena.stream)?;
        self.norm("token_refiner.final_norm", text, self.s("norm"), self.nt, HIDDEN)?;
        }
        {
            let _fine_host=HostSpan::new("layout.refiner_copy",0);
            let _fine_gpu=GpuSpan::new("layout.refiner_copy",self.arena.stream)?;
        self.arena.copy(text, self.s("norm").prefix(text.bytes)?)?;
        }
        if trace { self.diagnostic_boundary(&directory, "refiner_final_norm", text, &mut boundaries)?; }
        Ok(text)
    }
    fn packed_input(&self, trace_refiner: bool) -> Result<Buffer> {
        self.gather_weights("global")?;
        // Projections keep the reference dtype and order. Their BF16 casts use
        // the norm tail, disjoint from the refiner's first nt normalization rows.
        self.linear("proj_in", self.latent_video, self.s("fp_hidden"), self.nv)?;
        self.cast(self.s("fp_hidden"), self.s("video_embeds"), self.nv * HIDDEN, false)?;
        self.linear("audio_proj_in", self.latent_audio, self.s("fp_hidden"), self.na)?;
        self.cast(self.s("fp_hidden"), self.s("audio_embeds"), self.na * HIDDEN, false)?;
        let text = self.refiner(trace_refiner)?;
        let hidden = self.s("hidden");
        {
            let _fine_host=HostSpan::new("layout.zero",0);
            let _fine_gpu=GpuSpan::new("layout.zero",self.arena.stream)?;
        cuda(unsafe { cudaMemsetAsync(hidden.ptr, 0, hidden.bytes, self.arena.stream) })?;
        }
        self.scatter(text, self.text_indices, hidden, self.nt, HIDDEN, 2, true)?;
        self.scatter(self.s("video_embeds"), self.video_indices, hidden, self.nv, HIDDEN, 2, true)?;
        self.scatter(self.s("audio_embeds"), self.audio_indices, hidden, self.na, HIDDEN, 2, true)?;
        Ok(hidden)
    }
    fn modulation(&self, layer: usize, step: usize) -> Result<[Buffer; 6]> {
        let table = self.buffers.get(&format!("transformer_blocks.{layer}.adaln_proj.table")).ok_or("AdaLN table missing")?;
        let per_step = table.spec.shape[1] * 6 * HIDDEN * 2;
        let base = table.buffer.slice(self.steps[step].cursor * per_step, per_step)?;
        let at = |part: usize| base.slice(part * HIDDEN * 2, per_step - part * HIDDEN * 2);
        Ok([at(0)?, at(1)?, at(2)?, at(3)?, at(4)?, at(5)?])
    }
    fn modulate(&self, input: Buffer, scale: Buffer, shift: Buffer, indices: Buffer, table_stride: usize) -> Result<()> {
        cuda(unsafe { h3_modulate_bf16(input.ptr, scale.ptr, shift.ptr, indices.ptr.cast(),
            self.seq as i64, HIDDEN as c_int, table_stride as c_int, self.arena.stream) })
    }
    fn gated(&self, hidden: Buffer, branch: Buffer, gate: Buffer, indices: Buffer) -> Result<()> {
        cuda(unsafe { h3_gated_bf16(hidden.ptr, branch.ptr, gate.ptr, indices.ptr.cast(), hidden.ptr,
            self.seq as i64, HIDDEN as c_int, (6 * HIDDEN) as c_int, self.arena.stream) })
    }
    fn pw_norm_mod(&self, name: &str, input: Buffer, output: Buffer,
        scale: Buffer, shift: Buffer, indices: Buffer) -> Result<()> {
        let weight = self.w(&format!("{name}.weight"))?;
        if weight.spec.dtype != "torch.bfloat16" || weight.spec.shape != [HIDDEN] {
            return Err(format!("norm weight mismatch: {name}").into());
        }
        if input.bytes < self.seq * HIDDEN * 2 || output.bytes < self.seq * HIDDEN * 2 || indices.bytes < self.seq * 8 {
            return Err("pointwise norm/mod buffer too small".into());
        }
        cuda(unsafe { (if self.numerics.fused(){h3_prod_rms_mod_bf16}else{h3_pw_rms_mod_bf16})(input.ptr, weight.buffer.ptr, scale.ptr, shift.ptr,
            indices.ptr.cast(), output.ptr, self.seq as i64, HIDDEN as c_int,
            (6 * HIDDEN) as c_int, self.eps(name)?, self.arena.stream) })
    }
    fn pw_gated_norm_mod(&self, name: &str, hidden: Buffer, branch: Buffer, gate: Buffer,
        output: Buffer, scale: Buffer, shift: Buffer, indices: Buffer) -> Result<()> {
        let weight = self.w(&format!("{name}.weight"))?;
        if weight.spec.dtype != "torch.bfloat16" || weight.spec.shape != [HIDDEN] {
            return Err(format!("norm weight mismatch: {name}").into());
        }
        if [hidden.bytes, branch.bytes, output.bytes].iter().any(|n| *n < self.seq * HIDDEN * 2) || indices.bytes < self.seq * 8 {
            return Err("pointwise gated norm/mod buffer too small".into());
        }
        cuda(unsafe { (if self.numerics.fused(){h3_prod_gated_rms_mod_bf16}else{h3_pw_gated_rms_mod_bf16})(hidden.ptr, branch.ptr, gate.ptr,
            weight.buffer.ptr, scale.ptr, shift.ptr, indices.ptr.cast(), output.ptr,
            self.seq as i64, HIDDEN as c_int, (6 * HIDDEN) as c_int,
            self.eps(name)?, self.arena.stream) })
    }
    fn pw_gated(&self, hidden: Buffer, branch: Buffer, gate: Buffer, indices: Buffer) -> Result<()> {
        if hidden.bytes < self.seq * HIDDEN * 2 || branch.bytes < self.seq * HIDDEN * 2 || indices.bytes < self.seq * 8 {
            return Err("pointwise gate buffer too small".into());
        }
        cuda(unsafe { h3_pw_gated_bf16(hidden.ptr, branch.ptr, gate.ptr, indices.ptr.cast(),
            hidden.ptr, self.seq as i64, HIDDEN as c_int, (6 * HIDDEN) as c_int, self.arena.stream) })
    }
    fn final_gated_norm(&self,hidden:Buffer,branch:Buffer,gate:Buffer,step:usize)->Result<()> {
        let weight=self.w("norm_out.norm.weight")?;
        let output=self.s("fp_hidden");
        if weight.spec.dtype!="torch.bfloat16" || weight.spec.shape!=[HIDDEN]
            || output.bytes<self.seq*HIDDEN*4 {return Err("final fusion layout mismatch".into());}
        let table=self.steps[step].final_mod;
        let scale=table.slice(HIDDEN*2,table.bytes-HIDDEN*2)?;
        cuda(unsafe{h3_pw_final_gated_norm_mod_cast(hidden.ptr,branch.ptr,gate.ptr,weight.buffer.ptr,
            scale.ptr,table.ptr,self.steps[step].adaln_indices.ptr.cast(),self.steps[step].indices.ptr.cast(),
            output.ptr.cast(),self.seq as i64,HIDDEN as c_int,(6*HIDDEN) as c_int,(2*HIDDEN) as c_int,
            self.eps("norm_out.norm")?,self.arena.stream)})
    }
    fn block(&self,layer:usize,step:usize)->Result<()> { self.block_fused(layer,step,false,false) }
    fn block_fused(&self, layer: usize, step: usize, fuse_next:bool, fuse_final:bool) -> Result<()> {
        self.gather_weights(&format!("transformer_blocks.{layer}"))?;
        if let Some(sharded)=&self.resident_model.sharded{sharded.prefetch_block(layer,self.arena.stream)?;}
        if let Some(cp)=&self.cp{return self.cp_block(cp,layer,step);}
        let name = format!("transformer_blocks.{layer}");
        let hidden = self.s("hidden");
        let trace=layer==0 && crate::debug_audit::enabled()
            && std::env::var("H3_NATIVE_TRACE_BLOCK0").as_deref()==Ok("1")
            && std::env::var("H3_NATIVE_TRACE_STEP").ok().and_then(|v|v.parse().ok())==Some(step);
        let directory=self.output.join(format!("diagnostics/step{step:02}/block00"));
        let directory=trace.then_some(directory.as_path());let mut boundaries=Vec::new();
        self.phase_boundary(directory,&mut boundaries,"packed_input",hidden,self.seq,HIDDEN)?;
        let [shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp] = self.modulation(layer, step)?;
        let index = self.steps[step].adaln_indices;
        {
            let _fine_host=HostSpan::new("pointwise.main.norm1_modulate",0);
            let _fine_gpu=GpuSpan::new("pointwise.main.norm1_modulate",self.arena.stream)?;
        if !fuse_next || layer==0 {
        self.pw_norm_mod(&format!("{name}.norm1"), hidden, self.s("norm"), scale_msa, shift_msa, index)?;
        }
        }
        self.phase_boundary(directory,&mut boundaries,"norm1",self.s("norm"),self.seq,HIDDEN)?;
        let branch = self.attention_trace(&format!("{name}.attn"), self.s("norm"), self.seq, true,
            self.execution.sparse_layer(step,self.seq,layer)?,directory,&mut boundaries)?;
        self.phase_boundary(directory,&mut boundaries,"attention_out",branch,self.seq,HIDDEN)?;
        {
            let _fine_host=HostSpan::new("pointwise.main.gate_norm2_modulate",0);
            let _fine_gpu=GpuSpan::new("pointwise.main.gate_norm2_modulate",self.arena.stream)?;
        self.pw_gated_norm_mod(&format!("{name}.norm2"), hidden, branch, gate_msa,
            self.s("norm"), scale_mlp, shift_mlp, index)?;
        }
        self.phase_boundary(directory,&mut boundaries,"residual",hidden,self.seq,HIDDEN)?;
        self.phase_boundary(directory,&mut boundaries,"norm2",self.s("norm"),self.seq,HIDDEN)?;
        let branch = self.feedforward(&format!("{name}.ff"), self.s("norm"), self.seq, &mut None)?;
        self.phase_boundary(directory,&mut boundaries,"ff_projection",self.s("ff2"),self.seq,2*FF)?;
        self.phase_boundary(directory,&mut boundaries,"swiglu",self.s("ff"),self.seq,FF)?;
        self.phase_boundary(directory,&mut boundaries,"ff_out",branch,self.seq,HIDDEN)?;
        {
            let _fine_host=HostSpan::new("pointwise.main.gate_residual",0);
            let _fine_gpu=GpuSpan::new("pointwise.main.gate_residual",self.arena.stream)?;
        if layer+1==crate::execution::MAIN_LAYERS && fuse_final {
            self.final_gated_norm(hidden,branch,gate_mlp,step)?;
        } else if fuse_next && layer+1<crate::execution::MAIN_LAYERS {
            let [next_shift,next_scale,_,_,_,_]=self.modulation(layer+1,step)?;
            // Same gate/add BF16 boundary and same four-warp RMS reduction.
            // Produce the next block's norm1 directly while hidden is live.
            self.pw_gated_norm_mod(&format!("transformer_blocks.{}.norm1",layer+1),
                hidden,branch,gate_mlp,self.s("norm"),next_scale,next_shift,index)?;
        } else { self.pw_gated(hidden, branch, gate_mlp, index)?; }
        }
        self.phase_boundary(directory,&mut boundaries,"block_output",hidden,self.seq,HIDDEN)
    }
    fn phase_boundary(&self,directory:Option<&Path>,trace:&mut Vec<Value>,name:&str,
        buffer:Buffer,rows:usize,cols:usize)->Result<()> {
        let Some(directory)=directory else{return Ok(());};
        fs::create_dir_all(directory)?;
        let data=self.arena.download(buffer.prefix(rows*cols*2)?)?;
        let prefix=&data[..rows.min(64)*cols*2];
        fs::write(directory.join(format!("{name}.prefix64.raw")),prefix)?;
        trace.push(json!({"name":name,"dtype":"torch.bfloat16","shape":[rows,cols],
            "bytes":data.len(),"sha256":crate::integrity::digest(&data),"prefix_rows":rows.min(64)}));
        fs::write(directory.join("trace.json"),serde_json::to_vec_pretty(trace)?)?;
        Ok(())
    }
    fn trace_reference(&mut self, root: &Path) -> Result<TraceReference> {
        // Read metadata only. No reference activation file is opened or uploaded.
        let report: Value = serde_json::from_slice(&crate::startup_trace::read(root.join("report.json"))?)?;
        if report["passed"] != true || report["step"] != 2 ||
            report["scope"] != "offline frozen-step validation replay; not native continuous inference" ||
            report["runs"][0]["matches_frozen_gold"] != true {
            return Err("trace reference requires a passing original frozen step-2 replay".into());
        }
        let layers = report["runs"][0]["layers"].as_array().ok_or("trace reference has no layers")?;
        if layers.len() != 50 { return Err("trace reference must contain all 50 layers".into()); }
        let mut layer_crc32 = Vec::with_capacity(50);
        for (layer, entry) in layers.iter().enumerate() {
            if entry["layer"] != layer || entry["dtype"] != "torch.bfloat16" ||
                entry["shape"] != json!([1, self.seq, HIDDEN]) || entry["bytes"] != self.s("hidden").bytes {
                return Err(format!("trace reference layer {layer} layout/inventory mismatch").into());
            }
            let crc = entry["crc32"].as_u64().ok_or("trace reference CRC must be an unsigned integer")?;
            layer_crc32.push(u32::try_from(crc).map_err(|_| "trace reference CRC exceeds u32")?);
        }
        Ok(TraceReference { root: root.to_owned(), layer_crc32,
            gold_manifest_sha256: report["gold_manifest_sha256"].clone() })
    }

    fn save_first_divergence(&self, reference: &TraceReference, layer: usize, actual_crc: u32, input: &[u8]) -> Result<()> {
        let directory = self.output.join("diagnostics/step02/first_divergence");
        fs::create_dir_all(&directory)?;
        let report_path = directory.join("first_divergence.json");
        let mut report = json!({"mode":"continuous_infer_diagnostic","step":2,"layer":layer,
            "passed":false,"completed":false,"reference_root":reference.root,"reference_repeat":0,
            "reference_gold_manifest_sha256":reference.gold_manifest_sha256,
            "reference_crc32":format!("{:08x}", reference.layer_crc32[layer]),
            "actual_crc32":format!("{actual_crc:08x}"),
            "scope":"first differing block; all raw files are computed native live buffers",
            "reference_activations_read":false,"reference_activations_uploaded":false,
            "sampler_update_performed_for_step":false,"tensors":{}});
        fs::write(&report_path, serde_json::to_vec_pretty(&report)?)?;
        if self.cp.is_some() {
            // CP scratch holds owned rows/local heads, not the full-head
            // standalone tensors below. Preserve the global mismatch boundary.
            fs::write(directory.join("block_input.raw"),input)?;
            fs::write(directory.join("block_output.raw"),self.arena.download(self.s("hidden"))?)?;
            report["context_parallel"]=self.cp_report();
            report["scope"]=json!("CP global block input/output; operator traces use H3_NATIVE_TRACE_CP_BLOCK0");
            report["completed"]=json!(true);
            fs::write(&report_path,serde_json::to_vec_pretty(&report)?)?;
            return Ok(());
        }
        let router_heads=self.router.as_ref().ok_or("diagnostic router missing")?.plan.heads;
        let indices = Buffer { ptr: self.route_buffers["indices"] as *mut c_void,
            bytes: router_heads * self.seq.div_ceil(64) * self.kept * 4 };
        report["router_head_range"]=json!([HEADS-router_heads,HEADS]);
        for (name, buffer, shape, dtype) in [
            ("block_input", self.s("hidden"), vec![1, self.seq, HIDDEN], "torch.bfloat16"),
            ("block_output", self.s("hidden"), vec![1, self.seq, HIDDEN], "torch.bfloat16"),
            ("norm2_modulated", self.s("norm"), vec![1, self.seq, HIDDEN], "torch.bfloat16"),
            ("q_after_rope", self.s("q"), vec![1, self.seq, HEADS, 128], "torch.bfloat16"),
            ("k_after_rope", self.s("k"), vec![1, self.seq, HEADS, 128], "torch.bfloat16"),
            ("v", self.s("v"), vec![1, self.seq, HEADS, 128], "torch.bfloat16"),
            ("q_norm", self.s("qn"), vec![1, self.seq, HEADS, 128], "torch.bfloat16"),
            ("k_norm", self.s("kn"), vec![1, self.seq, HEADS, 128], "torch.bfloat16"),
            ("router_indices", indices, vec![1, router_heads, self.seq.div_ceil(64), self.kept], "torch.int32"),
            ("bsa_output", self.s("attn"), vec![1, self.seq, HEADS, 128], "torch.bfloat16"),
            ("ff.net.0.proj", self.s("ff2"), vec![1, self.seq, 2 * FF], "torch.bfloat16"),
            ("swiglu", self.s("ff"), vec![1, self.seq, FF], "torch.bfloat16"),
            ("ff.net.2", self.s("branch"), vec![1, self.seq, HIDDEN], "torch.bfloat16"),
        ] {
            let file = format!("{name}.raw");
            let data = if name == "block_input" { input.to_vec() } else { self.arena.download(buffer)? };
            fs::write(directory.join(&file), &data)?;
            let mut stride = vec![1; shape.len()];
            for i in (1..shape.len()).rev() { stride[i - 1] = stride[i] * shape[i]; }
            let record = json!({"file":file,"dtype":dtype,"shape":shape,"stride":stride,
                "bytes":data.len(),"crc32":format!("{:08x}", crate::diagnostic_crc32(&data)),
                "contiguous":true,"storage_offset":0});
            eprintln!("native first divergence captured {name}: {record}");
            report["tensors"][name] = record;
            fs::write(&report_path, serde_json::to_vec_pretty(&report)?)?;
        }
        report["completed"] = json!(true);
        fs::write(&report_path, serde_json::to_vec_pretty(&report)?)?;
        Ok(())
    }

    fn forward(&mut self, step: usize, reference_root: Option<&Path>) -> Result<()> {
        native_core_infra::execution_start::arm();
        if let Some(sharded)=&self.resident_model.sharded{sharded.begin_step();}
        let profile_label="forward";
        let _profile_host=HostSpan::new(profile_label,0);
        let _profile_gpu=GpuSpan::new(profile_label,self.arena.stream)?;
        let reference = if let Some(root) = reference_root {
            if step != 2 { return Err("reference trace only supports continuous step 2".into()); }
            Some(self.trace_reference(root)?)
        } else { None };
        let trace = reference.is_some() || (crate::debug_audit::enabled() && std::env::var("H3_NATIVE_TRACE_STEP").ok()
            .and_then(|value| value.parse::<usize>().ok()) == Some(step));
        let trace_limit = if trace && reference.is_none() {
            std::env::var("H3_NATIVE_TRACE_LAYER_LIMIT").ok().map(|v| v.parse::<usize>()).transpose()?.unwrap_or(50)
        } else { 50 };
        if !(1..=50).contains(&trace_limit) { return Err("trace layer limit must be 1..50".into()); }
        // Limited traces still gather on every rank. Only rank0 downloads the
        // common global boundaries, avoiding eight redundant GiB-sized copies.
        let trace_writer = trace_limit == 50 || self.cp.as_ref().map_or(true, |cp| cp.group.rank == 0);
        let trace_refiner = trace && trace_writer && crate::debug_audit::enabled()
            && std::env::var("H3_NATIVE_TRACE_REFINER").as_deref() == Ok("1");
        self.packed_input(trace_refiner)?;
        let mut trace_entries = vec![];
        let trace_buffer = |name: &str, buffer: Buffer, entries: &mut Vec<Value>| -> Result<(u32, Vec<u8>)> {
            let data = self.arena.download(buffer)?;
            let crc = crate::diagnostic_crc32(&data);
            let entry = json!({"name":name,"bytes":data.len(),"sha256":crate::integrity::digest(&data),
                "crc32":format!("{crc:08x}")});
            eprintln!("native step {step} trace: {entry}");
            entries.push(entry);
            let directory = self.output.join(format!("diagnostics/step{step:02}"));
            fs::create_dir_all(&directory)?;
            fs::write(directory.join("trace.json"), serde_json::to_vec_pretty(entries)?)?;
            Ok((crc, data))
        };
        // Retain the already-downloaded preceding trace as diagnostic input.
        // This adds no GPU allocation or copy kernel between model blocks.
        let mut preceding_hidden = if trace && trace_writer {
            let (_, data) = trace_buffer("packed_input", self.s("hidden"), &mut trace_entries)?;
            if reference.is_some() { Some(data) } else { None }
        } else { None };
        let mut diagnostics = json!({"step":step,
            "scope":"computed native intermediates; gold readback comparison only", "gates":[]});
        // Reference mode compares CRC metadata only; it does not read the
        // existing standalone packed-input/block0 activation fixtures.
        let fixture_gate = self.validate_reference && reference.is_none() && (step == 5 || trace && [0, 2].contains(&step));
        if fixture_gate {
            let actual = self.write(&format!("diagnostics/step{step:02}/packed_input.raw"), self.s("hidden"))?;
            let comparison = self.validation(&actual,
                &self.manifest["fixtures"][format!("block0_step{step}")]["inputs"]["hidden_states"])?;
            diagnostics["gates"].as_array_mut().unwrap().push(json!({"name":"packed_input","comparison":comparison}));
            fs::write(self.output.join(format!("diagnostics/step{step:02}/report.json")), serde_json::to_vec_pretty(&diagnostics)?)?;
            eprintln!("native step {step} packed-input diagnostic: {comparison}");
            if comparison["passed"] != true { return Err(format!("step {step} computed packed input differs from frozen fixture").into()); }
        }
        let can_fuse=self.numerics.sglang() && self.cp.is_none()
            && self.resident_model.sharded.is_none() && !self.validate_reference
            && !crate::debug_audit::enabled() && reference_root.is_none()
            && crate::pointwise_opt::enabled();
        let fuse_next=can_fuse && cross_layer_fusion_enabled();
        let fuse_final=can_fuse && final_norm_fusion_enabled();
        for layer in 0..crate::execution::MAIN_LAYERS {
            self.block_fused(layer,step,fuse_next,fuse_final)?;
            if trace && layer < trace_limit || fixture_gate && layer==0 {self.cp_gather_hidden()?;}
            if trace && trace_writer && layer < trace_limit {
                let (crc, data) = trace_buffer(&format!("block{layer:02}_output"), self.s("hidden"), &mut trace_entries)?;
                if let Some(reference) = &reference {
                    if crc != reference.layer_crc32[layer] {
                        self.save_first_divergence(reference, layer, crc,
                            preceding_hidden.as_deref().ok_or("missing preceding native trace")?)?;
                        return Err(format!("diagnostic mismatch: continuous step 2 layer {layer}, native CRC {crc:08x} != reference {:08x}; live buffers saved; sampler not updated",
                            reference.layer_crc32[layer]).into());
                    }
                    preceding_hidden = Some(data);
                }
            }
            if fixture_gate && layer == 0 {
                let actual = self.write(&format!("diagnostics/step{step:02}/block00_output.raw"), self.s("hidden"))?;
                let comparison = self.validation(&actual, &self.manifest["fixtures"][format!("block0_step{step}")]["output"])?;
                diagnostics["gates"].as_array_mut().unwrap().push(json!({"name":"block00_output","comparison":comparison}));
                fs::write(self.output.join(format!("diagnostics/step{step:02}/report.json")), serde_json::to_vec_pretty(&diagnostics)?)?;
                eprintln!("native step {step} block0 diagnostic: {comparison}");
                if comparison["passed"] != true { return Err(format!("step {step} computed block0 output differs from frozen fixture").into()); }
            }
            if layer % 10 == 9 {
                { let _fine_host=HostSpan::new("host.block_sync",0); self.arena.sync()?; }
                if crate::debug_audit::enabled() || self.validate_reference {
                    let _fine_host=HostSpan::new("host.block_log",0);
                    eprintln!("native step {step} block {layer} complete");
                }
            }
        }
        if let Some(reference) = &reference {
            fs::write(self.output.join("diagnostics/step02/reference_crc_gate.json"), serde_json::to_vec_pretty(&json!({
                "step":2,"layers_compared":50,"all_layer_crc_match":true,"reference_root":reference.root,
                "reference_activations_read":false,"reference_activations_uploaded":false,
                "scope":"localization only; normal full-output numerical gates still required"}))?)?;
        }
        self.cp_gather_hidden()?;
        self.gather_weights("global")?;
        if self.validate_reference && step == 5 { self.write("diagnostics/step05/block49_output.raw", self.s("hidden"))?; }
        {
            let _fine_host=HostSpan::new("pointwise.final_norm_modulate",0);
            let _fine_gpu=GpuSpan::new("pointwise.final_norm_modulate",self.arena.stream)?;
        if !fuse_final {
        self.norm("norm_out.norm", self.s("hidden"), self.s("norm"), self.seq, HIDDEN)?;
        let table = self.steps[step].final_mod;
        self.modulate(self.s("norm"), table.slice(HIDDEN * 2, table.bytes - HIDDEN * 2)?, table,
            self.steps[step].indices, 2 * HIDDEN)?;
        }
        }
        if self.validate_reference && step == 5 { self.write("diagnostics/step05/final_norm_modulated.raw", self.s("norm"))?; }
        if !fuse_final { self.cast(self.s("norm"), self.s("fp_hidden"), self.seq * HIDDEN, true)?; }
        self.linear("proj_out", self.s("fp_hidden"), self.s("all_video"), self.seq)?;
        if self.validate_reference && step == 5 { self.write("diagnostics/step05/video_head_all.raw", self.s("all_video"))?; }
        self.scatter(self.s("all_video"), self.video_indices, self.s("velocity_video"), self.nv, VIDEO, 4, false)?;
        self.linear("audio_proj_out", self.s("fp_hidden"), self.s("all_audio"), self.seq)?;
        if self.validate_reference && step == 5 {
            self.write("diagnostics/step05/audio_head_all.raw", self.s("all_audio"))?;
            diagnostics["final_intermediates_written"] = json!(true);
            fs::write(self.output.join("diagnostics/step05/report.json"), serde_json::to_vec_pretty(&diagnostics)?)?;
        }
        self.scatter(self.s("all_audio"), self.audio_indices, self.s("velocity_audio"), self.na, AUDIO, 4, false)?;
        let _fine_host=HostSpan::new("host.forward_final_sync",0);
        self.arena.sync()?;
        if let Some(sharded)=&self.resident_model.sharded{sharded.end_step(self.execution.attention_index(step)?)?;}
        Ok(())
    }
    fn update(&self, step: usize) -> Result<()> {
        let profile_label="sampler";
        let _profile_host=HostSpan::new(profile_label,0);
        let _profile_gpu=GpuSpan::new(profile_label,self.arena.stream)?;
        let s = self.steps[step];
        for (sample, velocity, rows, conditions, channels, timestep, ratio) in [
            (self.latent_video, self.s("velocity_video"), self.nv, self.condition_video, VIDEO, s.video_t, s.video_ratio),
            (self.latent_audio, self.s("velocity_audio"), self.na, self.condition_audio, AUDIO, s.audio_t, s.audio_ratio),
        ] {
            let offset = conditions * channels * 4;
            let elements = (rows - conditions) * channels;
            if elements == 0 { continue; }
            let sample = sample.slice(offset, elements * 4)?;
            let velocity = velocity.slice(offset, elements * 4)?;
            cuda(unsafe { h3_euler_f32(sample.ptr.cast(), velocity.ptr.cast(), sample.ptr.cast(),
                elements as i64, 1.0f32 - timestep, ratio, self.arena.stream) })?;
        }
        let _fine_host=HostSpan::new("host.sampler_sync",0);
        self.arena.sync()
    }
    fn write(&self, name: &str, buffer: Buffer) -> Result<Vec<u8>> {
        let _profile=HostSpan::new("output.download_write",buffer.bytes);
        let path = self.output.join(name);
        fs::create_dir_all(path.parent().ok_or("no output parent")?)?;
        let data = self.arena.download(buffer)?;
        fs::write(path, &data)?;
        Ok(data)
    }
    fn validation(&self, actual: &[u8], spec: &Value) -> Result<Value> {
        if !self.validate_reference {return Err("ordinary generation cannot read gold activation tensors".into());}
        let spec = TensorSpec::parse(spec)?;
        compare(actual, &spec.read(&self.root)?, spec.width()?)
    }
    fn audit(&self, phase: &str) -> Result<()> {
        if !crate::debug_audit::enabled() && !self.validate_reference{return Ok(());}
        let maps = fs::read_to_string("/proc/self/maps")?;
        fs::write(self.output.join(format!("native.{phase}.maps.txt")), &maps)?;
        if ["libtorch", "libpython", "libaten", "libc10"].iter().any(|x| maps.to_lowercase().contains(x)) {
            return Err("forbidden Python/ATen/libtorch/c10 dependency in native process".into());
        }
        Ok(())
    }

    fn diagnostic_boundary(&self, directory: &Path, name: &str, buffer: Buffer,
        trace: &mut Vec<Value>) -> Result<()> {
        let data = self.arena.download(buffer)?;
        let entry = json!({"name":name,"bytes":data.len(),"sha256":crate::integrity::digest(&data),
            "crc32":format!("{:08x}", crate::diagnostic_crc32(&data))});
        eprintln!("native standalone block diagnostic: {entry}");
        trace.push(entry);
        fs::write(directory.join("trace.json"), serde_json::to_vec_pretty(trace)?)?;
        Ok(())
    }

    fn block_diagnostic(&self, step: usize) -> Result<Value> {
        if self.router.as_ref().is_some_and(|r|r.plan.heads!=HEADS) {
            return Err("legacy standalone blockdiag requires a full-head fixture; use live phase boundaries for grouped attention".into());
        }
        if self.numerics == crate::numerics::Numerics::ProductionFused {
            return Err("legacy blockdiag fixtures use eager-bf16-v1; native-fused diagnostics require a matching oracle".into());
        }
        if self.cp.is_some(){return Err("standalone blockdiag requires world=1".into());}
        // This bounded validation mode deliberately starts from a captured
        // intermediate. No part of the continuous forward/infer path calls it.
        if step != 2 { return Err("blockdiag supports frozen step 2, layer 33 only".into()); }
        let fixture_root = PathBuf::from(std::env::var("H3_NATIVE_BLOCK_FIXTURE_ROOT")
            .map_err(|_| "blockdiag requires H3_NATIVE_BLOCK_FIXTURE_ROOT")?);
        let fixture: Value = serde_json::from_slice(&crate::startup_trace::read(fixture_root.join("report.json"))?)?;
        if fixture["passed"] != true || fixture["step"] != 2 {
            return Err("blockdiag requires a completed exact original step-2 replay".into());
        }
        let layer = 33;
        let source = &fixture["runs"][0]["layers"];
        if source[layer - 1]["layer"] != layer - 1 || source[layer]["layer"] != layer {
            return Err("blockdiag replay layer inventory mismatch".into());
        }
        let input_spec = TensorSpec::parse(&source[layer - 1]["tensor"])?;
        let output_spec = TensorSpec::parse(&source[layer]["tensor"])?;
        for spec in [&input_spec, &output_spec] {
            if spec.dtype != "torch.bfloat16" || spec.shape != [1, self.seq, HIDDEN] {
                return Err("blockdiag fixture shape/dtype mismatch".into());
            }
        }
        let input = input_spec.read(&fixture_root)?;
        let expected = output_spec.read(&fixture_root)?;
        let name = format!("transformer_blocks.{layer}");
        let attn = format!("{name}.attn");
        if self.manifest["modules"][&attn]["fused_projections"] != false {
            return Err("blockdiag is restricted to the actual unfused-QKV export".into());
        }
        let started = Instant::now();
        let mut report = json!({"mode":"blockdiag","count":step,"step":step,"layer":layer,
            "scope":"offline standalone native block operator-boundary diagnostic",
            "native_input_policy":"explicit later gold block input; never used by continuous inference",
            "later_gold_activations_used_for_comparison_only":false,
            "fixture_root":fixture_root,"input_fixture":input_spec.file,"output_fixture":output_spec.file,
            "load_seconds":self.load_seconds,"model_load_seconds":self.model_load_seconds,
            "clean_native_dependencies":true,"passed":false,"runs":[]});
        let mut runs = vec![];
        let mut all_exact = true;
        for repeat in 0..3 {
            let directory = self.output.join(format!("repeat{repeat}"));
            fs::create_dir_all(&directory)?;
            let mut trace = vec![];
            let boundary = |label: &str, buffer: Buffer, trace: &mut Vec<Value>|
                self.diagnostic_boundary(&directory, label, buffer, trace);
            let hidden = self.s("hidden");
            self.arena.write(hidden, &input)?;
            boundary("block_input", hidden, &mut trace)?;
            let [shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp] = self.modulation(layer, step)?;
            let index = self.steps[step].adaln_indices;
            self.norm(&format!("{name}.norm1"), hidden, self.s("norm"), self.seq, HIDDEN)?;
            self.modulate(self.s("norm"), scale_msa, shift_msa, index, 6 * HIDDEN)?;
            boundary("norm1_modulated", self.s("norm"), &mut trace)?;

            // Same order, buffers, streams and C ABIs as attention()/block().
            // Only readback boundaries are added in this separate mode.
            let (mut q, mut k, mut v, qn, kn) = (self.s("q"), self.s("k"), self.s("v"), self.s("qn"), self.s("kn"));
            for (suffix, output) in [("to_q", q), ("to_k", k), ("to_v", v)] {
                self.linear(&format!("{attn}.{suffix}"), self.s("norm"), output, self.seq)?;
                boundary(&format!("attn.{suffix}"), output, &mut trace)?;
            }
            self.norm(&format!("{attn}.norm_q"), q, qn, self.seq * HEADS, 128)?;
            self.norm(&format!("{attn}.norm_k"), k, kn, self.seq * HEADS, 128)?;
            cuda(unsafe { h3_rope_bf16(qn.ptr, self.cos.ptr, self.sin.ptr, q.ptr,
                self.seq as i64, HEADS as c_int, 96, self.arena.stream) })?;
            cuda(unsafe { h3_rope_bf16(kn.ptr, self.cos.ptr, self.sin.ptr, k.ptr,
                self.seq as i64, HEADS as c_int, 96, self.arena.stream) })?;
            boundary("q_after_rope", q, &mut trace)?;
            boundary("k_after_rope", k, &mut trace)?;
            {
                let _fine_host=HostSpan::new("router",0);
                let _fine_gpu=GpuSpan::new("router",self.arena.stream)?;
            self.router.as_ref().ok_or("full-head router unavailable in CP")?.launch(&self.route_buffers, self.arena.stream)?;
            }
            let indices = Buffer { ptr: self.route_buffers["indices"] as *mut c_void,
                bytes: HEADS * self.seq.div_ceil(64) * self.kept * 4 };
            boundary("router_indices", indices, &mut trace)?;
            let workspace = self.s("bsa_ws");
            cuda(unsafe { h3_bsa_forward_masked(q.ptr, k.ptr, v.ptr, indices.ptr.cast(),
                self.block_sizes.ptr.cast(), self.query_block_mask.ptr.cast(), self.s("attn").ptr, workspace.ptr, workspace.bytes,
                self.seq as c_int, HEADS as c_int, self.kept as c_int,
                1.0f32 / (128.0f32).sqrt(), self.arena.stream) })?;
            BSA_CPP_CALLS.fetch_add(1,std::sync::atomic::Ordering::Relaxed);
            boundary("bsa_output", self.s("attn"), &mut trace)?;
            self.linear(&format!("{attn}.to_out.0"), self.s("attn"), self.s("branch"), self.seq)?;
            boundary("attn.to_out.0", self.s("branch"), &mut trace)?;
            self.gated(hidden, self.s("branch"), gate_msa, index)?;
            boundary("hidden_after_attention", hidden, &mut trace)?;
            self.norm(&format!("{name}.norm2"), hidden, self.s("norm"), self.seq, HIDDEN)?;
            self.modulate(self.s("norm"), scale_mlp, shift_mlp, index, 6 * HIDDEN)?;
            boundary("norm2_modulated", self.s("norm"), &mut trace)?;
            self.linear(&format!("{name}.ff.net.0.proj"), self.s("norm"), self.s("ff2"), self.seq)?;
            boundary("ff.net.0.proj", self.s("ff2"), &mut trace)?;
            cuda(unsafe { h3_swiglu_bf16(self.s("ff2").ptr, self.s("ff").ptr,
                self.seq as i64, FF as c_int, self.arena.stream) })?;
            boundary("swiglu", self.s("ff"), &mut trace)?;
            self.linear(&format!("{name}.ff.net.2"), self.s("ff"), self.s("branch"), self.seq)?;
            boundary("ff.net.2", self.s("branch"), &mut trace)?;
            self.gated(hidden, self.s("branch"), gate_mlp, index)?;
            boundary("block_output", hidden, &mut trace)?;
            let actual = self.write(&format!("repeat{repeat}/block_output.raw"), hidden)?;
            let comparison = compare(&actual, &expected, 2)?;
            all_exact &= comparison["passed"] == true;
            let run = json!({"repeat":repeat,"trace":trace,"comparison":comparison});
            eprintln!("native standalone block repeat {repeat}: {}", run["comparison"]);
            runs.push(run);
            report["runs"] = json!(runs);
            fs::write(self.output.join("report.partial.json"), serde_json::to_vec_pretty(&report)?)?;
        }
        self.arena.sync()?;
        self.audit("complete")?;
        report["passed"] = json!(all_exact);
        report["wall_seconds"] = json!(started.elapsed().as_secs_f64());
        fs::write(self.output.join("report.json"), serde_json::to_vec_pretty(&report)?)?;
        Ok(report)
    }

    fn suite(&mut self, count: usize) -> Result<Value> {
        if count != self.steps.len() { return Err("suite requires the complete recipe schedule".into()); }
        let output = self.output.clone();
        let started = Instant::now();
        let mut gates = vec![];
        for (name, mode, count) in [("refiner", "refiner", 0), ("block0", "block", 0),
            ("block2", "block", 2), ("block5", "block", 5), ("forward", "forward", 0)] {
            self.output = output.join("preflight").join(name);
            fs::create_dir_all(&self.output)?;
            let result = self.execute(mode, count);
            self.output = output.clone();
            let result = match result {
                Ok(result) => result,
                Err(error) => {
                    let result = json!({"mode":mode,"count":count,"passed":false,"error":error.to_string()});
                    fs::write(output.join("preflight").join(name).join("report.json"),
                        serde_json::to_vec_pretty(&result)?)?;
                    result
                },
            };
            let gate = json!({"name":name,"mode":mode,"count":count,"passed":result["passed"],
                "results":result["results"],"error":result["error"],
                "report":format!("preflight/{name}/report.json")});
            eprintln!("native suite preflight: {gate}");
            gates.push(gate);
            if result["passed"] != true {
                let report = json!({"mode":"suite","requested_mode":"suite","count":count,"passed":false,
                    "preflight_gates":gates,"preflight_seconds":started.elapsed().as_secs_f64(),
                    "load_seconds":self.load_seconds,"model_load_seconds":self.model_load_seconds,
                    "failure":"preflight gate failed; continuous inference was not started"});
                fs::write(output.join("report.json"), serde_json::to_vec_pretty(&report)?)?;
                return Ok(report);
            }
        }
        let preflight_seconds = started.elapsed().as_secs_f64();
        // Refiner/block/forward only use scratch; the sampler has not modified
        // either initial latent allocation. Inference always starts from root
        // conditioning/noise, and no fixture activation is copied into it.
        let mut report = self.execute("infer", count)?;
        report["requested_mode"] = json!("suite");
        report["preflight_gates"] = json!(gates);
        report["preflight_seconds"] = json!(preflight_seconds);
        report["suite_wall_seconds"] = json!(started.elapsed().as_secs_f64());
        fs::write(output.join("report.json"), serde_json::to_vec_pretty(&report)?)?;
        Ok(report)
    }

    pub fn execute(&mut self, mode: &str, count: usize) -> Result<Value> {
        if let Some(request)=self.long_request.take(){
            if mode!="generate" || count!=self.steps.len(){return Err("long video requires the FL2VA generation recipe".into());}
            return self.execute_long(*request);
        }
        let transaction=crate::request_transaction::Phase::from_request(&self.native_request["preparation"])?;
        if transaction.is_some() && mode!="generate"{return Err("cache transactions require gold-free generation".into());}
        if let Some(sharded)=&self.resident_model.sharded{sharded.begin_request(transaction.as_ref())?;}
        if (mode=="generate") == self.validate_reference {
            return Err("generate requires the gold-free constructor; validation modes require reference-enabled construction".into());
        }
        if mode == "suite" { return self.suite(count); }
        if mode == "blockdiag" { return self.block_diagnostic(count); }
        // The text refiner touches only norm[0..nt]. The remaining rows are
        // dead until main-block normalization, after all modalities have been
        // scattered into hidden. Preserve every projection/refiner/scatter op
        // and its order without dedicated video/audio staging allocations.
        if !self.scratch.contains_key("video_embeds") {
            if self.validate_reference {
                self.scratch.insert("video_embeds", self.arena.alloc(self.nv * HIDDEN * 2)?);
                self.scratch.insert("audio_embeds", self.arena.alloc(self.na * HIDDEN * 2)?);
            }else{
                let views=crate::scratch_plan::embedding_aliases(self.seq,self.nt,self.nv,self.na)?;
                let norm=self.s("norm");
                self.scratch.insert("video_embeds",norm.slice(views.video_offset,views.video_bytes)?);
                self.scratch.insert("audio_embeds",norm.slice(views.audio_offset,views.audio_bytes)?);
            }
        }
        let start = Instant::now();
        let mut report = json!({"mode":mode, "count":count, "passed":false,
            "numerics":self.numerics.name(),
            "execution":self.execution.json(),
            
            "sparse_query_policy":"pure-video blocks; text/audio/mixed blocks dense",
            "scope":"native joint DiT and sampler; conditioning and decoding are external components",
            "load_seconds":self.load_seconds,"model_load_seconds":self.model_load_seconds,
            "model_read_parallelism":model_read_parallelism()?,
            "shape":{"sequence":self.seq,"video":self.nv,"audio":self.na,"text":self.nt},
            "native_input_files":self.native_input_files,
            "native_request":self.native_request,
            "later_gold_activations_used_for_comparison_only":self.validate_reference,
            "reference_validation_enabled":self.validate_reference,
            "ordinary_generation":!self.validate_reference,
            "clean_native_dependencies":if crate::debug_audit::enabled()||self.validate_reference{json!(true)}else{Value::Null},"debug_audit":crate::debug_audit::record(), "results":[]});
        match mode {
            "refiner" => {
                let output = self.refiner(false)?;
                let actual = self.write("refiner.native.raw", output)?;
                let spec = TensorSpec::parse(&self.manifest["fixtures"]["block0_step0"]["inputs"]["hidden_states"])?;
                let packed = spec.read(&self.root)?;
                let mut expected = Vec::with_capacity(actual.len());
                for &row in &self.text_indices_host {
                    let offset = row as usize * HIDDEN * 2;
                    expected.extend_from_slice(&packed[offset..offset + HIDDEN * 2]);
                }
                let comparison = compare(&actual, &expected, 2)?;
                report["passed"] = comparison["passed"].clone();
                report["results"] = json!([comparison]);
            },
            "block" => {
                if ![0, 2, 5].contains(&count) { return Err("block fixture exists only for steps 0, 2, 5".into()); }
                let name = format!("block0_step{count}");
                let fixture = &self.manifest["fixtures"][&name];
                let spec = TensorSpec::parse(&fixture["inputs"]["hidden_states"])?;
                self.arena.write(self.s("hidden"), &spec.read(&self.root)?)?;
                self.block(0, count)?;
                let actual = self.write(&format!("{name}.native.raw"), self.s("hidden"))?;
                let comparison = self.validation(&actual, &fixture["output"])?;
                report["passed"] = comparison["passed"].clone();
                report["results"] = json!([comparison]);
                report["native_input_policy"] = json!("explicit standalone block replay uses identical dumped block input");
            },
            "forward" | "infer" | "generate" => {
                let start_step=if mode=="generate"{self.execution.start}else{0};
                let steps = if mode == "forward" { 1 } else { count };
                if !(1..=self.steps.len()).contains(&steps) { return Err("infer count exceeds recipe schedule".into()); }
                // Suite reaches this through its continuous infer phase; its
                // standalone preflights never enable reference tracing.
                let reference_root = if mode == "infer" {
                    std::env::var_os("H3_NATIVE_TRACE_REFERENCE_ROOT").map(PathBuf::from)
                } else { None };
                if reference_root.is_some() && steps < 3 {
                    return Err("reference trace requires continuous infer with at least three steps".into());
                }
                if let Some(root) = &reference_root { report["trace_reference_root"] = json!(root); }
                let mut results = vec![];
                let mut all_exact = true;
                // Measure completed denoising, outside model/input loading and
                // final latent download. Debug capture is never a warm sample.
                self.arena.sync()?;
                let denoise_started = Instant::now();
                if let Some(time)=crate::startup_trace::timestamp(){report["denoise_start_monotonic_seconds"]=json!(time);}
                for step in start_step..start_step+steps {
                    let step_started=Instant::now();
                    profile::set_step(step as i32);
                    let profile_step=HostSpan::new("step_total",0);
                    self.forward(step, reference_root.as_deref().filter(|_| step == 2))?;
                    if mode=="generate" && !crate::debug_audit::enabled() {
                        self.update(step)?;
                        self.log_step(step,step_started)?;
                        results.push(json!({"step":step,"attention_evaluation":self.execution.attention_index(step)?,
                            "audio_frozen":self.condition_audio==self.na,"audit_skipped":true,"endpoint_downloads":0}));
                        drop(profile_step);
                        profile::collect_step()?;
                        continue;
                    }
                    let prefix = format!("steps/{step:02}");
                    let video = self.write(&format!("{prefix}/video_velocity.raw"), self.s("velocity_video"))?;
                    let audio = self.write(&format!("{prefix}/audio_velocity.raw"), self.s("velocity_audio"))?;
                    let vg = if self.validate_reference {self.validation(&video, &self.manifest["steps"][step]["outputs"]["noise_pred"])?}else{finite_f32(&video)?};
                    let ag = if self.validate_reference {self.validation(&audio, &self.manifest["steps"][step]["outputs"]["audio_noise_pred"])?}else{finite_f32(&audio)?};
                    all_exact &= vg["passed"] == true && ag["passed"] == true;
                    let mut result = json!({"step":step,"attention_evaluation":self.execution.attention_index(step)?,
                        "audio_frozen":self.condition_audio==self.na,"video_velocity":vg,"audio_velocity":ag});
                    if mode != "forward" {
                        self.update(step)?;
                        let video = self.write(&format!("{prefix}/video_latents.raw"), self.latent_video)?;
                        let audio = self.write(&format!("{prefix}/audio_latents.raw"), self.latent_audio)?;
                        let vg = if self.validate_reference {self.validation(&video, &self.manifest["steps"][step]["sampler_outputs"]["latents"])?}else{finite_f32(&video)?};
                        let ag = if self.validate_reference {self.validation(&audio, &self.manifest["steps"][step]["sampler_outputs"]["audio_latents"])?}else{finite_f32(&audio)?};
                        all_exact &= vg["passed"] == true && ag["passed"] == true;
                        result["video_latents"] = vg;
                        result["audio_latents"] = ag;
                    }
                    eprintln!("native step {step} comparisons: {result}");
                    self.log_step(step,step_started)?;
                    results.push(result);
                    // Record partial progress even if a later native operation fails.
                    report["results"] = json!(results);
                    fs::write(self.output.join("report.partial.json"), serde_json::to_vec_pretty(&report)?)?;
                    drop(profile_step);
                    profile::collect_step()?;
                }
                self.arena.sync()?;
                report["denoise_seconds"] = json!(denoise_started.elapsed().as_secs_f64());
                if let Some(time)=crate::startup_trace::timestamp(){report["denoise_end_monotonic_seconds"]=json!(time);}
                report["denoise_timing_scope"] = json!("synchronized host span: all DiT forwards and sampler updates; excludes loading and final latent download");
                report["results"] = json!(results);
                if mode != "forward" {
                    profile::set_step(self.steps.len() as i32);
                    let final_started=Instant::now();
                    // Only rank 0's final latents are read downstream (unpack, latent
                    // capture); peer ranks still download and check theirs, without a file.
                    let peer=mode=="generate" && !crate::debug_audit::enabled() && self.cp.as_ref().is_some_and(|cp|cp.group.rank!=0);
                    let final_video=if peer{self.arena.download(self.latent_video)?}else{self.write("final_video.raw", self.latent_video)?};
                    let final_audio=if peer{self.arena.download(self.latent_audio)?}else{self.write("final_audio.raw", self.latent_audio)?};
                    if mode=="generate" && !crate::debug_audit::enabled() {
                        for bytes in [&final_video,&final_audio] {
                            let _profile=HostSpan::new("output.final_finite",bytes.len());
                            if bytes.is_empty() || bytes.len()%4!=0 || bytes.chunks_exact(4).any(|x|!f32::from_le_bytes(x.try_into().unwrap()).is_finite()) {
                                return Err("nonfinite or malformed final DiT latents".into());
                            }
                        }
                    }
                    report["final_latents_seconds"]=json!(final_started.elapsed().as_secs_f64());
                }
                report["passed"] = json!(all_exact);
                report["steps_completed"] = json!(steps);
            },
            _ => return Err("mode must be refiner, block, forward, infer, generate or suite".into()),
        }
        self.arena.sync()?;
        self.audit("complete")?;
        let finishing=Instant::now();
        if let Some(sharded)=&self.resident_model.sharded {
            if transaction.as_ref().is_some_and(|p|!p.last()){sharded.drain(self.arena.stream)?;}
            // Free the request cache after the reply; the pipeline waits on await-release before decoding.
            else{sharded.defer_release(self.arena.stream)?;}
        }
        report["weight_finish_seconds"]=json!(finishing.elapsed().as_secs_f64());
        report["wall_seconds"] = json!(start.elapsed().as_secs_f64());
        if crate::debug_audit::enabled() || self.validate_reference {fs::write(self.output.join("report.json"), serde_json::to_vec_pretty(&report)?)?;}
        Ok(report)
    }
}
impl Runtime {
    /// After the reply: free the request weight cache whose release was deferred past it.
    pub fn release_deferred_weight_cache(&self)->Result<Option<f64>> {
        match &self.resident_model.sharded {Some(sharded)=>sharded.release_deferred(self.arena.stream),None=>Ok(None)}
    }
}
impl Drop for Runtime {
    fn drop(&mut self) {
        if let Some(sharded)=&self.resident_model.sharded {
            if sharded.transaction.borrow().is_idle(){let _=sharded.finish_request(self.arena.stream);}
        }
        let _ = self.arena.sync();
        // Destroy the cuDNN graph cache while its stream and CUDA allocations
        // still exist, then release the BLAS handle, then the owned arena.
        // Dense plans are shared with ResidentModel; cleanup only at last Rc drop.
        if !self.linear.is_null() { unsafe { h3_linear_destroy(self.linear); } }
    }
}

fn finite_request_input(dtype:&str,bytes:&[u8])->bool {
    match dtype {
        "torch.float32"=>bytes.chunks_exact(4).all(|x|f32::from_le_bytes(x.try_into().unwrap()).is_finite()),
        "torch.float64"=>bytes.chunks_exact(8).all(|x|f64::from_le_bytes(x.try_into().unwrap()).is_finite()),
        "torch.bfloat16"=>bytes.chunks_exact(2).all(|x|u16::from_le_bytes(x.try_into().unwrap())&0x7f80!=0x7f80),
        _=>true,
    }
}

// Included in native.rs after the unchanged Runtime math implementation.
#[derive(Clone)]
pub struct PreparedRequest {
    root: PathBuf,
    manifest: Value,
    layout: Value,
    data: HashMap<String,Vec<u8>>,
    long: Option<Box<PreparedLong>>,
}
const REQUEST_INPUTS: [&str;8] = ["hidden_states","audio_hidden_states","encoder_hidden_states",
    "position_ids","token_tags","video_indices","audio_indices","text_indices"];
const STRUCTURAL_INPUTS: [&str;5] = ["position_ids","token_tags","video_indices","audio_indices","text_indices"];

impl PreparedRequest {
    pub fn input_root(&self)->&Path{&self.root}
    pub fn is_windowed(&self)->bool{self.long.is_some()}
    pub fn structure_only(&self)->bool{self.manifest["structure_only"]==true}
    pub fn qwen_offloaded(&self)->bool{self.manifest["qwen_offloaded"]==true}
    pub fn decode_reserve_bytes(&self)->u64 {
        16u64<<30
    }
    pub fn nfe(&self)->Result<usize>{Ok(crate::recipe::Recipe::from_request(&self.manifest)?.nfe)}
    pub fn workflow(&self)->Result<&'static str>{Ok(crate::recipe::Recipe::from_request(&self.manifest)?.workflow)}
    pub fn transaction(&self)->Result<Option<crate::request_transaction::Phase>>{crate::request_transaction::Phase::from_request(&self.manifest)}
    /// CPU-only validation. No device state or resident weight is changed here.
    pub fn read(root:&Path)->Result<Self> {
        if !root.is_absolute(){return Err("request root must be absolute".into());}
        let manifest:Value=serde_json::from_slice(&crate::startup_trace::read(root.join("manifest.json"))?)?;
        if manifest["schema"]=="h3-native-long-request-v1"{return PreparedLong::read(root,manifest);}
        crate::execution::Execution::from_request(&manifest)?;
        crate::request_transaction::Phase::from_request(&manifest)?;
        if manifest["schema"]!="h3-native-request-v1" || manifest["status"]!="complete" ||
           manifest["producer"]!="h3-native-pipeline" || manifest["gold_activations_used"]!=false {
            return Err("requires complete native request preparation".into());
        }
        let inputs=&manifest["inputs"];
        let v=TensorSpec::parse(&inputs["hidden_states"])?;
        let a=TensorSpec::parse(&inputs["audio_hidden_states"])?;
        let t=TensorSpec::parse(&inputs["encoder_hidden_states"])?;
        for (spec,width,dtype) in [(&v,VIDEO,"torch.float32"),(&a,AUDIO,"torch.float32"),(&t,5120,"torch.bfloat16")] {
            if spec.shape.len()!=3 || spec.shape[0]!=1 || spec.shape[1]==0 || spec.shape[2]!=width || spec.dtype!=dtype {
                return Err("unsupported modality layout/dtype".into());
            }
        }
        let (nv,na,nt)=(v.shape[1],a.shape[1],t.shape[1]);
        let seq=nv.checked_add(na).and_then(|x|x.checked_add(nt)).ok_or("sequence overflow")?;
        crate::limits::packed_rows(&[nv,na,nt])?;
        if nt>crate::limits::MAX_TEXT_ROWS {return Err("native request exceeds conditioner model token limit".into());}
        let shape=&manifest["shape"];
        let integer=|name:&str|->Result<usize>{Ok(usize::try_from(shape[name].as_u64().ok_or("missing request geometry")?)?)};
        let (w,h,f)=(integer("width")?,integer("height")?,integer("num_frames")?);
        let recipe=crate::recipe::Recipe::from_request(&manifest)?;
        let min_frames=h3_native_pipeline::MIN_RENDER_FRAMES;
        if w<32 || h<32 || w%32!=0 || h%32!=0 || w.checked_mul(h).is_none() || !(min_frames..=345).contains(&f) || (f-5)%17!=0 || shape["fps"]!=24 {
            return Err("native video admission: invalid canvas, workflow frame range/grid or fps".into());
        }
        // Old request manifests describe the two-keyframe path. New requests
        // state this explicitly; no inference from token count or empty files.
        let keyframes=match shape.get("keyframe_count") {
            Some(value)=>usize::try_from(value.as_u64().ok_or("invalid keyframe_count")?)?,None=>2,
        };
        let ref_layout=if recipe.workflow=="ref2va" {
            let references=serde_json::from_value::<Vec<h3_native_pipeline::ref2va::ReferenceGeometry>>(manifest["references"].clone())?;
            Some(h3_native_pipeline::ref2va::layout(&vec![0;nt],h3_native_pipeline::Shape::new(w,h,f)?,&references)?)
        }else{None};
        let (expected_nv,condition_rows)=if let Some(layout)=&ref_layout{(layout.packed.video_indices.len(),layout.condition_video_rows)}else{crate::request_video_rows(w,h,f,keyframes)?};
        let expected_na=ref_layout.as_ref().map_or_else(||2*(f as f64/24.0*40.0).round_ties_even() as usize,|r|r.packed.audio_indices.len());
        let condition_audio=ref_layout.as_ref().map_or(0,|r|r.condition_audio_rows);
        if nv!=expected_nv || na!=expected_na || manifest["condition_video_rows"]!=json!(condition_rows) ||
           manifest["condition_audio_rows"]!=json!(condition_audio) || manifest["keyframe_noise_aug"].as_f64().map(|x|x as f32)!=Some(0.999f32) {
            return Err("request geometry, conditioning counts or base schedule mismatch".into());
        }
        let expected=[("position_ids",vec![seq,3],"torch.float64"),("token_tags",vec![seq],"torch.int64"),
            ("video_indices",vec![nv],"torch.int64"),("audio_indices",vec![na],"torch.int64"),("text_indices",vec![nt],"torch.int64")];
        for (name,shape,dtype) in expected {
            let spec=TensorSpec::parse(&inputs[name])?;
            if spec.shape!=shape || spec.dtype!=dtype {return Err(format!("invalid {name} layout").into());}
        }
        let mut data=HashMap::new();let mut layouts=serde_json::Map::new();
        if crate::debug_audit::enabled() {
            for name in REQUEST_INPUTS {
                let spec=TensorSpec::parse(&inputs[name])?;
                if spec.sha256.is_none(){return Err("debug input requires SHA256".into());}
                let bytes=spec.read(root)?;
                if !finite_request_input(&spec.dtype,&bytes){return Err(format!("nonfinite request input: {name}").into());}
                layouts.insert(name.into(),json!({"shape":spec.shape,"dtype":spec.dtype}));
                data.insert(name.to_owned(),bytes);
            }
        } else {
            // The same reads and checks, one input per thread (about 90 MB per request, read by every rank).
            let loaded=std::thread::scope(|scope|{
                let handles:Vec<_>=REQUEST_INPUTS.iter().map(|&name|{let inputs=&inputs;
                    scope.spawn(move||->std::result::Result<(Vec<u8>,Value),String>{
                        let spec=TensorSpec::parse(&inputs[name]).map_err(|e|e.to_string())?;
                        let bytes=fs::read(root.join(&spec.file)).map_err(|e|e.to_string())?;
                        if bytes.len()!=spec.bytes{return Err(format!("wrong request file size: {name}"));}
                        if !finite_request_input(&spec.dtype,&bytes){return Err(format!("nonfinite request input: {name}"));}
                        Ok((bytes,json!({"shape":spec.shape,"dtype":spec.dtype})))
                    })}).collect();
                handles.into_iter().map(|h|h.join().unwrap_or_else(|_|Err("request input thread panicked".into()))).collect::<Vec<_>>()
            });
            for (name,result) in REQUEST_INPUTS.iter().zip(loaded) {
                let (bytes,layout)=result?;layouts.insert((*name).into(),layout);data.insert((*name).to_owned(),bytes);
            }
        }
        validate_packed_indices(&i64_values(&data["token_tags"])? ,&i64_values(&data["video_indices"])? ,
            &i64_values(&data["audio_indices"])? ,&i64_values(&data["text_indices"])?)?;
        if let Some(mut expected)=ref_layout {
            let tags=i64_values(&data["token_tags"])?;
            if tags[..nt].iter().any(|v|*v!=0&&*v!=1){return Err("Ref2VA text tags must be 0/1".into());}
            expected.packed.token_tags[..nt].copy_from_slice(&tags[..nt]);
            for(name,bytes)in [("position_ids",h3_native_pipeline::f64_bytes(&expected.packed.position_ids)),
                ("token_tags",h3_native_pipeline::i64_bytes(&expected.packed.token_tags)),
                ("video_indices",h3_native_pipeline::i64_bytes(&expected.packed.video_indices)),
                ("audio_indices",h3_native_pipeline::i64_bytes(&expected.packed.audio_indices)),
                ("text_indices",h3_native_pipeline::i64_bytes(&expected.packed.text_indices))] {
                if data[name]!=bytes{return Err(format!("Ref2VA packed geometry differs: {name}").into());}
            }
        }
        let layout=json!({"inputs":layouts,"text_video_token_indices":manifest["text_video_token_indices"],"denoise_phase":manifest["denoise_phase"],"condition_video_rows":manifest["condition_video_rows"],
            "condition_audio_rows":manifest["condition_audio_rows"],"keyframe_noise_aug":manifest["keyframe_noise_aug"]});
        Ok(Self{root:root.to_owned(),manifest,layout,data,long:None})
    }
}

impl Runtime {
    pub fn new_shared(root:&Path,router:&Path,dense:&Path,out:&Path,input:&Path,
        model:Option<std::rc::Rc<ResidentModel>>)->Result<Self> {
        Self::build_shared(root,router,dense,out,Some(input),false,model)
    }
    pub fn shared_model(&self)->std::rc::Rc<ResidentModel>{self.resident_model.clone()}
    pub fn model_identity(&self)->Result<String> {self.resident_model.identity()}
    pub fn sparse_attention_report(&self)->Result<Value> {sparse_attention_report()}
    pub fn sharding_report(&self)->Value {
        self.resident_model.sharded.as_ref().map_or_else(||json!({"enabled":false}),|s|s.report())
    }
    pub fn prepared_fits(&self,request:&PreparedRequest)->Result<bool> {
        let initial=&self.native_request["preparation"];
        if crate::recipe::Recipe::from_request(initial)?!=crate::recipe::Recipe::from_request(&request.manifest)? {
            return Ok(false);
        }
        let mut layouts=serde_json::Map::new();
        for name in REQUEST_INPUTS {
            let spec=TensorSpec::parse(&initial["inputs"][name])?;
            layouts.insert(name.into(),json!({"shape":spec.shape,"dtype":spec.dtype}));
        }
        let layout=json!({"inputs":layouts,"text_video_token_indices":initial["text_video_token_indices"],"denoise_phase":initial["denoise_phase"],"condition_video_rows":initial["condition_video_rows"],
            "condition_audio_rows":initial["condition_audio_rows"],"keyframe_noise_aug":initial["keyframe_noise_aug"]});
        Ok(layout==request.layout && STRUCTURAL_INPUTS.iter().all(|name|
            self.resident_layout_inputs.get(*name)==request.data.get(*name)))
    }
    pub fn reset_prepared(&mut self,request:PreparedRequest,output:&Path)->Result<()> {
        if self.validate_reference || !self.prepared_fits(&request)?{return Err("request needs a different native layout".into());}
        if output.exists() && output!=self.output{return Err("output must be fresh".into());}
        fs::create_dir_all(output)?;
        // A retained layout has its own stream; rebind the shared communicator
        // before any request work. Requests are serialized and finish drained.
        if let Some(cp)=&self.cp {cp.group.bind(self.arena.stream)?;}
        for (name,buffer) in [("hidden_states",self.latent_video),("audio_hidden_states",self.latent_audio),("encoder_hidden_states",self.conditioning)] {
            self.arena.write(buffer,&request.data[name])?;
        }
        self.arena.sync()?;self.output=output.to_owned();
        self.native_request["input_root"]=json!(request.root);
        self.native_request["preparation"]=request.manifest;
        self.long_request=request.long;self.long_sparse_calls=None;
        self.audit("loaded")?;Ok(())
    }
    pub fn needs_layout_reclamation(&self)->Result<bool>{
        if self.resident_model.sharded.as_ref().is_none_or(|s|!s.workspace.mode.cache()){return Ok(false);}
        let cache=self.resident_model.sharded.as_ref().filter(|s|s.workspace.mode.cache() && s.cache.borrow().is_none())
            .map_or(0,|s|s.manifest.groups.iter().map(|g|g.bytes).sum::<usize>());
        let staging=if !self.validate_reference || self.scratch.contains_key("video_embeds"){0}else{(self.nv+self.na)*HIDDEN*2};
        let required=crate::scratch_plan::cache_execution_budget(cache,staging)?;
        if required==0{return Ok(false);}
        unsafe extern "C"{fn cudaMemGetInfo(free:*mut usize,total:*mut usize)->c_int;}
        let(mut free,mut total)=(0,0);cuda(unsafe{cudaMemGetInfo(&mut free,&mut total)})?;
        if free<required && self.resident_model.pool.cached_bytes.get()>0 {
            self.resident_model.pool.clear();
            cuda(unsafe{cudaMemGetInfo(&mut free,&mut total)})?;
        }
        let mut pressure=c_int::from(free<required);
        if pressure!=0 {
            eprintln!("request memory pressure: free={free} required={required} cache={cache} staging={staging} seq={} text={}",self.seq,self.nt);
        }
        if let Some(cp)=&self.cp {
            // Rank 0 also owns the encoders/decoders. All ranks must make the
            // same eviction decision before any starts rebuilding a layout.
            cp.group.bind(self.arena.stream)?;
            cp_check(unsafe{h3_comm_any(cp.group.handle,pressure,&mut pressure)})?;
        }
        Ok(pressure!=0)
    }
    pub fn resident_memory(&self)->Result<Value>{
        unsafe extern "C"{fn cudaMemGetInfo(free:*mut usize,total:*mut usize)->c_int;}
        self.arena.sync()?;let(mut free,mut total)=(0,0);cuda(unsafe{cudaMemGetInfo(&mut free,&mut total)})?;
        let (heads,workspace)=if let Some(cp)=&self.cp{(cp.heads,cp.bsa_workspace.bytes)}
            else{(self.router.as_ref().ok_or("memory report router missing")?.plan.heads,self.s("bsa_ws").bytes)};
        let mut served_workspace=0;
        cuda(unsafe{h3_bsa_workspace_bytes(self.seq as c_int,heads as c_int,self.kept as c_int,&mut served_workspace)})?;
        Ok(json!({"device_total_bytes":total,"device_free_bytes":free,"device_used_bytes":total-free,
            "sparse_workspace_bytes":workspace,"served_sparse_workspace_bytes":served_workspace,
            "sparse_workspace_bytes_saved":served_workspace.saturating_sub(workspace),
            "embedding_staging_reused":!self.validate_reference,
            "embedding_staging_bytes_saved":if self.validate_reference{0}else{(self.nv+self.na)*HIDDEN*2},
            "request_arena_bytes":self.arena.allocations.iter().map(|b|b.bytes).sum::<usize>(),
            "model_arena_bytes":self.resident_model.arena.allocations.iter().map(|b|b.bytes).sum::<usize>(),
            "weight_sharding":self.sharding_report(),
            "request_pool":self.resident_model.pool.report(),"dense_plan_cache":self.dense.cache_report()?,"router_plan_cache":self.resident_model.routers.report(),
            "request_allocation_count":self.arena.allocations.len(),"model_load_count":1,
            "projection_scratch_reused":self.s("fp_hidden").ptr==self.s("ff2").ptr,
            "projection_scratch_bytes_saved":if self.s("fp_hidden").ptr==self.s("ff2").ptr {
                self.s("fp_hidden").bytes+self.s("qn").bytes+self.s("kn").bytes
            }else{0}}))
    }
}

impl Runtime { pub fn sparse_calls_expected(&self)->Result<u64> {
    if let Some(calls)=self.long_sparse_calls{return Ok(calls);}
    let groups=self.router.as_ref().map_or(1,|r|HEADS/r.plan.heads);
    self.execution.sparse_calls_expected(self.seq,groups)
} }

impl Dense {
    fn cache_report(&self)->Result<Value>{
        type Report=unsafe extern "C" fn()->*const c_char;
        let pointer=unsafe{dlsym(self.library,CString::new("h3_dense_cache_report").unwrap().as_ptr())};
        if pointer.is_null(){return Err("generic dense cache telemetry ABI missing".into());}
        let report=unsafe{std::mem::transmute::<*mut c_void,Report>(pointer)()};
        if report.is_null(){return Err("null dense cache report".into());}
        Ok(serde_json::from_slice(unsafe{CStr::from_ptr(report)}.to_bytes())?)
    }
}

include!("dense_methods.rs");
include!("head_attention.rs");

include!("cp_runtime.rs");
include!("long_native.rs");

impl Runtime { pub fn nfe(&self)->usize {self.steps.len()} }
