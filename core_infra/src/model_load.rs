//! Streaming device load of checkpoint tensors read in place (checkpoint.rs) through the bounded
//! readers of weight_loader. `start` launches the readers before the caller creates its CUDA
//! context, so file reads overlap context creation; `finish` places every tensor in one
//! allocation (256-byte aligned views) in the dtype the model runs it in. A tensor stored in a
//! wider dtype is copied into a device staging ring and converted by one kernel per chunk on the
//! upload stream (infra_convert, kernels/convert.cu, which the model crate links).
//!
//! Policy (NATIVE_LOAD_*; defaults are the reader profile H3 measured best on B200):
//! WORKERS 24, CHUNK_MIB 32, PINNED_SLOTS 4, BUDGET_MIB (exactly what that needs), DIRECT 1
//! (readers' aligned buffers registered with CUDA, no pinned staging copy), VERIFY full|light
//! (full: SHA-256 of every byte before use; light: the trusted first/last 1 KiB guard, leaving the
//! full audit to the caller).
use crate::checkpoint::{Dtype, Manifest};
use crate::weight_loader::{device::Upload, tensor_allocation_layout, Config, ReaderPool};
use std::cell::Cell;
use std::collections::HashMap;
use std::ffi::{c_char, c_int, c_void, CStr};
use std::path::PathBuf;
use std::time::Instant;

pub type Result<T> = std::result::Result<T, String>;
type Stream = *mut c_void;

unsafe extern "C" {
    fn cudaMalloc(ptr: *mut *mut c_void, bytes: usize) -> c_int;
    fn cudaFree(ptr: *mut c_void) -> c_int;
    fn cudaStreamSynchronize(stream: Stream) -> c_int;
    fn cudaGetDevice(device: *mut c_int) -> c_int;
    fn cudaMemcpyPeerAsync(dst: *mut c_void, dst_device: c_int, src: *const c_void, src_device: c_int, bytes: usize,
                           stream: Stream) -> c_int;
    fn cudaGetErrorString(code: c_int) -> *const c_char;
    fn infra_convert(src: *const c_void, src_dtype: c_int, dst: *mut c_void, dst_dtype: c_int, n: i64, stream: Stream) -> c_int;
}

fn check(code: c_int, what: &str) -> Result<()> {
    if code == 0 { return Ok(()); }
    Err(format!("{what}: CUDA {code}: {}", unsafe { CStr::from_ptr(cudaGetErrorString(code)) }.to_string_lossy()))
}

fn convert_code(dtype: Dtype) -> Option<c_int> {
    match dtype { Dtype::F32 => Some(0), Dtype::F16 => Some(1), Dtype::BF16 => Some(2), _ => None }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Verify { Full, Light }

#[derive(Clone, Copy, Debug)]
pub struct Options { pub config: Config, pub direct: bool, pub balanced: bool, pub verify: Verify, pub staging_slots: usize,
                     pub ahead_budget: usize, pub retained_budget: usize }

impl Options {
    /// Prefer one admitted final-host read for resident independent groups.
    /// An explicit non-buffered file policy keeps the caller's original path.
    pub fn with_host_snapshot(mut self,budget:usize)->Result<Self> {
        if crate::weight_loader::snapshot_file_policy()? {self.retained_budget=budget;}
        Ok(self)
    }
    /// Serving starts from an admitted, SHA-pinned checkpoint declaration.
    /// Its trusted guards allow light foreground checks by default; the
    /// service owns the full post-request audit. Explicit full still wins.
    pub fn for_declaration(declaration:&crate::checkpoint::Declaration)->Result<Self>{
        Self::for_declaration_with_readers(declaration,24)
    }
    /// Declare a model's reader default before shared budget admission. An
    /// explicit deployment override still uses the same validated policy.
    pub fn for_declaration_with_readers(_declaration:&crate::checkpoint::Declaration,workers:usize)->Result<Self>{
        let mut options=Self::from_env_with_readers(workers)?;
        if std::env::var_os("NATIVE_LOAD_VERIFY").is_none(){options.verify=Verify::Light;}
        Ok(options)
    }
    pub fn from_env() -> Result<Self> {Self::from_env_with_readers(24)}
    fn from_env_with_readers(workers:usize) -> Result<Self> {
        fn number(name: &str, default: usize, low: usize, high: usize) -> Result<usize> {
            let n = match std::env::var(name) { Ok(s) => s.parse().map_err(|_| format!("invalid {name}"))?, Err(_) => default };
            if !(low..=high).contains(&n) { return Err(format!("{name} must be {low}..={high}")); }
            Ok(n)
        }
        let config = crate::startup_policy::readers_with_default_workers(workers)?;
        let ahead_budget = number("NATIVE_LOAD_AHEAD_GIB", 0, 0, 192)? << 30;
        let direct = match std::env::var("NATIVE_LOAD_DIRECT").as_deref() {
            Err(_) | Ok("1") => true, Ok("0") => false, _ => return Err("NATIVE_LOAD_DIRECT must be 0 or 1".into()),
        };
        let verify = match std::env::var("NATIVE_LOAD_VERIFY").as_deref() {
            Err(_) | Ok("full") => Verify::Full, Ok("light") => Verify::Light,
            _ => return Err("NATIVE_LOAD_VERIFY must be full or light".into()),
        };
        config.validate()?;
        Ok(Self { config, direct, balanced: true, verify, staging_slots: 4, ahead_budget, retained_budget: 0 })
    }
}

/// One tensor to load and the dtype the model uses it in.
#[derive(Clone, Debug)]
pub struct Request { pub name: String, pub dtype: Dtype }

struct Item { name: String, stored: Dtype, dtype: Dtype, shape: Vec<usize>, elements: usize }

/// Readers running; the device side has not started.
pub struct Pending { pool: ReaderPool, items: Vec<Item>, options: Options, started: Instant, read_bytes: usize,
                     stop_prefetch_after_upload: bool, host_snapshot:bool }

/// A loaded tensor: a view of the owning allocation.
#[derive(Clone, Debug)]
pub struct View { pub ptr: u64, pub bytes: usize, pub dtype: Dtype, pub shape: Vec<usize> }

#[derive(Clone, Copy, Debug, Default, serde::Serialize)]
pub struct Stats { pub tensors: usize, pub read_bytes: usize, pub device_bytes: usize, pub converted: usize, pub seconds: f64 }

pub struct HostWeights { chunks:crate::weight_loader::RetainedChunks, indices:HashMap<String,usize>, options:Options,
                         registrations:std::sync::Mutex<Option<std::sync::Arc<crate::weight_loader::device::Registrations>>> }
impl HostWeights {
    pub fn bytes(&self)->usize {self.chunks.bytes()}
    pub fn pinned_bytes(&self)->usize{self.registrations.lock().unwrap().as_ref().map(|r|r.bytes()).unwrap_or(0)}
    pub fn requires_pinning(&self)->bool {self.options.direct}
    pub fn names(&self)->impl Iterator<Item=&str> {self.indices.keys().map(String::as_str)}
    pub fn pin(self)->Result<std::sync::Arc<Self>> {
        self.pin_for_use()?;
        Ok(std::sync::Arc::new(self))
    }
    /// Publish registration on the device owner at the declared use boundary,
    /// after GPU work has drained. The live context must outlive these leases.
    pub fn pin_for_use(&self)->Result<()> {
        if !self.options.direct {return Ok(());}
        let mut registrations=self.registrations.lock().unwrap();
        if let Some(value)=registrations.as_ref() {
            if value.context_identity()!=crate::device_context::current_identity().map_err(|e|e.to_string())? {
                return Err("retained host registrations belong to another CUDA context".into());
            }
        }else {*registrations=Some(crate::weight_loader::device::Registrations::pin(&self.chunks)?);}
        Ok(())
    }
    /// # Safety
    /// Every destination must hold the original tensor bytes on this context's
    /// live stream. No file access or verification is repeated on RAM reload.
    pub unsafe fn upload(&self,requests:&[(&str,u64)],stream:Stream)->Result<()> {
        let _span=crate::startup_trace::Span::new("retained_host_upload","RAM reload",0);
        let mut upload=Upload::new_options(stream,self.options.config,self.options.direct,None)?;
        if let Some(registrations)=self.registrations.lock().unwrap().clone() {upload.use_registrations(registrations)?;}
        for &(name,dst) in requests {
            let job=*self.indices.get(name).ok_or_else(||format!("retained tensor missing: {name}"))?;
            unsafe{upload.copy_retained(dst as *mut c_void,&self.chunks,job)}?;
        }
        upload.finish()
    }
    /// Restore already-verified runtime-dtype tensors without reopening files.
    /// Conversion remains the cold loader's job; this path rejects a different
    /// stored dtype before allocating rather than silently copying wrong bytes.
    /// # Safety
    /// The current context and stream outlive the returned allocation/views.
    pub unsafe fn restore(self:&std::sync::Arc<Self>,manifest:&Manifest,requests:&[Request],stream:Stream)->Result<Loaded>{
        if requests.is_empty(){return Err("empty retained restore".into());}
        let mut names=std::collections::HashSet::new();let mut specs=Vec::new();
        for request in requests{
            let spec=manifest.tensor(&request.name)?;
            if !names.insert(request.name.as_str()) || spec.dtype!=request.dtype || !self.indices.contains_key(&request.name){
                return Err("retained restore identity/dtype differs".into());
            }
            if self.chunks.tensor_bytes(self.indices[&request.name])?!=spec.bytes{return Err("retained restore extent differs".into());}
            specs.push(spec);
        }
        let sizes=specs.iter().map(|s|s.bytes).collect::<Vec<_>>();
        let(offsets,total)=tensor_allocation_layout(&sizes)?;let started=Instant::now();
        let mut loaded=Loaded{slab:std::ptr::null_mut(),tensors:HashMap::new(),stats:Stats::default(),host:Some(self.clone())};
        check(unsafe{cudaMalloc(&mut loaded.slab,total)},"retained model allocation")?;
        for ((request,spec),offset) in requests.iter().zip(specs).zip(offsets){
            loaded.tensors.insert(request.name.clone(),View{ptr:loaded.slab as u64+offset as u64,
                bytes:spec.bytes,dtype:request.dtype,shape:spec.shape.clone()});
        }
        let destinations=requests.iter().map(|r|(r.name.as_str(),loaded.tensors[&r.name].ptr)).collect::<Vec<_>>();
        unsafe{self.upload(&destinations,stream)}?;
        loaded.stats=Stats{tensors:requests.len(),read_bytes:0,device_bytes:total,converted:0,seconds:started.elapsed().as_secs_f64()};
        Ok(loaded)
    }
}

pub struct Loaded { slab: *mut c_void, pub tensors: HashMap<String, View>, pub stats: Stats,
                    pub host:Option<std::sync::Arc<HostWeights>> }
// The allocation is device memory of the process's context: usable (and freed) from any host thread.
unsafe impl Send for Loaded {}

impl Loaded {
    pub fn get(&self, name: &str) -> Result<&View> { self.tensors.get(name).ok_or_else(|| format!("tensor {name} was not loaded")) }
    /// Allocate a component from its already verified, retained source bytes.
    /// No checkpoint is reopened. Stored and runtime dtypes must match.
    ///
    /// # Safety
    /// `stream` must belong to the current, live CUDA context. That context
    /// must outlive the returned allocation and the retained registrations.
    pub unsafe fn from_host(manifest:&Manifest, requests:&[Request], host:std::sync::Arc<HostWeights>, stream:Stream)->Result<Self> {
        let started=Instant::now();
        let items=host_items(manifest,requests,&host)?;
        let sizes=items.iter().map(|item|item.elements*item.dtype.size()).collect::<Vec<_>>();
        let (offsets,total)=tensor_allocation_layout(&sizes)?;
        let mut loaded=Loaded {slab:std::ptr::null_mut(),tensors:HashMap::new(),stats:Stats::default(),host:None};
        if total>0 {check(unsafe{cudaMalloc(&mut loaded.slab,total)},"retained model allocation")?;}
        let destinations=items.iter().zip(&offsets).map(|(item,offset)|
            (item.name.as_str(),loaded.slab as u64+*offset as u64)).collect::<Vec<_>>();
        unsafe{host.upload(&destinations,stream)}?;
        for ((item,offset),bytes) in items.iter().zip(offsets).zip(sizes) {
            loaded.tensors.insert(item.name.clone(),View {ptr:loaded.slab as u64+offset as u64,
                bytes,dtype:item.dtype,shape:item.shape.clone()});
        }
        loaded.stats=Stats {tensors:items.len(),read_bytes:0,device_bytes:total,converted:0,seconds:started.elapsed().as_secs_f64()};
        loaded.host=Some(host);Ok(loaded)
    }
    /// An exact copy of this allocation on the calling thread's current device, made by one peer
    /// copy from `source`, the device that holds it, so a second GPU gets a component without
    /// reading it again. The views keep their offsets; the copy is complete on return.
    ///
    /// # Safety
    /// `stream` belongs to the current device, which has peer access to `source`. This
    /// allocation stays alive and unmodified until the call returns.
    pub unsafe fn replicate(&self, source: c_int, stream: Stream) -> Result<Self> {
        let started=Instant::now();
        let base=self.slab as u64;
        let mut total=0usize;
        for view in self.tensors.values() {
            let end=view.ptr.checked_sub(base).and_then(|o|o.checked_add(view.bytes as u64)).filter(|_|base!=0)
                .ok_or("replicated views lie outside their allocation")?;
            total=total.max(end as usize);
        }
        let mut copy=Loaded {slab:std::ptr::null_mut(),tensors:HashMap::new(),stats:Stats::default(),host:None};
        if total>0 {
            let _span=crate::startup_trace::Span::new("model_replicate",format!("device {source}"),total);
            let mut device=0;
            check(unsafe{cudaGetDevice(&mut device)},"replica device")?;
            check(unsafe{cudaMalloc(&mut copy.slab,total)},"replica allocation")?;
            check(unsafe{cudaMemcpyPeerAsync(copy.slab,device,self.slab,source,total,stream)},"replica peer copy")?;
            check(unsafe{cudaStreamSynchronize(stream)},"replica peer copy")?;
        }
        copy.tensors=self.tensors.iter().map(|(name,view)|(name.clone(),
            View {ptr:copy.slab as u64+(view.ptr-base),..view.clone()})).collect();
        copy.stats=Stats {tensors:copy.tensors.len(),read_bytes:0,device_bytes:total,converted:0,seconds:started.elapsed().as_secs_f64()};
        Ok(copy)
    }
    /// # Safety
    /// The recipient takes sole ownership of cudaFree and must keep the
    /// allocation's context alive until all tensor views have been released.
    pub unsafe fn into_allocation(mut self)->(*mut c_void,HashMap<String,View>,Option<std::sync::Arc<HostWeights>>,Stats) {
        let slab=std::mem::replace(&mut self.slab,std::ptr::null_mut());
        (slab,std::mem::take(&mut self.tensors),self.host.take(),self.stats)
    }
}

fn host_items(manifest:&Manifest,requests:&[Request],host:&HostWeights)->Result<Vec<Item>> {
    let mut seen=std::collections::HashSet::new();
    requests.iter().map(|request| {
        let tensor=manifest.tensor(&request.name)?;
        if tensor.dtype!=request.dtype {return Err("retained component reload requires matching stored/runtime dtypes".into());}
        if !seen.insert(&request.name) || !host.indices.contains_key(&request.name) {
            return Err("retained component tensors differ from the admitted manifest".into());
        }
        if host.chunks.tensor_bytes(host.indices[&request.name])?!=tensor.bytes {
            return Err("retained component extent differs from the admitted manifest".into());
        }
        Ok(Item {name:request.name.clone(),stored:tensor.dtype,dtype:request.dtype,shape:tensor.shape.clone(),elements:tensor.elements()})
    }).collect()
}
impl Drop for Loaded {
    fn drop(&mut self) { if !self.slab.is_null() { unsafe { cudaFree(self.slab); } } }
}

/// Starts the readers for `requests` (no CUDA call). `roots` maps the manifest's root names to directories.
pub fn start(manifest: &Manifest, roots: &HashMap<String, PathBuf>, requests: &[Request], options: Options) -> Result<Pending> {
    crate::startup_trace::configure_from_env();
    let _span = crate::startup_trace::Span::new("model_load", "start_readers", 0);
    let (jobs,items,read_bytes)=describe(manifest,roots,requests,options)?;
    let started = Instant::now();
    let (verify_sha, guarded) = match options.verify { Verify::Full => (true, false), Verify::Light => (false, true) };
    let mut pool = if options.retained_budget != 0 || options.ahead_budget != 0 {
        let budget=if options.retained_budget!=0 {options.retained_budget}else{options.ahead_budget};
        let config = Config { budget_bytes: budget, ..options.config };
        ReaderPool::start_read_ahead_options(jobs, config, options.direct || options.retained_budget!=0, verify_sha, guarded)?
    } else {
        ReaderPool::start_checkpoint_mode(jobs, options.config, options.direct, verify_sha, guarded, options.balanced)?
    };
    if options.retained_budget!=0 {pool.retain_outputs()?;}
    Ok(Pending { pool, items, options, started, read_bytes, stop_prefetch_after_upload: true,host_snapshot:false })
}

fn describe(manifest:&Manifest,roots:&HashMap<String,PathBuf>,requests:&[Request],options:Options)
    ->Result<(Vec<crate::weight_loader::Job>,Vec<Item>,usize)> {
    let mut jobs = Vec::with_capacity(requests.len());
    let mut items = Vec::with_capacity(requests.len());
    let mut read_bytes = 0usize;
    for r in requests {
        let t = manifest.tensor(&r.name)?;
        if t.dtype != r.dtype && (convert_code(t.dtype).is_none() || convert_code(r.dtype).is_none()
            || (t.dtype != Dtype::F32 && r.dtype != Dtype::F32)) {
            return Err(format!("{}: no conversion from {} to {}", r.name, t.dtype.name(), r.dtype.name()));
        }
        if t.dtype != r.dtype && options.config.chunk_bytes % t.dtype.size() != 0 {
            return Err("load chunk must hold whole elements".into());
        }
        jobs.push(manifest.job(&r.name, roots)?);
        items.push(Item { name: r.name.clone(), stored: t.dtype, dtype: r.dtype, shape: t.shape.clone(), elements: t.elements() });
        read_bytes += t.bytes;
    }
    Ok((jobs,items,read_bytes))
}

impl Pending {
    /// Completes an admitted retained load without a GPU allocation. Direct
    /// mode pins its immutable host pages in the caller's live CUDA context.
    pub fn finish_host(self)->Result<std::sync::Arc<HostWeights>> {
        self.finish_host_unpinned()?.pin()
    }
    /// CPU-only completion for an idle/background RAM cache. Pinning happens
    /// on the owner after GPU work, not concurrently with kernel submission.
    pub fn finish_host_unpinned(mut self)->Result<HostWeights> {
        if self.options.retained_budget==0{return Err("host-only completion requires retained admission".into());}
        self.pool.consume_all(|_,_,_|Ok(()))?;
        let chunks=self.pool.take_retained()?.ok_or("host retention missing")?;
        let indices=self.items.iter().enumerate().map(|(i,item)|(item.name.clone(),i)).collect();
        Ok(HostWeights{chunks,indices,options:self.options,registrations:std::sync::Mutex::new(None)})
    }
    /// Allocates and fills the tensors on `stream`'s device; returns after the stream drains.
    ///
    /// # Safety
    /// `stream` must be a live CUDA stream of the current device.
    pub unsafe fn finish(self, stream: Stream) -> Result<Loaded> {
        unsafe{self.finish_registered(stream,None)}
    }
    unsafe fn finish_registered(mut self,stream:Stream,shared:Option<std::sync::Arc<crate::weight_loader::device::SharedRegistrations>>)->Result<Loaded>{
        let label = self.items.iter().filter_map(|i| i.name.split('.').next()).collect::<std::collections::BTreeSet<_>>()
            .into_iter().collect::<Vec<_>>().join("+");
        let _span = crate::startup_trace::Span::new("model_load", &label, self.read_bytes);
        let sizes: Vec<usize> = self.items.iter().map(|i| i.elements * i.dtype.size()).collect();
        let (offsets, total) = tensor_allocation_layout(&sizes)?;
        let mut loaded = Loaded { slab: std::ptr::null_mut(), tensors: HashMap::new(), stats: Stats::default(), host:None };
        if total > 0 { crate::startup_trace::measure("model_allocate", &label, total,
            || check(unsafe { cudaMalloc(&mut loaded.slab, total) }, "model allocation"))?; }
        let converted = self.items.iter().filter(|i| i.stored != i.dtype).count();
        let chunk = self.options.config.chunk_bytes;
        let slots = self.options.staging_slots.max(1);
        let mut staging: *mut c_void = std::ptr::null_mut();
        if converted > 0 { check(unsafe { cudaMalloc(&mut staging, slots * chunk) }, "conversion staging")?; }
        let mut registrations=None;
        let result = (|| -> Result<()> {
            // Ordered staged loads may still have CPU readers to start. Keep
            // the optional OS-cache helper alive until the last real deferred
            // payload has submitted its first successful upload. Independent
            // and already-started reader cohorts retain the original policy.
            let stop=if self.stop_prefetch_after_upload {
                std::env::var_os("NATIVE_LOAD_PREFETCH_STOP_FILE").map(PathBuf::from)
            } else { None };
            let mut upload = Upload::new_options(stream, self.options.config, self.options.direct && !self.host_snapshot, stop)?;
            if let Some(shared)=shared {upload.use_shared_registrations(shared)?;}
            let slab = loaded.slab as usize;
            let next = Cell::new(0usize);
            let items = &self.items;
            let staging_at = |k: usize| staging as usize + (k % slots) * chunk;
            unsafe {
                let _upload_span = crate::startup_trace::Span::new("model_upload", &label, self.read_bytes);
                self.pool.consume_upload_then(&mut upload,
                    |job, offset| {
                        let item = &items[job];
                        (if item.stored != item.dtype { staging_at(next.get()) } else { slab + offsets[job] + offset }) as *mut c_void
                    },
                    |job, offset, bytes| {
                        let item = &items[job];
                        if item.stored == item.dtype { return Ok(()); }
                        let (from, to) = (item.stored.size(), item.dtype.size());
                        let dst = slab + offsets[job] + offset / from * to;
                        let src = staging_at(next.get());
                        next.set(next.get() + 1);
                        check(infra_convert(src as *const c_void, convert_code(item.stored).unwrap(), dst as *mut c_void,
                                            convert_code(item.dtype).unwrap(), (bytes / from) as i64, stream), "weight conversion")
                    })?;
            }
            check(unsafe { cudaStreamSynchronize(stream) }, "model load drain")?;
            // Reusable reader registrations cover the DMA ring, not the
            // separate immutable copies. Pin those copies on their owner at
            // the model's later residency boundary, never credit ring pages.
            if self.options.retained_budget!=0 && self.options.direct && !self.pool.retains_copies() && !self.host_snapshot {
                registrations=Some(upload.retain_registrations()?);
            }
            Ok(())
        })();
        if !staging.is_null() { unsafe { cudaFree(staging); } }
        result?;
        if let Some(chunks)=self.pool.take_retained()? {
            let indices=self.items.iter().enumerate().map(|(i,item)|(item.name.clone(),i)).collect();
            loaded.host=Some(std::sync::Arc::new(HostWeights{chunks,indices,options:self.options,registrations:std::sync::Mutex::new(registrations)}));
        }
        for ((item, &offset), &bytes) in self.items.iter().zip(&offsets).zip(&sizes) {
            loaded.tensors.insert(item.name.clone(), View { ptr: loaded.slab as u64 + offset as u64, bytes, dtype: item.dtype,
                                                            shape: item.shape.clone() });
        }
        loaded.stats = Stats { tensors: self.items.len(), read_bytes: self.read_bytes, device_bytes: total, converted,
                               seconds: self.started.elapsed().as_secs_f64() };
        Ok(loaded)
    }
}

/// A load in ordered stages: stage 0's readers start at once (before the caller's CUDA context),
/// `finish_first` fills stage 0 on the caller's stream, and `spawn_rest` streams every later stage
/// one after another on a background thread with its own CUDA stream, so the caller can compute
/// with stage 0 (e.g. run the encoders) while the rest (e.g. the DiT) arrives.
/// By default reads are sequential by stage. With NATIVE_LOAD_AHEAD_GIB, the complete
/// retained-read schedule is admitted against one RAM bound and all CPU readers start
/// before CUDA. GPU uploads keep their stage order; foreground preparation keeps CPU priority.
enum Stage { Deferred(Vec<Request>), Reading(Pending) }
impl Stage {
    fn deferred_payload(&self)->bool {
        matches!(self,Self::Deferred(requests) if !requests.is_empty())
    }
    fn start_for_upload(self,manifest:&Manifest,roots:&HashMap<String,PathBuf>,options:Options,
                        following:&[Stage])->Result<Pending> {
        let mut pending=match self {Self::Reading(pending)=>pending,
            Self::Deferred(requests)=>start(manifest,roots,&requests,options)?};
        pending.handoff_prefetch_stop(following);
        Ok(pending)
    }
}
impl Pending {
    fn handoff_prefetch_stop(&mut self,following:&[Stage]) {
        if following.iter().any(Stage::deferred_payload){self.stop_prefetch_after_upload=false;}
    }
}
pub struct Staged { first: Option<Pending>, rest: Vec<Stage>, manifest: Manifest, roots: HashMap<String, PathBuf>,
                    options: Options, started: Instant }

unsafe extern "C" {
    fn cudaStreamCreateWithFlags(stream: *mut Stream, flags: u32) -> c_int;
    fn cudaStreamDestroy(stream: Stream) -> c_int;
}

/// The later stages' loads, in stage order, with each stage's completion time (seconds since `start_staged`).
pub struct RestHandle { worker: Option<std::thread::JoinHandle<Result<()>>>,
                        stages:std::sync::mpsc::Receiver<(Loaded,f64)>,
                        components:Option<ComponentRest>,
                        _registrations:Option<std::sync::Arc<crate::weight_loader::device::SharedRegistrations>> }
struct ComponentRest {ready:std::collections::VecDeque<crate::startup_result::ResultReceiver<(Loaded,f64)>>,
    workers:Vec<std::thread::JoinHandle<()>>}
impl RestHandle {
    pub fn next_stage(&mut self)->Result<(Loaded,f64)> {
        if let Some(rest)=&mut self.components {
            return rest.ready.pop_front().ok_or("no further component")?.wait();
        }
        match self.stages.recv() {
            Ok(stage)=>Ok(stage),
            Err(_)=>{
                let worker=self.worker.take().ok_or("stage worker already joined")?;
                worker.join().map_err(|_|"stage thread panicked")??;
                Err("no further load stage".into())
            }
        }
    }
    pub fn join(mut self) -> std::thread::Result<Result<Vec<(Loaded, f64)>>> {
        if let Some(mut rest)=self.components.take(){
            let mut values=Vec::new();let mut error=None;
            for receive in rest.ready.drain(..){
                match receive.wait(){Ok(stage)=>values.push(stage),Err(e)=>{if error.is_none(){error=Some(e);}}}
            }
            let mut panic=None;
            for worker in rest.workers {if let Err(e)=worker.join(){if panic.is_none(){panic=Some(e);}}}
            if let Some(panic)=panic{return Err(panic);}
            return Ok(match error{Some(e)=>Err(e),None=>Ok(values)});
        }
        let result=self.stages.iter().collect();
        match self.worker.take() {Some(worker)=>worker.join().map(|r|r.map(|_|result)),None=>Ok(Ok(result))}
    }
}
impl Drop for RestHandle {
    fn drop(&mut self) {
        // Preparation/conditioning errors cannot leave uploads detached from
        // the caller's CUDA context lifetime.
        if let Some(worker) = self.worker.take() { let _ = worker.join(); }
        if let Some(rest)=self.components.take(){for worker in rest.workers{let _=worker.join();}}
    }
}

pub fn start_staged(manifest: Manifest, roots: HashMap<String, PathBuf>, stages: Vec<Vec<Request>>, mut options: Options)
                    -> Result<Staged> {
    crate::startup_trace::configure_from_env();
    let started = Instant::now();
    if options.retained_budget!=0 {
        let sizes=stages.iter().flatten().map(|r|manifest.tensor(&r.name).map(|t|t.bytes)).collect::<Result<Vec<_>>>()?;
        let required=crate::weight_loader::read_ahead_bound(&sizes,options.config,true,stages.len())?;
        if required>options.retained_budget {return Err(format!("resident host weights need {required} bytes, budget is {}",options.retained_budget));}
        let descriptions=stages.iter().map(|r|describe(&manifest,&roots,r,options)).collect::<Result<Vec<_>>>()?;
        let (jobs,layouts):(Vec<_>,Vec<_>)=descriptions.into_iter().map(|(jobs,items,bytes)|(jobs,(items,bytes))).unzip();
        let (verify,guarded)=match options.verify {Verify::Full=>(true,false),Verify::Light=>(false,true)};
        let pools=ReaderPool::start_retained_stages(jobs,Config{budget_bytes:options.retained_budget,..options.config},verify,guarded)?;
        let mut pending=pools.into_iter().zip(layouts).map(|(pool,(items,read_bytes))|
            Pending{pool,items,options,started,read_bytes,stop_prefetch_after_upload:true,host_snapshot:false});
        let first=pending.next().ok_or("a staged load needs at least one stage")?;
        return Ok(Staged{first:Some(first),rest:pending.map(Stage::Reading).collect(),manifest,roots,options,started});
    }
    if options.ahead_budget != 0 {
        // Admit the complete CPU read-ahead, including aligned tails and each
        // stage's pinned-copy slots, against ONE budget before starting readers.
        let sizes = stages.iter().flatten().map(|r| manifest.tensor(&r.name).map(|t| t.bytes)).collect::<Result<Vec<_>>>()?;
        let required = crate::weight_loader::read_ahead_bound(&sizes, options.config, options.direct, stages.len())?;
        if required > options.ahead_budget {
            eprintln!("native staged read-ahead: bounded fallback; required={required} budget={}", options.ahead_budget);
            options.ahead_budget = 0;
        } else {
            eprintln!("native staged read-ahead: admitted; bound_bytes={required} budget={}", options.ahead_budget);
        }
    }
    let mut stages = stages.into_iter();
    let first = stages.next().ok_or("a staged load needs at least one stage")?;
    let mut pending = start(&manifest, &roots, &first, options)?;
    let rest = stages.map(|requests| {
        if options.ahead_budget != 0 { start(&manifest, &roots, &requests, options).map(Stage::Reading) }
        else { Ok(Stage::Deferred(requests)) }
    }).collect::<Result<Vec<_>>>()?;
    pending.handoff_prefetch_stop(&rest);
    Ok(Staged { first: Some(pending), rest, manifest, roots, options, started })
}

impl Staged {
    /// The resident host-warm API uses the same schedule and guards without H2D.
    pub fn finish_host(self)->Result<Vec<std::sync::Arc<HostWeights>>> {
        self.finish_host_unpinned()?.into_iter().map(HostWeights::pin).collect()
    }
    pub fn finish_host_unpinned(mut self)->Result<Vec<HostWeights>> {
        let mut hosts=vec![self.first.take().ok_or("stage 0 was already finished")?.finish_host_unpinned()?];
        for stage in self.rest {
            let pending=match stage {Stage::Reading(p)=>p,Stage::Deferred(requests)=>start(&self.manifest,&self.roots,&requests,self.options)?};
            hosts.push(pending.finish_host_unpinned()?);
        }
        Ok(hosts)
    }
    /// Fills stage 0 on `stream`.
    ///
    /// # Safety
    /// `stream` must be a live CUDA stream of the current device.
    pub unsafe fn finish_first(&mut self, stream: Stream) -> Result<Loaded> {
        let pending = self.first.take().ok_or("stage 0 was already finished")?;
        unsafe { pending.finish(stream) }
    }

    /// Streams the later stages on a background thread (current device, a private non-blocking
    /// stream). Join the handle before using their tensors: each stage has drained when it returns.
    pub fn spawn_rest(self) -> Result<RestHandle> {
        if self.first.is_some() { return Err("finish stage 0 before the later stages".into()); }
        let context=crate::device_context::current_identity().map_err(|e|e.to_string())?;
        let Staged { rest, manifest, roots, options, started, .. } = self;
        let (send,stages)=std::sync::mpsc::channel();
        let worker = std::thread::Builder::new().name("model-load-stages".into()).spawn(move || -> Result<()> {
            unsafe{crate::device_context::bind_current(context)}.map_err(|e|e.to_string())?;
            let mut stream: Stream = std::ptr::null_mut();
            check(unsafe { cudaStreamCreateWithFlags(&mut stream, 1) }, "stage stream")?;
            let result = (||->Result<()> {
                let mut rest=rest.into_iter();
                while let Some(stage)=rest.next() {
                    let pending=stage.start_for_upload(&manifest,&roots,options,rest.as_slice())?;
                    let loaded = unsafe { pending.finish(stream) }?;
                    send.send((loaded,started.elapsed().as_secs_f64())).map_err(|_|"stage owner cancelled")?;
                }
                Ok(())
            })();
            unsafe { cudaStreamDestroy(stream); }
            result
        }).map_err(|e| format!("stage thread: {e}"))?;
        Ok(RestHandle { worker: Some(worker),stages,components:None,_registrations:None })
    }
}

/// Independent component preload: bounded CPU buffers/readers are shared;
/// CUDA uploads proceed on component streams as soon as the context is ready.
/// Only the component being used is joined. Model dependency order is not an
/// unnecessary serialization barrier for reading/uploading immutable weights.
pub struct Components {
    pending:Option<Vec<Pending>>,ready:Vec<crate::startup_result::ResultReceiver<(Loaded,f64)>>,
    workers:Vec<std::thread::JoinHandle<()>>,registrations:Option<std::sync::Arc<crate::weight_loader::device::SharedRegistrations>>,
    started:Instant,first_finished:bool,context:usize,reader_budget_bytes:usize,shared_readers:bool,
    deferred:Vec<(usize,DeferredComponent)>,
}
enum DeferredComponent {
    Reader {jobs:Vec<crate::weight_loader::Job>,items:Vec<Item>,options:Options,
        policy:crate::weight_loader::FileReadPolicy,schedule:crate::weight_loader::ReadSchedule,read_bytes:usize},
    Host {manifest:std::sync::Arc<Manifest>,requests:Vec<Request>,host:std::sync::Arc<HostWeights>},
}
impl DeferredComponent {
    fn readers(self,started:Instant)->Result<Pending> {
        let Self::Reader{jobs,items,options,policy,schedule,read_bytes}=self else {
            return Err("retained upload has no cold reader takeover".into());
        };
        let _trace=crate::startup_trace::Span::new("startup_reader_prepare","deferred component pool",read_bytes);
        let (verify,guarded)=match options.verify{Verify::Full=>(true,false),Verify::Light=>(false,true)};
        let pool=ReaderPool::start_components_scheduled(vec![jobs],options.config,options.direct,verify,guarded,policy,schedule)?
            .pop().ok_or("deferred component reader pool missing")?;
        Ok(Pending{pool,items,options,started,read_bytes,stop_prefetch_after_upload:true,host_snapshot:false})
    }
    unsafe fn upload(self,stream:Stream,started:Instant)->Result<Loaded> {
        match self {
            Self::Host{manifest,requests,host}=>unsafe{Loaded::from_host(&manifest,&requests,host,stream)},
            reader=>unsafe{reader.readers(started)?.finish(stream)},
        }
    }
}
/// The final immutable CPU copies and the existing shared reader buffers have
/// one complete admission. This changes no reader count, chunk or DMA quota.
pub(crate) fn component_retention_bounds(manifest:&Manifest,components:&[Vec<Request>],options:Options)->Result<Vec<usize>> {
    if options.retained_budget==0 {return Ok(vec![0;components.len()]);}
    let mut total=options.config.budget_bytes;let mut bounds=Vec::with_capacity(components.len());
    for requests in components {
        let sizes=requests.iter().map(|request| {
            let tensor=manifest.tensor(&request.name)?;
            if tensor.dtype!=request.dtype {return Err("independent retained components require matching stored/runtime dtypes".into());}
            Ok(tensor.bytes)
        }).collect::<Result<Vec<_>>>()?;
        let bound=crate::weight_loader::retained_copy_bound(&sizes,options.config.chunk_bytes)?;
        total=total.checked_add(bound).ok_or("retained component budget overflow")?;bounds.push(bound);
    }
    if total>options.retained_budget {return Err(format!("retained components need {total} bytes including readers, budget is {}",options.retained_budget));}
    Ok(bounds)
}
fn component_descriptions(manifest:&Manifest,roots:&HashMap<String,PathBuf>,components:&[Vec<Request>],options:Options)
    ->Result<Vec<(Vec<crate::weight_loader::Job>,Vec<Item>,usize)>>{
    std::thread::scope(|scope| {
        let workers=components.iter().map(|requests|scope.spawn(move || {
            let name=requests.first().and_then(|r|r.name.split('.').next()).unwrap_or("component");
            let _span=crate::startup_trace::Span::new("startup_reader_prepare",format!("describe {name}"),0);
            describe(manifest,roots,requests,options)
        })).collect::<Vec<_>>();
        let results=workers.into_iter().map(|w|w.join().map_err(|_|"component metadata plan panicked".to_string()).and_then(|v|v)).collect::<Vec<_>>();
        results.into_iter().collect::<Result<Vec<_>>>()
    })
}

pub fn start_components(manifest:&Manifest,roots:&HashMap<String,PathBuf>,components:Vec<Vec<Request>>,options:Options)->Result<Components>{
    start_components_with_policy(manifest,roots,components,options,crate::weight_loader::FileReadPolicy::Auto)
}
/// Same tensor admission, integrity, CPU budget and CUDA upload policy.
pub fn start_components_with_policy(manifest:&Manifest,roots:&HashMap<String,PathBuf>,components:Vec<Vec<Request>>,options:Options,policy:crate::weight_loader::FileReadPolicy)->Result<Components>{
    start_components_scheduled(manifest,roots,components,options,policy,crate::weight_loader::ReadSchedule::ByteFair)
}
pub fn start_components_scheduled(manifest:&Manifest,roots:&HashMap<String,PathBuf>,components:Vec<Vec<Request>>,options:Options,policy:crate::weight_loader::FileReadPolicy,schedule:crate::weight_loader::ReadSchedule)->Result<Components>{
    crate::startup_trace::configure_from_env();
    if components.is_empty() || options.ahead_budget!=0 {
        return Err("independent component loads need groups without a separate read-ahead budget".into());
    }
    let snapshot=options.retained_budget!=0;
    let slots=if options.direct && !snapshot {options.config.pinned_slots}else{
        components.len().checked_mul(options.config.pinned_slots).ok_or("component ring budget overflow")?};
    let buffers=if snapshot {slots}else{2*options.config.workers+slots};
    let bound=buffers.checked_mul(options.config.chunk_bytes).ok_or("component buffer bound overflow")?;
    if bound>options.config.budget_bytes{return Err("component uploads exceed the declared buffer budget".into());}
    let retained_bounds=component_retention_bounds(manifest,&components,options)?;
    // Component metadata plans have no edges between them. They share the
    // same admitted worker/buffer pool once all plans pass validation.
    let descriptions=component_descriptions(manifest,roots,&components,options)?;
    let (jobs,layouts):(Vec<_>,Vec<_>)=descriptions.into_iter().map(|(jobs,items,bytes)|(jobs,(items,bytes))).unzip();
    let (verify,guarded)=match options.verify{Verify::Full=>(true,false),Verify::Light=>(false,true)};
    let started=Instant::now();
    let pools=if snapshot {
        ReaderPool::start_retained_components_scheduled(jobs,Config{budget_bytes:options.retained_budget,..options.config},verify,guarded,schedule)?
    }else {ReaderPool::start_components_scheduled(jobs,options.config,options.direct,verify,guarded,policy,schedule)?};
    let pending=pools.into_iter().zip(layouts).zip(retained_bounds).map(|((pool,(items,read_bytes)),bound)| {
        let mut component_options=options;
        if snapshot {component_options.retained_budget=bound;}
        Ok(Pending{pool,items,options:component_options,started,read_bytes,stop_prefetch_after_upload:true,host_snapshot:snapshot})
    }).collect::<Result<Vec<_>>>()?;
    Ok(Components{pending:Some(pending),ready:Vec::new(),workers:Vec::new(),registrations:None,started,first_finished:false,context:0,
        reader_budget_bytes:options.config.budget_bytes,shared_readers:true,deferred:vec![]})
}
/// The same tensor/receipt contract with separately declared component
/// reader cohorts. Every quota is admitted against one aggregate budget.
pub fn start_components_with_reader_configs(manifest:&Manifest,roots:&HashMap<String,PathBuf>,components:Vec<Vec<Request>>,
    options:Options,policy:crate::weight_loader::FileReadPolicy,schedule:crate::weight_loader::ReadSchedule,configs:Vec<Config>)->Result<Components>{
    let deferred=vec![false;components.len()];
    start_components_with_deferred_readers(manifest,roots,components,options,policy,schedule,configs,&deferred)
}
/// Admit every component quota together, but create no reader/buffer/file for
/// explicitly deferred groups until their declared load callback is released.
pub fn start_components_with_deferred_readers(manifest:&Manifest,roots:&HashMap<String,PathBuf>,components:Vec<Vec<Request>>,
    options:Options,policy:crate::weight_loader::FileReadPolicy,schedule:crate::weight_loader::ReadSchedule,configs:Vec<Config>,deferred:&[bool])->Result<Components>{
    if components.is_empty()||components.len()!=configs.len()||options.ahead_budget!=0||options.retained_budget!=0{
        return Err("component reader quotas require bounded independent streaming groups".into());
    }
    if deferred.len()!=components.len(){return Err("deferred component mask differs from admitted groups".into());}
    let admitted=crate::weight_loader::component_reader_configs(options.config,&configs.iter().map(|c|c.workers).collect::<Vec<_>>(),options.direct,policy)?;
    if configs!=admitted{return Err("component reader configurations differ from their admitted quotas".into());}
    let descriptions=component_descriptions(manifest,roots,&components,options)?;
    let (verify,guarded)=match options.verify{Verify::Full=>(true,false),Verify::Light=>(false,true)};
    let started=Instant::now();
    // Admission above checks the entire aggregate before any pool starts.
    // Match separate native owners: one component's buffer allocation and
    // reader setup must not delay every other independent component's I/O.
    let results=std::thread::scope(|scope| {
        let workers=descriptions.into_iter().zip(configs).enumerate().map(|(index,((jobs,items,read_bytes),config))| {
            let defer=deferred[index];
            scope.spawn(move ||->Result<std::result::Result<Pending,(usize,DeferredComponent)>> {
                if defer {return Ok(Err((index,DeferredComponent::Reader{jobs,items,options:Options{config,..options},policy,schedule,read_bytes})));}
                let _trace=crate::startup_trace::Span::new("startup_reader_prepare",format!("component pool {index}"),0);
                let pool=ReaderPool::start_components_scheduled(vec![jobs],config,options.direct,verify,guarded,policy,schedule)?.pop().ok_or("component reader pool missing")?;
                Ok(Ok(Pending{pool,items,options:Options{config,..options},started,read_bytes,stop_prefetch_after_upload:true,host_snapshot:false}))
            })
        }).collect::<Vec<_>>();
        // Join every setup owner, including after an error. Completed pools
        // close their own reader cohorts if a peer cannot be created.
        let results=workers.into_iter().map(|worker|worker.join().unwrap_or_else(|_|Err("component reader setup panicked".into())))
            .collect::<Vec<_>>();
        results.into_iter().collect::<Result<Vec<_>>>()
    })?;
    let(mut pending,mut deferred)=(vec![],vec![]);
    for result in results {match result {Ok(p)=>pending.push(p),Err(p)=>deferred.push(p)}}
    Ok(Components{pending:Some(pending),ready:vec![],workers:vec![],registrations:None,started,first_finished:false,context:0,
        reader_budget_bytes:options.config.budget_bytes,shared_readers:false,deferred})
}
impl Components {
    /// Resume independent uploads from retained, verified component sources.
    /// Metadata admission completes before any allocation or thread starts.
    ///
    /// # Safety
    /// The current CUDA context must outlive this value and its RestHandle.
    pub unsafe fn from_host(manifest:std::sync::Arc<Manifest>,requests:Vec<Vec<Request>>,
        hosts:Vec<std::sync::Arc<HostWeights>>)->Result<Self> {
        let deferred=vec![false;requests.len()];
        unsafe{Self::from_host_with_deferred(manifest,requests,hosts,&deferred)}
    }
    /// Retained sources obey the same per-component activation policy. This
    /// keeps an explicit host warm from eagerly uploading a deferred decoder.
    /// # Safety
    /// The current context must outlive this cohort and all returned uploads.
    pub unsafe fn from_host_with_deferred(manifest:std::sync::Arc<Manifest>,requests:Vec<Vec<Request>>,
        hosts:Vec<std::sync::Arc<HostWeights>>,deferred:&[bool])->Result<Self> {
        if requests.is_empty() || requests.len()!=hosts.len() {return Err("retained component count differs".into());}
        if deferred.len()!=requests.len(){return Err("retained deferred component mask differs".into());}
        for (requests,host) in requests.iter().zip(&hosts) {host_items(&manifest,requests,host)?;}
        let started=Instant::now();
        let context=crate::device_context::current_identity().map_err(|e|e.to_string())?;
        let mut value=Self {pending:None,ready:vec![],workers:vec![],registrations:None,started,first_finished:false,context,reader_budget_bytes:0,shared_readers:false,deferred:vec![]};
        for (index,(requests,host)) in requests.into_iter().zip(hosts).enumerate() {
            if deferred[index] {
                value.deferred.push((index,DeferredComponent::Host{manifest:manifest.clone(),requests,host}));continue;
            }
            let manifest=manifest.clone();let (send,receive)=crate::startup_result::channel();value.ready.push(receive);
            let worker=std::thread::Builder::new().name("retained-component-upload".into()).spawn(move || {
                let result=(||->Result<(Loaded,f64)> {
                    let _context=unsafe{crate::device_context::CurrentContext::enter(context)}.map_err(|e|e.to_string())?;
                    let mut stream=std::ptr::null_mut();check(unsafe{cudaStreamCreateWithFlags(&mut stream,1)},"retained component stream")?;
                    let result=unsafe{Loaded::from_host(&manifest,&requests,host,stream)};
                    unsafe{cudaStreamDestroy(stream);}
                    result.map(|loaded|(loaded,started.elapsed().as_secs_f64()))
                })();
                send.send(result);
            }).map_err(|e|format!("retained component upload worker: {e}"))?;
            value.workers.push(worker);
        }
        Ok(value)
    }
    /// Whether uploads have taken ownership of the reusable reader pools.
    /// A caller can reject host-only takeover without consuming this value.
    pub fn device_started(&self)->bool{self.pending.is_none()}
    /// Adopt the original CPU read progress into a retained host cache, then
    /// register its pages on the caller's current CUDA context. This command
    /// cannot take over a load after independent GPU uploads have started.
    pub fn finish_host(self,retained_budget:usize)->Result<Vec<std::sync::Arc<HostWeights>>> {
        self.finish_host_unpinned(retained_budget)?.into_iter().map(HostWeights::pin).collect()
    }
    /// CPU-only takeover. All component consumers must drain concurrently:
    /// one component cannot monopolize the cohort's shared reusable buffers.
    /// Admit the complete aligned output PLUS the existing reader budget
    /// before any consumer can allocate retained storage or publish a result.
    pub fn finish_host_unpinned(mut self,retained_budget:usize)->Result<Vec<HostWeights>> {
        if !self.deferred.is_empty() {
            // Explicit host materialization intentionally includes all groups.
            // Preserve manifest order and the already admitted aggregate reader
            // budget; no late component reopens any existing reader's source.
            let mut eager=self.pending.take().ok_or("host takeover requires CPU-only component readers")?.into_iter();
            let mut deferred=std::mem::take(&mut self.deferred).into_iter().collect::<std::collections::BTreeMap<_,_>>();
            let count=eager.len()+deferred.len();let mut pending=Vec::with_capacity(count);
            for index in 0..count {
                pending.push(match deferred.remove(&index){Some(source)=>source.readers(self.started)?,None=>eager.next().ok_or("deferred host reader order")?});
            }
            self.pending=Some(pending);
        }
        let pending=self.pending.as_ref().ok_or("host takeover requires CPU-only component readers")?;
        let buffers=self.reader_budget_bytes;
        let mut total=buffers;
        let mut bounds=Vec::with_capacity(pending.len());
        for component in pending {
            let sizes=component.items.iter().map(|item|item.elements.checked_mul(item.stored.size()).ok_or("retained tensor extent overflow".into()))
                .collect::<Result<Vec<_>>>()?;
            let bytes=crate::weight_loader::retained_copy_bound(&sizes,component.options.config.chunk_bytes)?;
            total=total.checked_add(bytes).ok_or("retained component budget overflow")?;
            bounds.push(bytes);
        }
        if total>retained_budget{return Err(format!("component host takeover needs {total} bytes including readers, budget is {retained_budget}"));}
        let pending=self.pending.take().unwrap();
        std::thread::scope(|scope|{
            let workers=pending.into_iter().zip(bounds).map(|(pending,bound)|scope.spawn(move||->Result<HostWeights>{
                let _=crate::cpu_priority::background();
                let Pending{mut pool,items,mut options,..}=pending;
                let chunks=if pool.owns_retained_read_buffers() {
                    pool.consume_all(|_,_,_|Ok(()))?;
                    pool.take_retained()?.ok_or("retained component source missing")?
                }else {pool.finish_retained_copy(bound)?};
                let indices=items.into_iter().enumerate().map(|(i,item)|(item.name,i)).collect();
                options.retained_budget=bound;
                Ok(HostWeights{chunks,indices,options,registrations:std::sync::Mutex::new(None)})
            })).collect::<Vec<_>>();
            // Join every owner even when an earlier component fails. Failed
            // reader pools close the shared schedule and cancel its peers.
            let results=workers.into_iter().map(|worker|worker.join().map_err(|_|"component host takeover worker panicked".to_string())?)
                .collect::<Vec<_>>();
            results.into_iter().collect()
        })
    }
    /// Start every independent upload without waiting for component 0. The
    /// serving thread can prepare context-bound metadata while DMA proceeds.
    /// # Safety
    /// The current CUDA context must outlive this value and its RestHandle.
    /// Drop joins the uploads before the caller may destroy that context.
    pub unsafe fn device_ready(&mut self)->Result<()> {
        if self.pending.is_none(){return Ok(());}
        self.context=crate::device_context::current_identity().map_err(|e|e.to_string())?;
        self.registrations=if self.shared_readers{Some(crate::weight_loader::device::SharedRegistrations::new()?)}else{None};
        for pending in self.pending.take().unwrap(){
            let context=self.context;let registrations=self.registrations.clone();let started=self.started;
            let (send,receive)=crate::startup_result::channel();self.ready.push(receive);
            let worker=std::thread::Builder::new().name("component-upload".into()).spawn(move||{
                let result=(||->Result<(Loaded,f64)>{
                    crate::startup_trace::measure("cuda_context_bind","component-upload",0,||
                        unsafe{crate::device_context::bind_current(context)}).map_err(|e|e.to_string())?;
                    let mut stream=std::ptr::null_mut();crate::startup_trace::measure("cuda_stream_create","component-upload",0,||
                        check(unsafe{cudaStreamCreateWithFlags(&mut stream,1)},"component stream"))?;
                    let result=unsafe{pending.finish_registered(stream,registrations)};
                    crate::startup_trace::measure("cuda_stream_destroy","component-upload",0,||unsafe{cudaStreamDestroy(stream);});
                    result.map(|loaded|(loaded,started.elapsed().as_secs_f64()))
                })();
                send.send(result);
            }).map_err(|e|format!("component upload worker: {e}"))?;
            self.workers.push(worker);
        }
        Ok(())
    }
    /// # Safety
    /// The current CUDA context and `stream` must outlive this load. Each
    /// component's own stream is drained before its views become available.
    pub unsafe fn finish_first(&mut self,_stream:Stream)->Result<Loaded>{
        if !self.deferred.is_empty(){return Err("deferred components require independent result handles".into());}
        if self.first_finished{return Err("first component already finished".into());}
        unsafe{self.device_ready()}?;
        let (loaded,_)=self.ready.remove(0).wait()?;
        self.first_finished=true;Ok(loaded)
    }
    pub fn spawn_rest(mut self)->Result<RestHandle>{
        if !self.first_finished{return Err("first component must be joined before its dependents".into());}
        let ready=std::mem::take(&mut self.ready);let workers=std::mem::take(&mut self.workers);
        let (send,stages)=std::sync::mpsc::channel();
        drop(send);
        Ok(RestHandle{worker:None,stages,components:Some(ComponentRest{ready:ready.into(),workers}),
            _registrations:self.registrations.take()})
    }
    /// Transfer all independent receipts without waiting for any component.
    /// The returned owner joins every upload on success, failure and close.
    pub fn into_uploads(mut self)->Result<IndependentUploads> {
        if self.pending.is_some() || self.first_finished {return Err("independent uploads must be activated before subscribing".into());}
        let deferred=std::mem::take(&mut self.deferred).into_iter().collect::<std::collections::BTreeMap<_,_>>();
        let count=self.ready.len()+deferred.len();let mut eager=std::mem::take(&mut self.ready).into_iter();
        let ready=(0..count).map(|index|if deferred.contains_key(&index){None}else{eager.next()}).collect();
        Ok(IndependentUploads {ready,deferred,context:self.context,started:self.started,
            workers:std::mem::take(&mut self.workers),_registrations:self.registrations.take()})
    }
}
pub struct IndependentUploads {
    ready:Vec<Option<crate::startup_result::ResultReceiver<(Loaded,f64)>>>,
    workers:Vec<std::thread::JoinHandle<()>>,
    _registrations:Option<std::sync::Arc<crate::weight_loader::device::SharedRegistrations>>,
    deferred:std::collections::BTreeMap<usize,DeferredComponent>,context:usize,started:Instant,
}
impl IndependentUploads {
    pub fn take(&mut self,index:usize)->Result<crate::startup_result::ResultReceiver<(Loaded,f64)>> {
        if let Some(source)=self.deferred.remove(&index) {
            let(context,started)=(self.context,self.started);let(send,receive)=crate::startup_result::channel();
            let worker=std::thread::Builder::new().name("deferred-component-upload".into()).spawn(move || {
                let result=(||->Result<(Loaded,f64)> {
                    let _context=unsafe{crate::device_context::CurrentContext::enter(context)}.map_err(|e|e.to_string())?;
                    let mut stream=std::ptr::null_mut();check(unsafe{cudaStreamCreateWithFlags(&mut stream,1)},"deferred component stream")?;
                    let result=unsafe{source.upload(stream,started)};
                    unsafe{cudaStreamDestroy(stream);}
                    result.map(|loaded|(loaded,started.elapsed().as_secs_f64()))
                })();send.send(result);
            }).map_err(|e|format!("deferred component upload worker: {e}"))?;
            self.workers.push(worker);self.ready[index]=Some(receive);
        }
        self.ready.get_mut(index).and_then(Option::take).ok_or_else(||"independent receipt unavailable or already subscribed".into())
    }
}
impl Drop for IndependentUploads {
    fn drop(&mut self) {for worker in self.workers.drain(..){let _=worker.join();}}
}
impl Drop for Components {
    fn drop(&mut self){
        self.pending.take();
        for worker in self.workers.drain(..){let _=worker.join();}
    }
}

/// CPU RAM-cache preparation is separate from GPU readiness. Its owner joins
/// on success, error and cancellation; only finish() pins the resulting pages
/// in the calling thread's live CUDA context. No CUDA call runs on this worker.
pub struct HostPreparation {worker:Option<std::thread::JoinHandle<Result<Vec<HostWeights>>>>}
impl HostPreparation {
    pub fn start(manifest:Manifest,roots:HashMap<String,PathBuf>,stages:Vec<Vec<Request>>,options:Options)->Result<Self> {
        if options.retained_budget==0{return Err("RAM cache requires an admitted retained byte budget".into());}
        let worker=std::thread::Builder::new().name("host-weight-cache".into()).spawn(move||{
            let _=crate::cpu_priority::background();
            let _span=crate::startup_trace::Span::new("host_cache_prepare","CPU RAM cache",0);
            start_staged(manifest,roots,stages,options)?.finish_host_unpinned()
        }).map_err(|e|format!("host cache thread: {e}"))?;
        Ok(Self{worker:Some(worker)})
    }
    pub fn finish(mut self)->Result<Vec<std::sync::Arc<HostWeights>>> {
        self.worker.take().unwrap().join().map_err(|_|"host cache thread panicked")??
            .into_iter().map(HostWeights::pin).collect()
    }
}
impl Drop for HostPreparation {
    fn drop(&mut self){if let Some(worker)=self.worker.take(){let _=worker.join();}}
}


#[cfg(test)]
mod staged_prefetch_handoff_tests {
    use super::*;
    use sha2::{Digest,Sha256};
    use std::sync::atomic::{AtomicUsize,Ordering};

    struct Fixture {root:PathBuf,manifest:Manifest,roots:HashMap<String,PathBuf>}
    impl Fixture {
        fn new()->Self {
            static NEXT:AtomicUsize=AtomicUsize::new(0);
            let root=std::env::temp_dir().join(format!("staged-prefetch-{}-{}",std::process::id(),NEXT.fetch_add(1,Ordering::Relaxed)));
            std::fs::create_dir(&root).unwrap();
            let mut files=serde_json::Map::new();let mut tensors=serde_json::Map::new();
            for (index,name) in ["encoder","dit","tail"].into_iter().enumerate() {
                let bytes:Vec<u8>=(0..65536).map(|n|((n+index)%251) as u8).collect();
                let path=format!("{name}.raw");std::fs::write(root.join(&path),&bytes).unwrap();
                let sha=|data:&[u8]|format!("{:x}",Sha256::digest(data));
                files.insert(name.into(),serde_json::json!({"root":"weights","path":path,"bytes":bytes.len(),"sha256":sha(&bytes)}));
                tensors.insert(name.into(),serde_json::json!({"file":name,"offset":0,"bytes":bytes.len(),"dtype":"BF16","shape":[bytes.len()/2],
                    "sha256":sha(&bytes),"head_sha256":sha(&bytes[..1024]),"tail_sha256":sha(&bytes[bytes.len()-1024..])}));
            }
            let json=serde_json::to_vec(&serde_json::json!({"schema":"native-checkpoint-v1","segment_bytes":1024,"files":files,"tensors":tensors})).unwrap();
            let manifest=Manifest::parse(&json,None).unwrap();
            let roots=HashMap::from([("weights".into(),root.clone())]);
            Self{root,manifest,roots}
        }
        fn start(&self,stages:Vec<Vec<Request>>,options:Options)->Staged {
            start_staged(self.manifest.clone(),self.roots.clone(),stages,options).unwrap()
        }
    }
    impl Drop for Fixture {fn drop(&mut self){let _=std::fs::remove_dir_all(&self.root);}}
    fn options()->Options {Options{config:Config{workers:1,chunk_bytes:4096,pinned_slots:1,budget_bytes:16384},
        direct:false,balanced:true,verify:Verify::Full,staging_slots:1,ahead_budget:0,retained_budget:0}}
    fn request(name:&str)->Vec<Request>{vec![Request{name:name.into(),dtype:Dtype::BF16}]}
    // Drain the real bounded readers and full SHA checks without CUDA. Upload
    // ownership is decided by the same Pending instances production finishes.
    fn drain(mut pending:Pending)->Result<usize> {
        let mut bytes=0;pending.pool.consume_all(|_,_,chunk|{bytes+=chunk.len();Ok(())})?;
        Ok(bytes)
    }

    #[test]
    fn last_deferred_payload_owns_stop_across_empty_and_trailing_empty_stages(){
        let f=Fixture::new();
        let mut staged=f.start(vec![request("encoder"),vec![],request("dit"),vec![]],options());
        let first=staged.first.take().unwrap();assert!(!first.stop_prefetch_after_upload);
        assert_eq!(drain(first).unwrap(),65536);
        let mut stages=staged.rest.into_iter();
        let empty=stages.next().unwrap().start_for_upload(&f.manifest,&f.roots,options(),stages.as_slice()).unwrap();
        assert!(!empty.stop_prefetch_after_upload);assert_eq!(drain(empty).unwrap(),0);
        let dit=stages.next().unwrap().start_for_upload(&f.manifest,&f.roots,options(),stages.as_slice()).unwrap();
        assert!(dit.stop_prefetch_after_upload);assert_eq!(drain(dit).unwrap(),65536);
        let trailing=stages.next().unwrap().start_for_upload(&f.manifest,&f.roots,options(),stages.as_slice()).unwrap();
        assert!(trailing.stop_prefetch_after_upload);assert_eq!(drain(trailing).unwrap(),0);
        // Empty loads submit no H2D and therefore cannot touch Upload's stop file.
    }
    #[test]
    fn each_nonfinal_deferred_payload_hands_stop_forward(){
        let f=Fixture::new();let mut staged=f.start(vec![request("encoder"),request("dit"),request("tail")],options());
        let first=staged.first.take().unwrap();assert!(!first.stop_prefetch_after_upload);drain(first).unwrap();
        let mut stages=staged.rest.into_iter();
        let middle=stages.next().unwrap().start_for_upload(&f.manifest,&f.roots,options(),stages.as_slice()).unwrap();
        assert!(!middle.stop_prefetch_after_upload);drain(middle).unwrap();
        let last=stages.next().unwrap().start_for_upload(&f.manifest,&f.roots,options(),stages.as_slice()).unwrap();
        assert!(last.stop_prefetch_after_upload);drain(last).unwrap();
    }
    #[test]
    fn no_future_payload_preserves_first_stop_including_all_empty_schedule(){
        let f=Fixture::new();
        for stages in [vec![request("encoder")],vec![request("encoder"),vec![],vec![]],vec![vec![],vec![]]] {
            let mut staged=f.start(stages,options());let first=staged.first.take().unwrap();
            assert!(first.stop_prefetch_after_upload);drain(first).unwrap();
            // Dropping never-started empty stages creates no readers/upload.
        }
    }
    #[test]
    fn admitted_ahead_readers_keep_original_stop_and_fallback_hands_it_forward(){
        let f=Fixture::new();let stages=vec![request("encoder"),request("dit")];
        let mut ahead=options();ahead.ahead_budget=1<<20;
        let mut staged=f.start(stages.clone(),ahead);assert_ne!(staged.options.ahead_budget,0);
        let first=staged.first.take().unwrap();assert!(first.stop_prefetch_after_upload);drain(first).unwrap();
        assert!(staged.rest.iter().all(|s|matches!(s,Stage::Reading(_))));
        for stage in staged.rest {let pending=stage.start_for_upload(&f.manifest,&f.roots,ahead,&[]).unwrap();
            assert!(pending.stop_prefetch_after_upload);drain(pending).unwrap();}
        let mut fallback=options();fallback.ahead_budget=1;
        let mut staged=f.start(stages,fallback);assert_eq!(staged.options.ahead_budget,0);
        let first=staged.first.take().unwrap();assert!(!first.stop_prefetch_after_upload);drain(first).unwrap();
        let pending=staged.rest.pop().unwrap().start_for_upload(&f.manifest,&f.roots,staged.options,&[]).unwrap();
        assert!(pending.stop_prefetch_after_upload);drain(pending).unwrap();
    }
    #[test]
    fn retained_readers_keep_original_stop_and_host_completion_remains_cpu_only(){
        let f=Fixture::new();let mut retained=options();retained.retained_budget=1<<20;
        let staged=f.start(vec![request("encoder"),request("dit")],retained);
        assert!(staged.first.as_ref().unwrap().stop_prefetch_after_upload);
        assert!(staged.rest.iter().all(|s|matches!(s,Stage::Reading(p) if p.stop_prefetch_after_upload)));
        let hosts=staged.finish_host_unpinned().unwrap();assert_eq!(hosts.len(),2);
        assert!(hosts.iter().all(|host|host.bytes()==65536 && host.pinned_bytes()==0));
    }
    #[test]
    fn retained_reload_admits_exact_extents_and_dtype_without_reopening_sources(){
        let f=Fixture::new();let mut opts=options();opts.retained_budget=1<<20;
        let mut hosts=f.start(vec![request("encoder")],opts).finish_host_unpinned().unwrap();
        let host=hosts.pop().unwrap();std::fs::remove_file(f.root.join("encoder.raw")).unwrap();
        assert_eq!(host_items(&f.manifest,&request("encoder"),&host).unwrap().len(),1);
        assert!(host_items(&f.manifest,&request("dit"),&host).err().unwrap().contains("tensors differ"));
        let duplicate=[request("encoder"),request("encoder")].concat();
        assert!(host_items(&f.manifest,&duplicate,&host).err().unwrap().contains("tensors differ"));
        let mut conversion=request("encoder");conversion[0].dtype=Dtype::F32;
        assert!(host_items(&f.manifest,&conversion,&host).err().unwrap().contains("dtypes"));
        let mut changed=f.manifest.clone();changed.tensors.get_mut("encoder").unwrap().bytes=32768;
        assert!(host_items(&changed,&request("encoder"),&host).err().unwrap().contains("extent differs"));
    }
    #[test]
    fn standalone_and_shared_components_keep_original_stop_ownership(){
        let f=Fixture::new();let pending=start(&f.manifest,&f.roots,&request("encoder"),options()).unwrap();
        assert!(pending.stop_prefetch_after_upload);drain(pending).unwrap();
        let mut opts=options();opts.config.budget_bytes=32768;
        let mut components=start_components(&f.manifest,&f.roots,vec![request("encoder"),request("dit")],opts).unwrap();
        let pending=components.pending.take().unwrap();assert!(pending.iter().all(|p|p.stop_prefetch_after_upload));
        std::thread::scope(|scope|{
            let readers=pending.into_iter().map(|p|scope.spawn(move||drain(p))).collect::<Vec<_>>();
            for reader in readers {assert_eq!(reader.join().unwrap().unwrap(),65536);}
        });
    }
    #[test]
    fn deferred_failure_and_cancel_join_real_bounded_readers(){
        let f=Fixture::new();let mut staged=f.start(vec![request("encoder"),request("dit")],options());
        let first=staged.first.take().unwrap();assert!(!first.stop_prefetch_after_upload);drain(first).unwrap();
        // The future stage has not opened this payload. Corruption must still
        // fail its full verifier; stop handoff cannot waive the original guard.
        std::fs::write(f.root.join("dit.raw"),vec![0;65536]).unwrap();
        let mut pending=staged.rest.pop().unwrap().start_for_upload(&f.manifest,&f.roots,options(),&[]).unwrap();
        assert!(pending.stop_prefetch_after_upload);
        assert!(pending.pool.consume_all(|_,_,_|Ok(())).is_err());drop(pending);
        let pending=start(&f.manifest,&f.roots,&request("encoder"),options()).unwrap();
        let (send,done)=std::sync::mpsc::channel();
        let worker=std::thread::spawn(move||{
            let mut pending=pending;
            assert!(pending.pool.consume_all(|_,_,_|Err("cancel before final staged upload".into())).is_err());
            // More chunks than the bounded queue can hold: Drop must disconnect
            // and join a reader even when it is waiting for recycle/backpressure.
            drop(pending);send.send(()).unwrap();
        });
        done.recv_timeout(std::time::Duration::from_secs(5)).expect("cancelled reader did not join");worker.join().unwrap();
    }
}
