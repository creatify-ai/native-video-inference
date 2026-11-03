//! Immutable, fully verified pinned DiT bytes survive GPU model eviction.
//! First fill retains the existing bounded readers and full SHA checks. A hit
//! uses those same bytes directly as H2D sources, with no payload read/hash/copy.
use crate::weight_loader::{Config, Job, ReaderPool};
use crate::Result;
use serde_json::{json, Value};
use std::{cell::RefCell, collections::HashMap, ffi::{c_char,c_int,c_void,CStr},
          path::PathBuf, ptr, sync::Arc, time::Instant};

unsafe extern "C" {
    fn cudaHostAlloc(pointer:*mut *mut c_void, bytes:usize, flags:u32)->c_int;
    fn cudaFreeHost(pointer:*mut c_void)->c_int;
    fn cudaDeviceSynchronize()->c_int;
    fn cudaStreamSynchronize(stream:*mut c_void)->c_int;
    fn cudaMemcpyAsync(dst:*mut c_void, src:*const c_void, bytes:usize, kind:c_int, stream:*mut c_void)->c_int;
    fn cudaGetErrorString(code:c_int)->*const c_char;
    fn cudaMemGetInfo(free:*mut usize,total:*mut usize)->c_int;
}
fn cuda(code:c_int)->Result<()> {
    if code==0 {Ok(())} else {
        Err(format!("pinned DiT cache CUDA {code}: {}",
            unsafe {CStr::from_ptr(cudaGetErrorString(code))}.to_string_lossy()).into())
    }
}

pub fn enabled()->Result<bool> {
    match std::env::var("H3_DIT_HOST_CACHE").as_deref() {
        Err(_)|Ok("0")=>Ok(false), Ok("1")=>{
            if std::env::var("H3_CP_WORLD").is_ok_and(|v|v!="1")
                || std::env::var_os("H3_DIT_MODEL_CATALOG").is_some()
                || (std::env::var("H3_MODEL_LOAD_ASYNC_SHA").is_ok_and(|v|v!="0")
                    && std::env::var("H3_DIT_CACHE_ASYNC_SHA").as_deref()!=Ok("1")) {
                return Err("pinned DiT cache requires a fixed single-GPU model and foreground SHA".into());
            }
            Ok(true)
        }, _=>Err("H3_DIT_HOST_CACHE must be 0 or 1".into()),
    }
}

#[derive(Clone,Hash,Eq,PartialEq)]
struct Key { path:PathBuf, bytes:usize, sha256:String, file_offset:u64,file_bytes:Option<u64> }
impl Key {
    fn new(job:&Job)->Result<Self> {
        if !job.path.is_absolute() || job.bytes==0 {
            return Err("pinned cache requires an absolute path and nonempty payload".into());
        }
        let sha256=job.sha256.clone().ok_or("pinned cache requires full SHA256 for each payload")?;
        if sha256.len()!=64 || !sha256.bytes().all(|b|b.is_ascii_hexdigit() && !b.is_ascii_uppercase()) {
            return Err("pinned cache digest is invalid".into());
        }
        Ok(Self {path:job.path.clone(),bytes:job.bytes,sha256,file_offset:job.file_offset,file_bytes:job.file_bytes})
    }
}

struct Allocation {pointer:*mut u8, bytes:usize}
// Only immutable, fully loaded entries cross to the audit worker. A prefill
// worker exclusively owns a new allocation until size/guard validation passes.
unsafe impl Send for Allocation {}
unsafe impl Sync for Allocation {}
#[path="background_cache.rs"] mod background;
pub use background::ComputePermit;
fn async_sha()->Result<bool>{match std::env::var("H3_DIT_CACHE_ASYNC_SHA").as_deref(){Err(_)|Ok("0")=>Ok(false),Ok("1")=>Ok(true),_=>Err("H3_DIT_CACHE_ASYNC_SHA must be 0 or 1".into())}}

/// Optional originals must never be filled by a foreground upload.
pub fn deferred()->Result<bool> {
    match std::env::var("H3_DIT_CACHE_DEFER_ORIGINAL").as_deref() {
        Err(_)|Ok("0")=>Ok(false), Ok("1")=>{
            if std::env::var_os("H3_DIT_ON_DEMAND_CATALOG").is_none()
                || std::env::var("H3_DIT_CACHE_ASYNC_SHA").as_deref()!=Ok("1")
                || std::env::var("H3_DIT_CACHE_REQUEST_PRIORITY").as_deref()!=Ok("1") {
                return Err("deferred originals require the guarded on-demand catalog and request priority".into());
            }
            Ok(true)
        },_=>Err("H3_DIT_CACHE_DEFER_ORIGINAL must be 0 or 1".into()),
    }
}

impl Allocation {
    fn new(bytes:usize)->Result<Self> {
        let mut pointer=ptr::null_mut();
        let result=unsafe {cudaHostAlloc(&mut pointer,bytes,0)};
        if pointer.is_null() {
            cuda(result)?;
            return Err("CUDA returned a null pinned allocation".into());
        }
        // Retain ownership even when an API reports an earlier async error
        // after returning a nonnull allocation.
        let allocation=Self {pointer:pointer.cast(),bytes};
        cuda(result)?;
        Ok(allocation)
    }
}
impl Drop for Allocation {
    fn drop(&mut self) {
        // This allocation may span several old/new model streams. The device
        // drain covers all of them before freeing any queued H2D source.
        unsafe {cudaDeviceSynchronize(); cudaFreeHost(self.pointer.cast());}
    }
}
#[derive(Clone)]
struct Entry {allocation:Arc<Allocation>, offset:usize, bytes:usize, finite_bf16_verified:bool}
impl Entry {
    fn pointer(&self,offset:usize,bytes:usize)->Result<*mut u8> {
        if offset.checked_add(bytes).is_none_or(|n|n>self.bytes)
            || self.offset.checked_add(self.bytes).is_none_or(|n|n>self.allocation.bytes) {
            return Err("pinned cache view exceeds verified bytes".into());
        }
        Ok(unsafe {self.allocation.pointer.add(self.offset+offset)})
    }
}

#[derive(Default)]
struct Counters {
    loads:usize, payload_read_bytes:usize, payload_sha_bytes:usize,
    fill_copy_bytes:usize, cache_hit_files:usize, cache_hit_upload_bytes:usize,
    upload_bytes:usize, pin_alloc_seconds:f64, h2d_api_seconds:f64,
    upload_and_wait_seconds:f64,
}
pub struct Cache {
    entries:HashMap<Key,Entry>, budget:usize, allocated:usize, counters:Counters, background:Option<background::Background>,
}
impl Cache {
    fn from_env()->Result<Self> {
        let gib:usize=std::env::var("H3_DIT_HOST_CACHE_GIB").unwrap_or_else(|_|"43".into()).parse()?;
        let maximum=if std::env::var_os("H3_DIT_ON_DEMAND_CATALOG").is_some() {96}else{64};
        if !(1..=maximum).contains(&gib) {return Err(format!("H3_DIT_HOST_CACHE_GIB must be 1..={maximum}").into());}
        Ok(Self {entries:HashMap::new(),budget:gib.checked_mul(1<<30).ok_or("pinned budget overflow")?,
                 allocated:0,counters:Counters::default(),background:if async_sha()? {Some(background::Background::new()?)}else{None}})
    }

    /// Destination allocations belong to the caller. No CUDA destination is
    /// retained here; only immutable host bytes outlive a GPU model owner.
    unsafe fn upload(&mut self,jobs:&[Job],destinations:&[*mut c_void],stream:*mut c_void,config:Config,
                     finite_bf16:bool)->Result<Value> {
        if jobs.len()!=destinations.len() || destinations.iter().any(|p|p.is_null()) {
            return Err("pinned cache destination count/pointer mismatch".into());
        }
        config.validate()?;
        // No active generation shares this command loop with a weight switch.
        // Pause optional IO; publish only completed immutable entries.
        if deferred()? {self.harvest();}
        else if let Some(background)=&mut self.background {let ready=background.pause_prefill();background.foreground();self.publish_background(ready);}
        let keys=jobs.iter().map(Key::new).collect::<Result<Vec<_>>>()?;
        if deferred()? && keys.iter().any(|key|!self.entries.contains_key(key)) {
            return Err("deferred original cache incomplete: foreground fill forbidden".into());
        }
        let mut missing=Vec::new();
        let mut positions=HashMap::new();
        let mut needed=0usize;
        for (index,key) in keys.iter().enumerate() {
            if self.entries.contains_key(key) || positions.contains_key(key) {continue;}
            let extent=key.bytes.checked_add(255).ok_or("pinned extent overflow")? & !255;
            positions.insert(key.clone(),needed);
            needed=needed.checked_add(extent).ok_or("pinned allocation overflow")?;
            missing.push(index);
        }
        if self.allocated.checked_add(needed).is_none_or(|n|n>self.budget) {
            return Err(format!("pinned DiT cache budget exceeded: retained={} additional={} budget={}",
                               self.allocated,needed,self.budget).into());
        }
        let allocated_at=Instant::now();
        let allocation=if needed>0 {Some(Arc::new(Allocation::new(needed)?))} else {None};
        let pin_alloc_seconds=if needed>0 {allocated_at.elapsed().as_secs_f64()} else {0.0};
        let mut views=keys.iter().map(|key|self.entries.get(key).cloned().unwrap_or_else(||Entry {
            allocation:allocation.as_ref().unwrap().clone(),offset:positions[key],bytes:key.bytes,
            finite_bf16_verified:false})).collect::<Vec<_>>();
        let read_bytes=missing.iter().try_fold(0usize,|n,&i|n.checked_add(jobs[i].bytes).ok_or("pinned read count overflow"))?;
        let mut h2d_api_seconds=0.0;
        let began=Instant::now();
        let operation=(||->Result<()> {
            // Only missing data is read. ReaderPool promotes no cache entry
            // until every complete payload's original SHA has passed.
            if !missing.is_empty() {
                let source=missing.iter().map(|&i|jobs[i].clone()).collect();
                let mut readers=if self.background.is_some(){ReaderPool::start_guarded(source,config)?}else{ReaderPool::start(source,config)?};
                readers.consume_all(|i,offset,data| {
                    let original=missing[i];
                    let host=views[original].pointer(offset,data.len()).map_err(|e|e.to_string())?;
                    unsafe {ptr::copy_nonoverlapping(data.as_ptr(),host,data.len());}
                    let begun=Instant::now();
                    let result=cuda(unsafe {cudaMemcpyAsync(destinations[original].cast::<u8>().add(offset).cast(),
                                                          host.cast(),data.len(),1,stream)});
                    h2d_api_seconds+=begun.elapsed().as_secs_f64();
                    result.map_err(|e|e.to_string())
                })?;
            }
            if finite_bf16 {
                for view in &mut views {
                    if view.finite_bf16_verified {continue;}
                    if view.bytes%2!=0 {return Err("post-training cache payload is not BF16".into());}
                    let bytes=unsafe {std::slice::from_raw_parts(view.pointer(0,view.bytes)?,view.bytes)};
                    if bytes.chunks_exact(2).any(|v|u16::from_le_bytes([v[0],v[1]]) & 0x7f80==0x7f80) {
                        return Err("nonfinite post-training weight".into());
                    }
                    view.finite_bf16_verified=true;
                }
            }
            for (index,key) in keys.iter().enumerate() {
                // Alias views of a newly read payload need their own GPU copy.
                // Already uploaded missing entries can be skipped here.
                if missing.contains(&index) {continue;}
                let view=&views[index];
                for offset in (0..key.bytes).step_by(config.chunk_bytes) {
                    let bytes=config.chunk_bytes.min(key.bytes-offset);
                    let begun=Instant::now();
                    cuda(unsafe {cudaMemcpyAsync(destinations[index].cast::<u8>().add(offset).cast(),
                                                view.pointer(offset,bytes)?.cast(),bytes,1,stream)})?;
                    h2d_api_seconds+=begun.elapsed().as_secs_f64();
                }
            }
            Ok(())
        })();
        // Even a failed launch/reader/hash is drained before unverified host
        // allocations can disappear. Failure never publishes partial entries.
        let synchronized=cuda(unsafe {cudaStreamSynchronize(stream)});
        operation?; synchronized?;
        let elapsed=began.elapsed().as_secs_f64();
        let bytes=jobs.iter().map(|j|j.bytes).sum::<usize>();
        let hit_files=keys.iter().filter(|k|self.entries.contains_key(*k)).count();
        let hit_bytes=keys.iter().filter(|k|self.entries.contains_key(*k)).map(|k|k.bytes).sum::<usize>();
        for (key,view) in keys.into_iter().zip(views) {self.entries.insert(key,view);}
        self.allocated+=needed;
        self.counters.loads+=1;self.counters.payload_read_bytes+=read_bytes;
        self.counters.payload_sha_bytes+=if self.background.is_some(){0}else{read_bytes};self.counters.fill_copy_bytes+=read_bytes;
        self.counters.cache_hit_files+=hit_files;self.counters.cache_hit_upload_bytes+=hit_bytes;
        self.counters.upload_bytes+=bytes;self.counters.pin_alloc_seconds+=pin_alloc_seconds;
        self.counters.h2d_api_seconds+=h2d_api_seconds;self.counters.upload_and_wait_seconds+=elapsed;
        Ok(json!({"files":jobs.len(),"cache_hit_files":hit_files,"cache_hit_upload_bytes":hit_bytes,
            "payload_read_bytes":read_bytes,"payload_sha_bytes":if self.background.is_some(){0}else{read_bytes},"fill_copy_bytes":read_bytes,
            "upload_bytes":bytes,"pin_alloc_seconds":pin_alloc_seconds,"h2d_api_seconds":h2d_api_seconds,
            "upload_and_wait_seconds":elapsed,"cache":self.report()}))
    }
    fn contains(&mut self,jobs:&[Job])->Result<bool> {
        self.harvest();
        if self.background.as_ref().is_some_and(|b|!b.healthy()){return Ok(false);}
        for job in jobs {if !self.entries.contains_key(&Key::new(job)?){return Ok(false);}}
        Ok(true)
    }
    fn publish_background(&mut self,ready:Vec<(Key,Entry)>){
        for (key,entry) in ready {
            if self.entries.contains_key(&key){continue;}
            self.allocated+=entry.allocation.bytes;self.entries.insert(key,entry);
        }
    }
    fn harvest(&mut self){if let Some(background)=&mut self.background {let ready=background.harvest();self.publish_background(ready);}}
    fn kick(&mut self,jobs:Vec<Job>)->Result<()> {
        self.harvest();
        let Some(background)=&mut self.background else{return Ok(());};
        // Bind current payloads to the independent audit before returning.
        for (key,entry) in &self.entries {background.submit(key.clone(),entry.clone())?;}
        let mut keys=std::collections::HashSet::new();let mut missing=Vec::new();let mut needed=0usize;
        for job in jobs {let key=Key::new(&job)?;if self.entries.contains_key(&key)||!keys.insert(key){continue;}
            needed=needed.checked_add(job.bytes.checked_add(255).ok_or("prefill extent overflow")?&!255).ok_or("prefill budget overflow")?;missing.push(job);}
        if self.allocated.checked_add(needed).is_none_or(|n|n>self.budget){
            if deferred()? {background.warn("optional RAM prefill exceeds pinned budget".into());return Ok(());}
            return Err("RAM prefill exceeds pinned budget".into());
        }
        let result=background.start_prefill(missing,needed);
        if deferred()? {if let Err(error)=result{background.warn(format!("optional RAM prefill: {error}"));}Ok(())}else{result}
    }
    fn report(&self)->Value {
        json!({"enabled":true,"fully_verified_files":self.background.as_ref().map_or(self.entries.len(),|b|b.report()["passed_files"].as_u64().unwrap_or(0) as usize),"guarded_files":self.entries.len(),"background":self.background.as_ref().map(|b|b.report()),"pinned_bytes":self.allocated,
            "budget_bytes":self.budget,"loads":self.counters.loads,
            "payload_read_bytes":self.counters.payload_read_bytes,"payload_sha_bytes":self.counters.payload_sha_bytes,
            "fill_copy_bytes":self.counters.fill_copy_bytes,"cache_hit_files":self.counters.cache_hit_files,
            "cache_hit_upload_bytes":self.counters.cache_hit_upload_bytes,"upload_bytes":self.counters.upload_bytes,
            "pin_alloc_seconds":self.counters.pin_alloc_seconds,"h2d_api_seconds":self.counters.h2d_api_seconds,
            "upload_and_wait_seconds":self.counters.upload_and_wait_seconds})
    }
}
thread_local! {static CACHE:RefCell<Option<Cache>>=const {RefCell::new(None)};}

/// The cache is single-thread owned by this fixed resident's CUDA command loop.
pub unsafe fn upload(jobs:&[Job],destinations:&[*mut c_void],stream:*mut c_void,finite_bf16:bool)->Result<Value> {
    if !enabled()? {return Err("pinned DiT cache is not enabled".into());}
    let config=Config::from_env()?.ok_or("pinned DiT cache requires the bounded model pipeline")?;
    CACHE.with(|cell| {
        let mut cache=cell.borrow_mut();
        if cache.is_none() {*cache=Some(Cache::from_env()?);}
        unsafe {cache.as_mut().unwrap().upload(jobs,destinations,stream,config,finite_bf16)}
    })
}
pub fn report()->Value {
    CACHE.with(|cell| {let mut held=cell.borrow_mut();if let Some(cache)=held.as_mut(){cache.harvest();cache.report()}else{json!({"enabled":false})}})
}
pub fn synchronize()->Result<()> {cuda(unsafe {cudaDeviceSynchronize()})}
pub fn device_memory()->Result<Value> {
    let(mut free,mut total)=(0usize,0usize);
    cuda(unsafe {cudaMemGetInfo(&mut free,&mut total)})?;
    Ok(json!({"device_free_bytes":free,"device_total_bytes":total}))
}

pub fn kick_background(jobs:Vec<Job>)->Result<()> {
    if !async_sha()?{return Ok(());}
    CACHE.with(|cell| {
        let mut held=cell.borrow_mut();
        if held.is_none(){*held=Some(Cache::from_env()?);}
        held.as_mut().unwrap().kick(jobs)
    })
}
pub fn enter_compute()->Result<Option<ComputePermit>> {
    CACHE.with(|cell|cell.borrow().as_ref().and_then(|c|c.background.as_ref()).map(|b|b.enter_compute()).transpose())
}
pub fn enter_request()->Result<Option<ComputePermit>> {
    CACHE.with(|cell| {
        let mut held=cell.borrow_mut();
        // Input-first startup pauses optional work before the first preload.
        // Create CPU bookkeeping only; no pinned allocation, IO or CUDA call.
        if held.is_none() && enabled()? && async_sha()? {*held=Some(Cache::from_env()?);}
        held.as_ref().and_then(|c|c.background.as_ref()).map(|b|b.enter_request()).transpose()
    })
}

/// A miss is a decision, not a fill or wait operation. Entries never evict.
pub fn can_upload(jobs:&[Job])->Result<bool> {
    if !enabled()? {return Ok(false);}
    if !deferred()? {return Ok(true);}
    CACHE.with(|cell| {
        let mut held=cell.borrow_mut();
        held.as_mut().map_or(Ok(false),|cache|cache.contains(jobs))
    })
}
/// Called only after a complete result (or explicitly by an idle benchmark).
pub fn allow_background()->Result<()> {
    CACHE.with(|cell| {
        let held=cell.borrow();
        if let Some(background)=held.as_ref().and_then(|c|c.background.as_ref()) {background.allow_idle();}
        Ok(())
    })
}

/// The caller must retain this owner until its stream is synchronized.
pub struct QueuedSources { _views:Vec<Entry> }
pub unsafe fn enqueue_ready(jobs:&[Job],destinations:&[*mut c_void],stream:*mut c_void)->Result<QueuedSources> {
    if jobs.is_empty() || jobs.len()!=destinations.len() || stream.is_null()
        || destinations.iter().any(|p|p.is_null()) || !enabled()? {
        return Err("invalid verified pinned enqueue".into());
    }
    let config=Config::from_env()?.ok_or("queued pinned upload requires bounded pipeline")?;
    CACHE.with(|cell|{
        let mut held=cell.borrow_mut();let cache=held.as_mut().ok_or("queued pinned cache missing")?;
        if !cache.contains(jobs)? {return Err("queued pinned cache incomplete; foreground fill forbidden".into());}
        let keys=jobs.iter().map(Key::new).collect::<Result<Vec<_>>>()?;
        let views=keys.iter().map(|k|cache.entries[k].clone()).collect::<Vec<_>>();
        let mut api_seconds=0.0;
        let result=(||->Result<()>{
            for (i,view) in views.iter().enumerate(){
                for offset in (0..view.bytes).step_by(config.chunk_bytes){
                    let bytes=config.chunk_bytes.min(view.bytes-offset);let began=Instant::now();
                    cuda(unsafe{cudaMemcpyAsync(destinations[i].cast::<u8>().add(offset).cast(),
                        view.pointer(offset,bytes)?.cast(),bytes,1,stream)})?;
                    api_seconds+=began.elapsed().as_secs_f64();
                }
            }
            Ok(())
        })();
        if result.is_err(){let drained=cuda(unsafe{cudaStreamSynchronize(stream)});result?;drained?;}
        let bytes=jobs.iter().map(|j|j.bytes).sum::<usize>();
        cache.counters.loads+=1;cache.counters.cache_hit_files+=jobs.len();
        cache.counters.cache_hit_upload_bytes+=bytes;cache.counters.upload_bytes+=bytes;
        cache.counters.h2d_api_seconds+=api_seconds;
        // upload_and_wait_seconds measures only the existing blocking API.
        // The layered owner reports its separate enqueue/compute wall span.
        Ok(QueuedSources{_views:views})
    })
}

/// Immutable DMA views can move to a loader thread without using that thread's
/// thread-local cache. Only integer destination addresses cross the boundary.
pub struct PreparedSources { views:Vec<Entry>, destinations:Vec<usize>,chunk_bytes:usize }
pub fn prepare_ready(jobs:&[Job],destinations:&[*mut c_void])->Result<PreparedSources>{
    if jobs.is_empty() || jobs.len()!=destinations.len() || destinations.iter().any(|p|p.is_null()) || !enabled()?{
        return Err("invalid prepared pinned upload".into());
    }
    let config=Config::from_env()?.ok_or("prepared pinned upload requires pipeline")?;
    CACHE.with(|cell|{
        let mut held=cell.borrow_mut();let cache=held.as_mut().ok_or("prepared cache missing")?;
        if !cache.contains(jobs)?{return Err("prepared cache incomplete; foreground fill forbidden".into());}
        let keys=jobs.iter().map(Key::new).collect::<Result<Vec<_>>>()?;
        Ok(PreparedSources{views:keys.iter().map(|k|cache.entries[k].clone()).collect(),
            destinations:destinations.iter().map(|p|*p as usize).collect(),chunk_bytes:config.chunk_bytes})
    })
}
impl PreparedSources {
    pub fn bytes(&self)->usize{self.views.iter().map(|v|v.bytes).sum()}
    pub fn files(&self)->usize{self.views.len()}
    /// Caller retains this object and the destination owner through the drain.
    pub unsafe fn enqueue(&self,stream:*mut c_void)->Result<f64>{
        if stream.is_null(){return Err("prepared stream is null".into());}
        let mut seconds=0.0;
        let result=(||->Result<()>{
            for (index,view) in self.views.iter().enumerate(){
                for offset in (0..view.bytes).step_by(self.chunk_bytes){
                    let bytes=self.chunk_bytes.min(view.bytes-offset);let began=Instant::now();
                    cuda(unsafe{cudaMemcpyAsync((self.destinations[index] as *mut u8).add(offset).cast(),
                        view.pointer(offset,bytes)?.cast(),bytes,1,stream)})?;
                    seconds+=began.elapsed().as_secs_f64();
                }
            }Ok(())
        })();
        if result.is_err(){let drained=cuda(unsafe{cudaStreamSynchronize(stream)});result?;drained?;}
        Ok(seconds)
    }
}
pub fn record_prepared_upload(files:usize,bytes:usize,api_seconds:f64)->Result<()> {
    CACHE.with(|cell|{
        let mut held=cell.borrow_mut();let cache=held.as_mut().ok_or("completed prepared cache missing")?;
        cache.counters.loads+=1;cache.counters.cache_hit_files+=files;cache.counters.cache_hit_upload_bytes+=bytes;
        cache.counters.upload_bytes+=bytes;cache.counters.h2d_api_seconds+=api_seconds;Ok(())
    })
}
