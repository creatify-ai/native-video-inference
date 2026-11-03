//! CPU-only preparation of immutable pinned originals; no GPU weights or merge.
//! Publication crosses a channel only after the whole tensor's size/guard pass.
use super::{Allocation,Entry,Key};
use crate::{weight_loader::{Job,light},Result};
use sha2::{Digest,Sha256};
use serde_json::{json,Value};
use std::{collections::HashSet,fs::File,io::{Read,Seek,SeekFrom},sync::{Arc,Mutex,mpsc,atomic::{AtomicBool,Ordering}},thread,time::{Duration,Instant}};

#[derive(Default)]
struct Stats {submitted:usize,checked:usize,passed:usize,checked_bytes:usize,prefill_files:usize,prefill_bytes:usize,
    prefill_pin_seconds:f64,prefill_read_seconds:f64,audit_seconds:f64,warnings:Vec<String>}
type AuditItem=(Key,Entry);
/// Foreground owns request execution. Optional IO resumes on every return path.
pub struct ComputePermit {active:Arc<AtomicBool>}
impl Drop for ComputePermit {fn drop(&mut self){self.active.store(false,Ordering::Release);}}
pub(super) struct Background {
    closed:Arc<AtomicBool>,gate:Arc<AtomicBool>,prefill_stop:Arc<AtomicBool>,
    compute_active:Arc<AtomicBool>,request_active:Arc<AtomicBool>,cuda_gate:Arc<Mutex<()>>,
    stats:Arc<Mutex<Stats>>,seen:Arc<Mutex<HashSet<Key>>>,
    audit_send:Option<mpsc::Sender<AuditItem>>,audit_thread:Option<thread::JoinHandle<()>>,
    prefill_thread:Option<thread::JoinHandle<()>>,ready:Option<mpsc::Receiver<AuditItem>>,
    pub reserved:usize,idle_enabled:Arc<AtomicBool>,deferred:bool,
}
fn lower_priority(){
    #[cfg(target_os="linux")]
    unsafe {unsafe extern "C" {fn gettid()->i32;fn setpriority(which:i32,who:u32,priority:i32)->i32;}
        let _=setpriority(0,gettid() as u32,10);}
}
fn warning(stats:&Mutex<Stats>,message:String){
    eprintln!("{}",json!({"event":"model.sha256_audit","state":"warning","reason":message,"action":"restart or reload this model"}));
    stats.lock().unwrap().warnings.push(message);
}
fn submit(send:&mpsc::Sender<AuditItem>,seen:&Mutex<HashSet<Key>>,stats:&Mutex<Stats>,key:Key,entry:Entry)->Result<()> {
    if !seen.lock().unwrap().insert(key.clone()){return Ok(());}
    stats.lock().unwrap().submitted+=1;
    send.send((key,entry)).map_err(|_|"background SHA worker exited".into())
}
fn wait_for_idle(active:&AtomicBool,request:&AtomicBool,cancel:&AtomicBool,closed:&AtomicBool)->bool {
    while active.load(Ordering::Acquire)||request.load(Ordering::Acquire) {
        if cancel.load(Ordering::Acquire)||closed.load(Ordering::Acquire){return false;}
        thread::sleep(Duration::from_millis(2));
    }
    !cancel.load(Ordering::Acquire)&&!closed.load(Ordering::Acquire)
}
impl Background {
    pub fn new()->Result<Self> {
        let deferred=super::deferred()?;
        let idle_enabled=Arc::new(AtomicBool::new(!deferred));
        let compute_active=Arc::new(AtomicBool::new(false));let request_active=Arc::new(AtomicBool::new(false));
        let (audit_idle,audit_compute,audit_request)=(idle_enabled.clone(),compute_active.clone(),request_active.clone());
        let closed=Arc::new(AtomicBool::new(false));let gate=Arc::new(AtomicBool::new(false));
        let stats=Arc::new(Mutex::new(Stats::default()));let (send,receive)=mpsc::channel::<AuditItem>();
        let (stop,allowed,observed)=(closed.clone(),gate.clone(),stats.clone());
        let audit=thread::Builder::new().name("dit-full-sha".into()).spawn(move|| {
            lower_priority();
            while !stop.load(Ordering::Acquire) {
                let (key,entry)=match receive.recv_timeout(Duration::from_millis(25)) {
                    Ok(v)=>v,Err(mpsc::RecvTimeoutError::Timeout)=>continue,Err(_)=>break};
                let mut hash=Sha256::new();let begin=Instant::now();let mut cancelled=false;
                for offset in (0..entry.bytes).step_by(16<<20) {
                    while (!allowed.load(Ordering::Acquire) || (deferred && (!audit_idle.load(Ordering::Acquire)
                        || audit_compute.load(Ordering::Acquire) || audit_request.load(Ordering::Acquire)))) && !stop.load(Ordering::Acquire) {
                        thread::sleep(Duration::from_millis(2));
                    }
                    if stop.load(Ordering::Acquire){cancelled=true;break;}
                    let bytes=(16<<20).min(entry.bytes-offset);
                    // Entry ownership keeps these immutable loaded bytes alive
                    // even if the foreground releases a GPU model or exits.
                    let pointer=match entry.pointer(offset,bytes){Ok(p)=>p,Err(e)=>{warning(&observed,e.to_string());cancelled=true;break;}};
                    hash.update(unsafe{std::slice::from_raw_parts(pointer,bytes)});
                }
                if cancelled{break;}
                let actual=format!("{:x}",hash.finalize());
                if actual!=key.sha256 {warning(&observed,format!("pinned payload SHA mismatch {} offset {}: expected {} actual {}",key.path.display(),key.file_offset,key.sha256,actual));}
                let mut state=observed.lock().unwrap();state.checked+=1;state.passed+=usize::from(actual==key.sha256);state.checked_bytes+=entry.bytes;
                state.audit_seconds+=begin.elapsed().as_secs_f64();
                if state.checked==state.submitted {
                    eprintln!("{}",json!({"event":"model.sha256_audit","state":if state.warnings.is_empty(){"PASS"}else{"warning"},
                        "checked_files":state.checked,"checked_bytes":state.checked_bytes,"scope":"all submitted immutable pinned payloads","ready_barrier":false}));
                }
            }
        })?;
        Ok(Self{closed,gate,prefill_stop:Arc::new(AtomicBool::new(false)),
            compute_active,request_active,cuda_gate:Arc::new(Mutex::new(())),stats,seen:Arc::new(Mutex::new(HashSet::new())),
            audit_send:Some(send),audit_thread:Some(audit),prefill_thread:None,ready:None,reserved:0,idle_enabled,deferred})
    }
    pub fn allow_idle(&self){self.idle_enabled.store(true,Ordering::Release);}
    pub fn warn(&self,message:String){warning(&self.stats,message);}
    pub fn healthy(&self)->bool{self.stats.lock().unwrap().warnings.is_empty()}
    pub fn submit(&self,key:Key,entry:Entry)->Result<()> {
        submit(self.audit_send.as_ref().ok_or("background SHA closed")?,&self.seen,&self.stats,key,entry)
    }
    pub fn enter_compute(&self)->Result<ComputePermit> {
        // Wait only for a CUDA pin/free API already in flight, never for IO,
        // a whole tensor, the optional model, or its full SHA.
        let _cuda=self.cuda_gate.lock().map_err(|_|"background CUDA gate poisoned")?;
        if self.compute_active.swap(true,Ordering::AcqRel){return Err("nested background compute permit".into());}
        Ok(ComputePermit{active:self.compute_active.clone()})
    }
    pub fn enter_request(&self)->Result<ComputePermit> {
        let _cuda=self.cuda_gate.lock().map_err(|_|"background CUDA gate poisoned")?;
        if self.request_active.swap(true,Ordering::AcqRel){return Err("nested background request permit".into());}
        Ok(ComputePermit{active:self.request_active.clone()})
    }
    pub fn harvest(&mut self)->Vec<AuditItem> {
        if self.prefill_thread.as_ref().is_some_and(|t|t.is_finished()) {
            if self.prefill_thread.take().unwrap().join().is_err(){warning(&self.stats,"background pinned reader panicked".into());}
            self.reserved=0;
        }
        self.ready.as_ref().map(|r|r.try_iter().collect()).unwrap_or_default()
    }
    /// Selection occurs only between complete requests. Stop at the next read
    /// chunk, retain already published payloads, and give needed IO priority.
    pub fn pause_prefill(&mut self)->Vec<AuditItem> {
        self.prefill_stop.store(true,Ordering::Release);
        self.compute_active.store(false,Ordering::Release);
        self.request_active.store(false,Ordering::Release);
        if let Some(thread)=self.prefill_thread.take(){
            if thread.join().is_err(){warning(&self.stats,"background pinned reader panicked".into());}
        }
        self.reserved=0;self.gate.store(true,Ordering::Release);self.harvest()
    }
    pub fn foreground(&self){self.gate.store(false,Ordering::Release);}
    pub fn start_prefill(&mut self,jobs:Vec<Job>,reserved:usize)->Result<()> {
        if self.prefill_thread.is_some(){return Ok(());}
        self.gate.store(jobs.is_empty(),Ordering::Release);
        if jobs.is_empty(){return Ok(());}
        self.prefill_stop=Arc::new(AtomicBool::new(false));self.reserved=reserved;
        let (send,receive)=mpsc::channel();self.ready=Some(receive);
        let (cancel,closed,gate,stats,seen,audit)=(self.prefill_stop.clone(),self.closed.clone(),self.gate.clone(),
            self.stats.clone(),self.seen.clone(),self.audit_send.as_ref().unwrap().clone());
        let (active,request,cuda_gate)=(self.compute_active.clone(),self.request_active.clone(),self.cuda_gate.clone());
        let idle=self.idle_enabled.clone();
        self.prefill_thread=Some(thread::Builder::new().name("dit-ram-prefill".into()).spawn(move|| {
            lower_priority();
            for job in jobs {
                while !idle.load(Ordering::Acquire)&&!cancel.load(Ordering::Acquire)&&!closed.load(Ordering::Acquire){thread::sleep(Duration::from_millis(2));}
                if !wait_for_idle(&active,&request,&cancel,&closed){break;}
                // Retain an outer owner until safe cleanup, including error and
                // cancellation paths. Allocation::drop calls CUDA APIs.
                let mut private_allocation:Option<Arc<Allocation>>=None;
                let result=(||->Result<Option<AuditItem>> {
                    let key=Key::new(&job)?;
                    let expected=light::expected(Some(&key.sha256),key.bytes)?.ok_or("background fill requires trusted light SHA")?;
                    let mut guard=light::ReaderGuard::new(expected);
                    let mut file=File::open(&job.path)?;let before=file.metadata()?;
                    let file_bytes=job.file_bytes.unwrap_or(job.bytes as u64);
                    if before.len()!=file_bytes||job.file_offset.checked_add(job.bytes as u64).is_none_or(|n|n>file_bytes){return Err("background file extent changed".into());}
                    file.seek(SeekFrom::Start(job.file_offset))?;
                    let extent=job.bytes.checked_add(255).ok_or("prefill extent overflow")?&!255;
                    let (allocation,pin_seconds)=loop {
                        if !wait_for_idle(&active,&request,&cancel,&closed){return Ok(None);}
                        let _cuda=cuda_gate.lock().map_err(|_|"background CUDA gate poisoned")?;
                        if active.load(Ordering::Acquire)||request.load(Ordering::Acquire){continue;}
                        let began=Instant::now();let allocation=Arc::new(Allocation::new(extent)?);
                        break (allocation,began.elapsed().as_secs_f64());
                    };
                    private_allocation=Some(allocation.clone());
                    let entry=Entry{allocation,offset:0,bytes:job.bytes,finite_bf16_verified:false};let read_at=Instant::now();
                    for offset in (0..job.bytes).step_by(16<<20) {
                        if !wait_for_idle(&active,&request,&cancel,&closed){return Ok(None);}
                        let size=(16<<20).min(job.bytes-offset);
                        let buffer=unsafe{std::slice::from_raw_parts_mut(entry.pointer(offset,size)?,size)};
                        file.read_exact(buffer)?;guard.observe(offset,buffer);
                    }
                    guard.finish()?;
                    if file.metadata()?.len()!=file_bytes {return Err("background file size changed during read".into());}
                    let mut state=stats.lock().unwrap();state.prefill_files+=1;state.prefill_bytes+=job.bytes;
                    state.prefill_pin_seconds+=pin_seconds;state.prefill_read_seconds+=read_at.elapsed().as_secs_f64();
                    Ok(Some((key,entry)))
                })();
                let stop=match result {
                    Ok(Some((key,entry)))=>{
                        if let Err(error)=submit(&audit,&seen,&stats,key.clone(),entry.clone()){warning(&stats,error.to_string());true}
                        else{send.send((key,entry)).is_err()}
                    },Ok(None)=>true,Err(error)=>{warning(&stats,format!("RAM prefill {}: {error}",job.path.display()));true},
                };
                // A successful publication has other owners, so this is only a
                // reference decrement. Failed/private tensors must free at idle.
                if private_allocation.as_ref().is_some_and(|a|Arc::strong_count(a)==1) {
                    loop {
                        while active.load(Ordering::Acquire)||request.load(Ordering::Acquire){thread::sleep(Duration::from_millis(2));}
                        let _cuda=cuda_gate.lock().unwrap();
                        if active.load(Ordering::Acquire)||request.load(Ordering::Acquire){continue;}
                        drop(private_allocation.take());break;
                    }
                }else{drop(private_allocation.take());}
                if stop{break;}
            }
            gate.store(true,Ordering::Release);
        })?);
        Ok(())
    }
    pub fn report(&self)->Value {
        let state=self.stats.lock().unwrap();let filling=self.prefill_thread.as_ref().is_some_and(|t|!t.is_finished());
        json!({"state":if !state.warnings.is_empty(){"warning"}else if filling{"filling"}else if state.checked<state.submitted{"auditing"}else{"PASS"},
            "submitted_files":state.submitted,"checked_files":state.checked,"passed_files":state.passed,"checked_bytes":state.checked_bytes,
            "prefill_files":state.prefill_files,"prefill_bytes":state.prefill_bytes,"prefill_pin_seconds":state.prefill_pin_seconds,
            "prefill_read_seconds":state.prefill_read_seconds,"audit_worker_seconds":state.audit_seconds,"reserved_bytes":self.reserved,
            "warnings":state.warnings,"ready_barrier":false,"workers":2,"gpu_uploads":0,
            "deferred_originals":self.deferred,"idle_enabled":self.idle_enabled.load(Ordering::Acquire),
            "paused_for_dit_request":self.compute_active.load(Ordering::Acquire),
            "paused_for_full_request":self.request_active.load(Ordering::Acquire)})
    }
}
impl Drop for Background {
    fn drop(&mut self){
        self.closed.store(true,Ordering::Release);self.gate.store(true,Ordering::Release);
        let _=self.pause_prefill();self.audit_send.take();
        if let Some(thread)=self.audit_thread.take(){let _=thread.join();}
    }
}
