//! H3 loading policy over the model-independent bounded reader/upload mechanisms.
use native_core_infra::weight_loader as infra;
#[path="light_sha.rs"]
pub mod light;
pub use infra::{Job,ReadChunk,flag,tensor_allocation_layout,validate_tensor_allocations,tensor_allocation_locations};
pub type Result<T>=std::result::Result<T,String>;
#[derive(Clone, Copy, Debug)]
pub struct Config {
    pub workers: usize,
    pub chunk_bytes: usize,
    pub pinned_slots: usize,
    pub budget_bytes: usize,
}
impl Config {
    pub fn from_env() -> Result<Option<Self>> {
        match std::env::var("H3_MODEL_LOAD_PIPELINE").as_deref() {
            Err(_)|Ok("0")=>return Ok(None), Ok("1")=>{},
            _=>return Err("H3_MODEL_LOAD_PIPELINE must be 0 or 1".into()),
        }
        fn setting(name:&str, default:usize, low:usize, high:usize)->Result<usize> {
            let n=match std::env::var(name) {Ok(s)=>s.parse().map_err(|_|format!("invalid {name}"))?,Err(_)=>default};
            if !(low..=high).contains(&n) {return Err(format!("{name} must be {low}..={high}"));} Ok(n)
        }
        let wide_readers=flag("H3_MODEL_LOAD_WIDE_READERS")?;
        let config=Self {
            workers:setting("H3_MODEL_LOAD_WORKERS",4,1,if wide_readers {64}else{32})?,
            chunk_bytes:setting("H3_MODEL_LOAD_CHUNK_MIB",16,1,128)?<<20,
            pinned_slots:setting("H3_MODEL_LOAD_PINNED_SLOTS",4,1,16)?,
            budget_bytes:setting("H3_MODEL_LOAD_BUDGET_MIB",512,16,
                if wide_readers {8192}
                else if flag("H3_MODEL_LOAD_LARGE_BUDGET")? {4096}else{2048})?<<20,
        };
        config.validate()?; Ok(Some(config))
    }
    pub fn validate(self)->Result<()> {
        if self.workers==0 || self.chunk_bytes==0 || self.pinned_slots==0 {return Err("zero model load capacity".into());}
        let total=self.workers.checked_mul(2).and_then(|n|n.checked_add(self.pinned_slots))
            .and_then(|n|n.checked_mul(self.chunk_bytes)).ok_or("model load budget overflow")?;
        if total>self.budget_bytes {return Err(format!("model loader needs {total} bytes, budget is {}",self.budget_bytes));}
        Ok(())
    }
}

impl From<Config> for infra::Config {
    fn from(c:Config)->Self{Self{workers:c.workers,chunk_bytes:c.chunk_bytes,pinned_slots:c.pinned_slots,budget_bytes:c.budget_bytes}}
}
fn guarded_jobs(mut jobs:Vec<Job>,guarded:bool)->Result<Vec<Job>> {
    if guarded {for job in &mut jobs {
        let expected=light::expected(job.sha256.as_deref(),job.bytes)?.ok_or("trusted light SHA guard missing")?;
        job.guard=Some(native_core_infra::integrity::Expected{bytes:expected.bytes,head_sha256:expected.head_sha256,tail_sha256:expected.tail_sha256});
    }}
    Ok(jobs)
}
pub struct ReaderPool(infra::ReaderPool);
impl ReaderPool {
    pub fn start(jobs:Vec<Job>,config:Config)->Result<Self> {
        crate::startup_trace::initialize();
        let guarded=light::async_enabled()?;
        Self::scheduled(jobs,config,flag("H3_MODEL_LOAD_DIRECT_PINNED")?,!guarded,guarded)
    }
    fn scheduled(jobs:Vec<Job>,config:Config,direct:bool,verify_sha:bool,guarded:bool)->Result<Self> {
        crate::startup_trace::initialize();
        Ok(Self(infra::ReaderPool::start_scheduled_mode(guarded_jobs(jobs,guarded)?,config.into(),direct,verify_sha,guarded,
            flag("H3_MODEL_LOAD_BALANCED_READERS")?,flag("H3_MODEL_LOAD_MMAP")?)?))
    }
    pub(crate) fn start_options(jobs:Vec<Job>,config:Config,direct:bool,verify_sha:bool)->Result<Self>{Self::scheduled(jobs,config,direct,verify_sha,false)}
    pub(crate) fn start_guarded(jobs:Vec<Job>,config:Config)->Result<Self>{Self::scheduled(jobs,config,false,false,true)}
    pub fn start_read_ahead(jobs:Vec<Job>,config:Config)->Result<Self> {
        crate::startup_trace::initialize();
        let guarded=light::async_enabled()?;
        if flag("H3_MODEL_LOAD_MMAP")?{return Err("encoder read-ahead requires owned reader buffers".into());}
        Ok(Self(infra::ReaderPool::start_read_ahead_options(guarded_jobs(jobs,guarded)?,config.into(),
            flag("H3_MODEL_LOAD_DIRECT_PINNED")?,!guarded,guarded)?))
    }
    pub fn consume_all(&mut self,upload:impl FnMut(usize,usize,&[u8])->Result<()>)->Result<()> {self.0.consume_all(upload)}
    #[cfg(feature="cuda")]
    pub unsafe fn consume_upload(&mut self,upload:&mut device::Upload,destination:impl FnMut(usize,usize)->*mut std::ffi::c_void)->Result<()> {
        unsafe{self.0.consume_upload(&mut upload.0,destination)}
    }
}
#[cfg(feature="cuda")]
pub mod device {
    use super::{Config,Result,flag};
    use std::ffi::c_void;
    pub struct Upload(pub(super) native_core_infra::weight_loader::device::Upload);
    impl Upload {
        pub fn new(stream:*mut c_void,config:Config)->Result<Self> {
            crate::startup_trace::initialize();
            Ok(Self(native_core_infra::weight_loader::device::Upload::with_policy(stream,config.into(),
                flag("H3_MODEL_LOAD_DIRECT_PINNED")?,flag("H3_MODEL_LOAD_MMAP")?,
                std::env::var_os("H3_MODEL_LOAD_PREFETCH_STOP_FILE").map(Into::into))?))
        }
        pub(crate) fn new_options(stream:*mut c_void,config:Config,direct:bool)->Result<Self> {
            Ok(Self(native_core_infra::weight_loader::device::Upload::new_options(stream,config.into(),direct,
                std::env::var_os("H3_MODEL_LOAD_PREFETCH_STOP_FILE").map(Into::into))?))
        }
        pub unsafe fn copy(&mut self,dst:*mut c_void,data:&[u8])->Result<()> {unsafe{self.0.copy(dst,data)}}
        pub fn finish(&self)->Result<()> {self.0.finish()}
        pub unsafe fn verify(&self,src:*const c_void,bytes:usize,expected:&str)->Result<()> {
            unsafe{self.0.verify(src,bytes,expected)}
        }
    }
}
