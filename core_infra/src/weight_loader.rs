//! Bounded, full-integrity tensor reads with caller-supplied integrity policy.
//! Each worker owns two reusable chunks. Upload callbacks must copy the bytes
//! before returning; only then is the CPU chunk returned to its worker.
use sha2::{Digest, Sha256};
use std::{fs::File, io::{Read,Seek,SeekFrom}, path::{Component, PathBuf}, sync::{mpsc, Arc, atomic::{AtomicUsize,Ordering}}, thread};
use crate::integrity as light;

pub type Result<T> = std::result::Result<T, String>;

/// Complete retained-read bound for a staged schedule, including per-chunk
/// alignment and each stage's pinned-copy ring. No model or CUDA dependency.
pub fn read_ahead_bound(sizes:&[usize],config:Config,direct:bool,stages:usize)->Result<usize> {
    if config.chunk_bytes==0 || stages==0 {return Err("invalid read-ahead schedule".into());}
    let aligned=|n:usize|->Result<usize>{
        if direct {n.checked_add(4095).map(|v|v/4096*4096).ok_or("read-ahead alignment overflow".into())}
        else {Ok(n)}
    };
    let mut bound=stages.checked_mul(config.pinned_slots).and_then(|n|n.checked_mul(config.chunk_bytes))
        .ok_or("read-ahead ring bound overflow")?;
    for &bytes in sizes {
        let full=bytes/config.chunk_bytes;
        let payload=full.checked_mul(aligned(config.chunk_bytes)?)
            .and_then(|n|n.checked_add(aligned(bytes%config.chunk_bytes).ok()?)).ok_or("read-ahead payload bound overflow")?;
        bound=bound.checked_add(payload).ok_or("read-ahead schedule bound overflow")?;
    }
    Ok(bound)
}

/// Aligned immutable storage for copying already-running reusable readers.
/// Their existing bounded buffers are admitted separately, once per cohort.
pub(crate) fn retained_copy_bound(sizes:&[usize],chunk:usize)->Result<usize> {
    if chunk==0{return Err("retained copy needs a nonzero chunk".into());}
    let aligned=|n:usize|n.checked_add(4095).map(|v|v/4096*4096).ok_or("retained alignment overflow");
    let mut bytes=0usize;
    for &size in sizes {
        let full=(size/chunk).checked_mul(aligned(chunk)?).ok_or("retained payload overflow")?;
        let tail=aligned(size%chunk)?;
        bytes=bytes.checked_add(full).and_then(|n|n.checked_add(tail)).ok_or("retained payload overflow")?;
    }
    Ok(bytes)
}

pub fn flag(name:&str)->Result<bool> {
    match std::env::var(name).as_deref() {
        Err(_)|Ok("0")=>Ok(false),Ok("1")=>Ok(true),_=>Err(format!("{name} must be 0 or 1")),
    }
}

#[derive(Clone,Copy,PartialEq,Eq)]
enum FileIo {Buffered,Auto,Direct}
/// Source-file read policy, independent of CUDA direct uploads and integrity.
#[derive(Clone,Copy,Debug,PartialEq,Eq)]
pub enum FileReadPolicy {Buffered,Auto,Direct}
impl FileReadPolicy {
    pub fn as_str(self)->&'static str {match self {Self::Buffered=>"buffered",Self::Auto=>"auto",Self::Direct=>"direct"}}
    pub fn effective(self)->Result<Self>{Ok(match FileIo::from_env(self.into())?{FileIo::Buffered=>Self::Buffered,FileIo::Auto=>Self::Auto,FileIo::Direct=>Self::Direct})}
}
/// How independent components share one bounded reader cohort. Models may
/// prefer early progress per component or equal issued payload bytes. FileOffset
/// keeps byte fairness across groups and reads each packed source in offset order.
#[derive(Clone,Copy,Debug,PartialEq,Eq)]
pub enum ReadSchedule {ByteFair,RoundRobin,FileOffset}
impl ReadSchedule {
    pub fn as_str(self)->&'static str {match self{Self::ByteFair=>"byte_fair",Self::RoundRobin=>"round_robin",Self::FileOffset=>"file_offset"}}
    fn parse(value:&str)->Result<Self>{match value{"byte_fair"=>Ok(Self::ByteFair),"round_robin"=>Ok(Self::RoundRobin),"file_offset"=>Ok(Self::FileOffset),
        _=>Err("NATIVE_LOAD_READ_SCHEDULE must be byte_fair, round_robin or file_offset".into())}}
    pub(crate)fn from_env(default:Self)->Result<Self>{match std::env::var("NATIVE_LOAD_READ_SCHEDULE"){
        Ok(value)=>Self::parse(&value),Err(std::env::VarError::NotPresent)=>Ok(default),Err(_)=>Err("invalid reader schedule".into())}}
}
impl From<FileReadPolicy> for FileIo {
    fn from(policy:FileReadPolicy)->Self {match policy {FileReadPolicy::Buffered=>Self::Buffered,FileReadPolicy::Auto=>Self::Auto,FileReadPolicy::Direct=>Self::Direct}}
}
pub(crate) fn snapshot_file_policy()->Result<bool> {Ok(FileIo::from_env(FileIo::Buffered)?==FileIo::Buffered)}
impl FileIo {
    fn from_env(default:Self)->Result<Self>{
        // Compatibility with frozen diagnostics; models use the shared policy.
        if flag("NATIVE_LOAD_COMPONENT_FILE_DIRECT")?{return Ok(Self::Direct);}
        match std::env::var("NATIVE_LOAD_FILE_IO").as_deref(){
            Err(_)=>Ok(default),Ok("auto")=>Ok(Self::Auto),Ok("buffered")=>Ok(Self::Buffered),Ok("direct")=>Ok(Self::Direct),
            _=>Err("NATIVE_LOAD_FILE_IO must be auto, buffered or direct".into()),
        }
    }
    fn for_job(self,job:&Job,chunk:usize,mapped:bool)->Self{
        if self==Self::Auto && (!cfg!(target_os="linux") || job.file_bytes.is_none()
            || mapped || chunk<8192 || chunk%4096!=0){Self::Buffered}else{self}
    }
    fn for_pool<'a>(self,jobs:impl Iterator<Item=&'a Job>,config:Config,direct:bool)->Self{
        if self!=Self::Auto{return self;}
        let mut count=0usize;let mut packed=false;
        for job in jobs{count+=1;packed|=job.file_bytes.is_some();}
        if !cfg!(target_os="linux") || !direct || !packed || config.chunk_bytes<8192 || config.chunk_bytes%4096!=0{
            return Self::Buffered;
        }
        let count=count.min(config.workers);
        let required=count.checked_mul(2).and_then(|n|n.checked_mul(config.chunk_bytes+4096))
            .and_then(|n|n.checked_add(count*2*(config.chunk_bytes/4096+1)));
        if required.is_none_or(|n|n>config.budget_bytes){Self::Buffered}else{self}
    }
}
fn direct_unavailable(error:&std::io::Error)->bool{matches!(error.raw_os_error(),Some(1|22|38|95))}

/// CUDA allocations are at least 256-byte aligned. Preserve that guarantee
/// for each immutable tensor view in a single owned allocation.
pub fn tensor_allocation_layout(sizes:&[usize])->Result<(Vec<usize>,usize)> {
    let mut offsets=Vec::with_capacity(sizes.len());
    let mut bytes=0usize;
    for &size in sizes {
        if size==0{return Err("zero-sized model tensor allocation".into());}
        let offset=bytes.checked_add(255).ok_or("model allocation alignment overflow")? & !255;
        bytes=offset.checked_add(size).ok_or("model allocation size overflow")?;
        offsets.push(offset);
    }
    Ok((offsets,bytes))
}

/// Verify view ownership without assuming one CUDA allocation per tensor.
pub fn validate_tensor_allocations(owned:&[(usize,usize)],views:&[(usize,usize)])->Result<()> {
    tensor_allocation_locations(owned,views).map(|_|())
}

/// Map each immutable view to its owned allocation and byte offset. Offload
/// frees/restores owners once, then reconstructs every view at the same offset.
pub fn tensor_allocation_locations(owned:&[(usize,usize)],views:&[(usize,usize)])->Result<Vec<(usize,usize)>> {
    let ranges=owned.iter().map(|&(start,bytes)|start.checked_add(bytes).map(|end|(start,end))
        .ok_or_else(||"owned model allocation range overflow".to_string())).collect::<Result<Vec<_>>>()?;
    let mut locations=Vec::with_capacity(views.len());
    for &(start,bytes) in views {
        let end=start.checked_add(bytes).ok_or("model tensor range overflow")?;
        if bytes==0{return Err("model tensor view exceeds its owned allocation".into());}
        let location=ranges.iter().enumerate().find_map(|(index,&(left,right))|
            (left<=start && end<=right).then(||(index,start-left)))
            .ok_or("model tensor view exceeds its owned allocation")?;
        locations.push(location);
    }
    Ok(locations)
}

// A lease is moved between reader and uploader; it is never cloned. CUDA
// registrations keep the allocation alive without accessing its contents.
struct AlignedStorage {pointer:*mut u8,layout:std::alloc::Layout,anonymous:bool}
unsafe impl Send for AlignedStorage {}
unsafe impl Sync for AlignedStorage {}
impl Drop for AlignedStorage {
    fn drop(&mut self){
        if self.anonymous {
            #[cfg(target_os="linux")]
            unsafe{mapping::munmap(self.pointer.cast(),self.layout.size());}
        }else{unsafe{std::alloc::dealloc(self.pointer,self.layout);}}
    }
}
/// A read-only file mapping stays alive until its upload event has completed
/// and CUDA has unregistered the pages. Config bounds the leases per reader.
struct MappedStorage {pointer:*mut u8,bytes:usize,prefix:usize}
unsafe impl Send for MappedStorage {}
#[cfg(target_os="linux")]
mod mapping {
    use super::{MappedStorage,Result};
    use std::{ffi::{c_int,c_void},os::fd::AsRawFd};
    unsafe extern "C" {
        fn mmap(address:*mut c_void,bytes:usize,protection:c_int,flags:c_int,fd:c_int,offset:i64)->*mut c_void;
        pub(super) fn munmap(address:*mut c_void,bytes:usize)->c_int;
        fn getpagesize()->c_int;
        fn mincore(address:*mut c_void,bytes:usize,bitmap:*mut u8)->c_int;
    }
    pub(super) fn anonymous_zeroed(bytes:usize)->Result<*mut u8> {
        // MAP_PRIVATE | MAP_ANONYMOUS provides initialized zero bytes without
        // serially touching every page before CPU readers can start. There is
        // no uninitialized Rust slice, file mapping or payload copy here.
        let pointer=unsafe{mmap(std::ptr::null_mut(),bytes,3,2|0x20,-1,0)};
        if pointer as isize == -1{return Err(format!("anonymous reader mapping: {}",std::io::Error::last_os_error()));}
        Ok(pointer.cast())
    }
    pub(super) fn extent(bytes:usize)->Result<usize> {
        let page=unsafe{getpagesize()} as usize;
        if !page.is_power_of_two(){return Err("invalid host page size".into());}
        bytes.checked_add(page-1).and_then(|n|n.checked_add(page-1)).map(|n|n/page*page)
            .ok_or_else(||"mapped reader extent overflow".into())
    }
    pub(super) fn budget(config:super::Config,jobs:&[super::Job])->Result<usize> {
        let page=unsafe{getpagesize()} as usize;
        if !page.is_power_of_two(){return Err("invalid host page size".into());}
        let extent=if config.chunk_bytes%page==0 {
            let prefix=jobs.iter().map(|job|(job.file_offset%page as u64) as usize).max().unwrap_or(0);
            config.chunk_bytes.checked_add(prefix).and_then(|n|n.checked_add(page-1))
                .map(|n|n/page*page).ok_or("mapped reader extent overflow")?
        }else{extent(config.chunk_bytes)?};
        // Reserve the pinned-copy fallback even on devices that can upload
        // directly from read-only page-cache mappings. No file is pinned whole.
        config.workers.checked_mul(2).and_then(|n|n.checked_mul(extent))
            .and_then(|n|config.pinned_slots.checked_mul(config.chunk_bytes).and_then(|slots|n.checked_add(slots)))
            .ok_or_else(||"mapped reader budget overflow".into())
    }
    pub(super) fn read(file:&std::fs::File,offset:u64,bytes:usize)->Result<MappedStorage> {
        let page=unsafe{getpagesize()} as u64;
        if !page.is_power_of_two() || bytes==0 {return Err("invalid mapped model range".into());}
        let base=offset & !(page-1);
        let prefix=usize::try_from(offset-base).map_err(|_|"mapped model prefix overflow")?;
        let count=prefix.checked_add(bytes).and_then(|n|n.checked_add(page as usize-1))
            .map(|n|n/page as usize*page as usize).ok_or("mapped model extent overflow")?;
        let base=i64::try_from(base).map_err(|_|"mapped model offset overflow")?;
        // PROT_READ, MAP_PRIVATE | MAP_POPULATE: fault pages in the reader,
        // without copying them to an anonymous staging allocation.
        let pointer=unsafe{mmap(std::ptr::null_mut(),count,1,0x8002,file.as_raw_fd(),base)};
        if pointer as isize == -1 {return Err(format!("model mmap: {}",std::io::Error::last_os_error()));}
        Ok(MappedStorage{pointer:pointer.cast(),bytes:count,prefix})
    }
    /// Inspect existing page-cache residency without faulting or retaining a
    /// payload mapping. The per-reader bitmap is at most chunk_bytes / page.
    pub(super) fn resident(file:&std::fs::File,offset:u64,bytes:usize,bitmap:&mut Vec<u8>)->Result<bool>{
        let page=unsafe{getpagesize()} as usize;
        if !page.is_power_of_two() || bytes==0{return Err("invalid cache probe extent".into());}
        let base=offset & !(page as u64-1);let prefix=(offset-base) as usize;
        let count=prefix.checked_add(bytes).ok_or("cache probe overflow")?;
        let base=i64::try_from(base).map_err(|_|"cache probe offset overflow")?;
        let pointer=unsafe{mmap(std::ptr::null_mut(),count,1,2,file.as_raw_fd(),base)};
        if pointer as isize == -1{return Err(format!("cache probe mmap: {}",std::io::Error::last_os_error()));}
        let mapping=MappedStorage{pointer:pointer.cast(),bytes:count,prefix};
        bitmap.resize(count.div_ceil(page),0);
        let status=unsafe{mincore(mapping.pointer.cast(),count,bitmap.as_mut_ptr())};
        if status!=0{return Err(format!("cache probe mincore: {}",std::io::Error::last_os_error()));}
        Ok(bitmap.iter().all(|v|v&1!=0))
    }
}
impl Drop for MappedStorage {
    fn drop(&mut self){
        #[cfg(target_os="linux")]
        unsafe{mapping::munmap(self.pointer.cast(),self.bytes);}
    }
}
pub struct ReadChunk {ordinary:Vec<u8>,aligned:Option<Arc<AlignedStorage>>,mapped:Option<MappedStorage>,prefix:usize,length:usize,target:Option<(Arc<HostTarget>,usize)>}
/// Immutable ownership of a one-use reader buffer, independent of its CUDA
/// registration. Retaining these leases does not copy the checkpoint again.
pub struct RetainedChunks { chunks:Vec<Vec<(usize,Arc<AlignedStorage>,usize)>> }
impl RetainedChunks {
    pub fn bytes(&self)->usize {self.chunks.iter().flatten().map(|(_,_,n)|n).sum()}
    pub fn tensor_bytes(&self,job:usize)->Result<usize>{
        let mut end=0usize;
        for(offset,_,bytes)in self.chunks.get(job).ok_or("retained tensor index")?{
            if *offset!=end{return Err("retained tensor chunk order".into());}
            end=end.checked_add(*bytes).ok_or("retained tensor extent overflow")?;
        }Ok(end)
    }
    pub fn read(&self,job:usize)->Result<Vec<u8>> {
        let chunks=self.chunks.get(job).ok_or("retained tensor index")?;
        let mut out=Vec::new();
        for (offset,storage,bytes) in chunks {
            if *offset!=out.len(){return Err("retained tensor chunk order".into());}
            out.extend_from_slice(unsafe{std::slice::from_raw_parts(storage.pointer,*bytes)});
        }
        Ok(out)
    }
}
/// An exclusively initialized host destination, retained through all readers.
/// Callers publish it only after every tensor's guard has passed.
pub struct HostTarget {pointer:*mut u8,bytes:usize,_owner:Arc<dyn Send+Sync>}
unsafe impl Send for HostTarget {}
unsafe impl Sync for HostTarget {}
impl HostTarget {
    /// `pointer..pointer+bytes` must be writable storage owned by `owner`.
    /// No other code may access it while the reader pool initializes it.
    pub unsafe fn new<T:Send+Sync+'static>(owner:Arc<T>,pointer:*mut u8,bytes:usize)->Result<Self>{
        if pointer.is_null()||bytes==0||(pointer as usize).checked_add(bytes).is_none(){return Err("invalid host reader destination".into());}
        Ok(Self{pointer,bytes,_owner:owner})
    }
}
impl ReadChunk {
    fn reader(bytes:usize,direct:bool,file_io:FileIo)->Result<Self>{
        let extra=usize::from(direct && file_io!=FileIo::Buffered)*4096;
        let mut value=Self::new(bytes.checked_add(extra).ok_or("reader alignment headroom overflow")?,direct)?;
        value.resize(bytes,0);Ok(value)
    }
    fn new(bytes:usize,direct:bool)->Result<Self> {
        let size=bytes.checked_add(4095).ok_or("aligned chunk overflow")?/4096*4096;
        let layout=std::alloc::Layout::from_size_align(size,4096).map_err(|e|e.to_string())?;
        // Linux supplies initialized zero pages without a serial memset of
        // workers * 2 * chunk_bytes before any reader can start. This is the
        // framework default, rather than an optimization each model must opt in.
        let lazy=match std::env::var("NATIVE_CORE_READ_LAZY_ZEROED").as_deref() {
            Err(_)=>cfg!(target_os="linux"),Ok("0")=>false,Ok("1")=>true,
            _=>return Err("NATIVE_CORE_READ_LAZY_ZEROED must be 0 or 1".into()),
        };
        // Buffered readers need initialized bytes too, but need not touch all
        // workers' MiB-sized buffers serially before any reader can start.
        // Keep sub-page/non-page-aligned ordinary allocations exact-sized.
        if !direct && (!lazy || bytes<4096 || bytes%4096!=0) {
            return Ok(Self{ordinary:vec![0;bytes],aligned:None,mapped:None,prefix:0,length:bytes,target:None});
        }
        Self::aligned(bytes,layout,lazy)
    }
    fn aligned(bytes:usize,layout:std::alloc::Layout,lazy:bool)->Result<Self> {
        let anonymous=lazy && cfg!(target_os="linux");
        let pointer=if anonymous {
            #[cfg(target_os="linux")]
            {mapping::anonymous_zeroed(layout.size())?}
            #[cfg(not(target_os="linux"))]
            {unreachable!()}
        }else{unsafe{std::alloc::alloc_zeroed(layout)}};
        if pointer.is_null(){return Err("model reader allocation failed".into());}
        Ok(Self{ordinary:Vec::new(),aligned:Some(Arc::new(AlignedStorage{pointer,layout,anonymous})),mapped:None,prefix:0,length:bytes,target:None})
    }
    fn resize(&mut self,bytes:usize,_value:u8){
        self.prefix=0;
        if let Some(storage)=&self.aligned{assert!(bytes<=storage.layout.size());}
        else{self.ordinary.resize(bytes,0);}
        self.length=bytes;
    }
}
impl std::ops::Deref for ReadChunk {
    type Target=[u8];
    fn deref(&self)->&[u8]{
        if let Some((target,offset))=&self.target{return unsafe{std::slice::from_raw_parts(target.pointer.add(*offset),self.length)};}
        if let Some(storage)=&self.mapped{unsafe{std::slice::from_raw_parts(storage.pointer.add(storage.prefix),self.length)}}
        else if let Some(storage)=&self.aligned{unsafe{std::slice::from_raw_parts(storage.pointer.add(self.prefix),self.length)}}
        else{&self.ordinary}
    }
}
impl std::ops::DerefMut for ReadChunk {
    fn deref_mut(&mut self)->&mut[u8]{
        if let Some((target,offset))=&self.target{return unsafe{std::slice::from_raw_parts_mut(target.pointer.add(*offset),self.length)};}
        assert!(self.mapped.is_none(),"model mapping is read-only");
        if let Some(storage)=&self.aligned{unsafe{std::slice::from_raw_parts_mut(storage.pointer.add(self.prefix),self.length)}}
        else{&mut self.ordinary}
    }
}

#[derive(Clone, Debug)]
pub struct Job {
    pub path: PathBuf,
    pub bytes: usize,
    pub sha256: Option<String>,
    pub file_offset:u64,
    pub file_bytes:Option<u64>,
    pub finite_bf16:bool,
    pub guard:Option<light::Expected>,
}
impl Job {
    pub fn new(root: &std::path::Path, file: &str, bytes: usize, sha256: Option<String>) -> Result<Self> {
        if file.is_empty() || PathBuf::from(file).components().any(|p| !matches!(p, Component::Normal(_))) {
            return Err("model tensor path must stay under its root".into());
        }
        if sha256.as_ref().is_some_and(|s| s.len()!=64 || !s.bytes().all(|c|c.is_ascii_hexdigit() && !c.is_ascii_uppercase())) {
            return Err("invalid model SHA256".into());
        }
        Ok(Self {path:root.join(file),bytes,sha256,file_offset:0,file_bytes:None,finite_bf16:false,guard:None})
    }
    pub fn slice(root:&std::path::Path,file:&str,offset:u64,bytes:usize,sha256:String,file_bytes:u64)->Result<Self> {
        let mut job=Self::new(root,file,bytes,Some(sha256))?;
        if bytes==0 || offset.checked_add(bytes as u64).is_none_or(|end|end>file_bytes) {
            return Err("packed tensor range exceeds its immutable file".into());
        }
        job.file_offset=offset;job.file_bytes=Some(file_bytes);Ok(job)
    }
}

#[derive(Clone, Copy, Debug,PartialEq,Eq,serde::Serialize)]
pub struct Config {
    pub workers: usize,
    pub chunk_bytes: usize,
    pub pinned_slots: usize,
    pub budget_bytes: usize,
}
impl Config {
    /// Shared serving reader budget; models and host archives use one policy.
    pub fn from_env()->Result<Self>{
        fn number(name:&str,default:usize,low:usize,high:usize)->Result<usize>{
            let n=match std::env::var(name){Ok(s)=>s.parse().map_err(|_|format!("invalid {name}"))?,Err(_)=>default};
            if !(low..=high).contains(&n){return Err(format!("{name} must be {low}..={high}"));}Ok(n)
        }
        let workers=number("NATIVE_LOAD_WORKERS",24,1,64)?;
        let chunk_bytes=number("NATIVE_LOAD_CHUNK_MIB",32,1,128)?<<20;
        let pinned_slots=number("NATIVE_LOAD_PINNED_SLOTS",4,1,16)?;
        let need=(workers*2+pinned_slots)*chunk_bytes;
        let budget_bytes=match std::env::var("NATIVE_LOAD_BUDGET_MIB"){
            Ok(s)=>s.parse::<usize>().map_err(|_|"invalid NATIVE_LOAD_BUDGET_MIB")?
                .checked_mul(1<<20).ok_or("reader budget overflow")?,Err(_)=>need,
        };
        let config=Self{workers,chunk_bytes,pinned_slots,budget_bytes};config.validate()?;Ok(config)
    }

    pub fn validate(self)->Result<()> {
        if self.workers==0 || self.chunk_bytes==0 || self.pinned_slots==0 {return Err("zero model load capacity".into());}
        let total=self.workers.checked_mul(2).and_then(|n|n.checked_add(self.pinned_slots))
            .and_then(|n|n.checked_mul(self.chunk_bytes)).ok_or("model load budget overflow")?;
        if total>self.budget_bytes {return Err(format!("model loader needs {total} bytes, budget is {}",self.budget_bytes));}
        Ok(())
    }
}

/// Split an explicit aggregate budget into independent component cohorts.
/// Admission is metadata-only; no buffer, reader or file is created here.
pub fn component_reader_configs(parent:Config,workers:&[usize],direct:bool,policy:FileReadPolicy)->Result<Vec<Config>>{
    parent.validate()?;
    if workers.is_empty()||workers.iter().any(|&n|!(1..=64).contains(&n))
        ||workers.iter().try_fold(0usize,|sum,&n|sum.checked_add(n)).is_none_or(|sum|sum>256){
        return Err("component reader quotas require 1..=64 each and at most 256 total".into());
    }
    let io=FileIo::from_env(policy.into())?;
    if io==FileIo::Direct&&(!direct||parent.chunk_bytes<8192||parent.chunk_bytes%4096!=0){
        return Err("direct component readers require aligned direct upload buffers".into());
    }
    let mut total=0usize;let mut configs=Vec::with_capacity(workers.len());
    for &workers in workers {
        let mut bytes=(2*workers+parent.pinned_slots).checked_mul(parent.chunk_bytes).ok_or("component reader bound overflow")?;
        if direct&&io!=FileIo::Buffered&&parent.chunk_bytes>=8192&&parent.chunk_bytes%4096==0{
            let aligned=parent.chunk_bytes.checked_add(8191).ok_or("component alignment overflow")?/4096*4096;
            let bound=(2*workers).checked_mul(aligned).and_then(|n|n.checked_add(2*workers*(parent.chunk_bytes/4096+1)))
                .ok_or("component direct reader bound overflow")?;bytes=bytes.max(bound);
        }
        total=total.checked_add(bytes).ok_or("aggregate reader bound overflow")?;
        let config=Config{workers,budget_bytes:bytes,..parent};config.validate()?;configs.push(config);
    }
    if total>parent.budget_bytes{return Err(format!("component readers need {total} bytes; aggregate budget is {}",parent.budget_bytes));}
    Ok(configs)
}

/// Direct DMA holds existing reader leases, so event depth is independent of
/// the separately budgeted pinned-copy ring. Unset preserves the old policy.
pub fn upload_slot_count(config:Config,direct:bool,requested:Option<usize>)->Result<usize> {
    config.validate()?;
    match requested {
        None=>Ok(config.pinned_slots),
        Some(depth) if direct && (1..=16).contains(&depth)=>Ok(depth),
        Some(_) if !direct=>Err("NATIVE_LOAD_DIRECT_INFLIGHT requires direct uploads".into()),
        Some(_)=>Err("NATIVE_LOAD_DIRECT_INFLIGHT must be 1..=16".into()),
    }
}

pub fn configured_upload_slot_count(config:Config,direct:bool)->Result<usize> {
    let requested=std::env::var("NATIVE_LOAD_DIRECT_INFLIGHT").ok()
        .map(|value|value.parse::<usize>().map_err(|_|"invalid NATIVE_LOAD_DIRECT_INFLIGHT")).transpose()?;
    upload_slot_count(config,direct,requested)
}
enum Packet {
    Chunk {worker:usize,job:usize,offset:usize,data:ReadChunk},
    Done {job:usize,result:Result<()>},
}
struct Worker {
    recycle:Option<mpsc::SyncSender<ReadChunk>>,
    thread:Option<thread::JoinHandle<()>>,
}
struct SharedReaders {threads:std::sync::Mutex<Vec<thread::JoinHandle<()>>>,cancel:Arc<std::sync::atomic::AtomicBool>}
impl Drop for SharedReaders {
    fn drop(&mut self){for worker in self.threads.get_mut().unwrap().drain(..){let _=worker.join();}}
}
pub struct ReaderPool {
    workers:Vec<Worker>,
    packets:Option<mpsc::Receiver<Packet>>,
    jobs:Arc<Vec<Job>>,
    one_use:bool,retained:Option<RetainedChunks>,complete:bool,
    copy_retained:Option<(usize,usize)>,chunk_bytes:usize,
    shared:Option<Arc<SharedReaders>>,
    offsets:Vec<usize>,verified:Vec<bool>,remaining:usize,
    control:Option<Arc<crate::background_io::Control>>,
}
/// Equalize issued payload bytes, rather than tensor counts. A small tensor
/// group must not wait behind one large tensor from every other group for each
/// of its tiny parameters. Each group's source order is admitted separately.
fn interleave_component_jobs(mut orders:Vec<std::vec::IntoIter<(usize,usize,Job)>>)->Vec<(usize,usize,Job)> {
    use std::{cmp::Reverse,collections::BinaryHeap};
    let mut ready:BinaryHeap<_>=(0..orders.len()).map(|index|Reverse((0u128,index))).collect();
    let mut jobs=Vec::new();
    while let Some(Reverse((issued,stage)))=ready.pop() {
        if let Some(job)=orders[stage].next() {
            let issued=issued+job.2.bytes as u128;jobs.push(job);
            if orders[stage].len()!=0 {ready.push(Reverse((issued,stage)));}
        }
    }
    jobs
}
fn scheduled_job_order(entries:&[Job],schedule:ReadSchedule)->Vec<usize> {
    let mut order:Vec<_>=(0..entries.len()).collect();
    if schedule==ReadSchedule::FileOffset {
        order.sort_by(|&a,&b|entries[a].path.cmp(&entries[b].path)
            .then(entries[a].file_offset.cmp(&entries[b].file_offset)).then(a.cmp(&b)));
    }else{order.sort_by(|&a,&b|entries[b].bytes.cmp(&entries[a].bytes).then(a.cmp(&b)));}
    order
}
fn component_jobs(mut orders:Vec<std::vec::IntoIter<(usize,usize,Job)>>,schedule:ReadSchedule)->Vec<(usize,usize,Job)> {
    if schedule!=ReadSchedule::RoundRobin{return interleave_component_jobs(orders);}
    let mut jobs=Vec::new();
    loop {
        let before=jobs.len();
        for order in &mut orders{if let Some(job)=order.next(){jobs.push(job);}}
        if jobs.len()==before{return jobs;}
    }
}
impl ReaderPool {
    /// Independent components share ONE bounded worker/buffer budget. Their
    /// consumers may upload concurrently. Byte-fair assignment starts every
    /// group early without making small groups wait for equal tensor counts
    /// of much larger payloads. No reader or buffer quota is multiplied.
    pub fn start_components(stages:Vec<Vec<Job>>,config:Config,direct:bool,verify_sha:bool,guarded:bool)->Result<Vec<Self>> {
        Self::start_components_with_policy(stages,config,direct,verify_sha,guarded,FileReadPolicy::Auto)
    }
    /// Explicit NATIVE_LOAD_FILE_IO overrides the declared default policy.
    pub fn start_components_with_policy(stages:Vec<Vec<Job>>,config:Config,direct:bool,verify_sha:bool,guarded:bool,policy:FileReadPolicy)->Result<Vec<Self>> {
        Self::start_components_scheduled(stages,config,direct,verify_sha,guarded,policy,ReadSchedule::ByteFair)
    }
    pub fn start_components_scheduled(stages:Vec<Vec<Job>>,config:Config,direct:bool,verify_sha:bool,guarded:bool,policy:FileReadPolicy,schedule:ReadSchedule)->Result<Vec<Self>> {
        config.validate()?;
        if stages.is_empty(){return Err("component list is empty".into());}
        if stages.iter().flatten().any(|j|j.finite_bf16 && (j.bytes%2!=0 || config.chunk_bytes%2!=0)) {
            return Err("finite BF16 validation requires even tensor/chunk lengths".into());
        }
        if stages.iter().flatten().any(|j|!verify_sha && !guarded && j.file_bytes.is_some()) {
            return Err("packed tensors require foreground integrity".into());
        }
        let count=config.workers.min(stages.iter().map(Vec::len).sum());
        let file_io=FileIo::from_env(policy.into())?.for_pool(stages.iter().flatten(),config,direct);
        if file_io==FileIo::Direct && (!direct || config.chunk_bytes<8192 || config.chunk_bytes%4096!=0){
            return Err("direct file reads require exclusive aligned direct buffers".into());
        }
        if direct {
            let extra=usize::from(file_io!=FileIo::Buffered)*4096;
            let aligned=config.chunk_bytes.checked_add(extra).and_then(|n|n.checked_add(4095))
                .ok_or("component chunk alignment overflow")?/4096*4096;
            // Direct upload slots own reader leases and allocate no second
            // host ring. Alignment/bitmap metadata fit the original bound.
            let metadata=if file_io!=FileIo::Buffered{count*2*(config.chunk_bytes/4096+1)}else{0};
            let actual=(2*count).checked_mul(aligned).and_then(|n|n.checked_add(metadata))
                .ok_or("component reader bound overflow")?;
            if actual>config.budget_bytes{return Err("component reader allocations exceed budget".into());}
        }
        let shared=Arc::new(SharedReaders{threads:std::sync::Mutex::new(Vec::new()),cancel:Arc::new(std::sync::atomic::AtomicBool::new(false))});
        let mut recycle=Vec::new();let mut free=Vec::new();
        let buffers_span=crate::startup_trace::Span::new("startup_reader_prepare",format!("shared buffers: {count} workers direct={direct}"),2*count*config.chunk_bytes);
        for _ in 0..count {
            let (send,receive)=mpsc::sync_channel(2);
            for _ in 0..2 {
                let data=ReadChunk::reader(config.chunk_bytes,direct,file_io)?;
                send.send(data).map_err(|_|"component buffer initialization")?;
            }
            recycle.push(send);free.push(receive);
        }
        drop(buffers_span);
        let plans_span=crate::startup_trace::Span::new("startup_reader_prepare","component queues and byte-fair schedule",0);
        let mut pools=Vec::new();let mut senders=Vec::new();let mut orders=Vec::new();
        for (stage,entries) in stages.into_iter().enumerate() {
            let entries=Arc::new(entries);
            let order=scheduled_job_order(&entries,schedule);
            orders.push(order.into_iter().map(|index|(stage,index,entries[index].clone())).collect::<Vec<_>>().into_iter());
            let (send,packets)=mpsc::sync_channel(count.max(1));senders.push(send);
            pools.push(Self{workers:recycle.iter().map(|s|Worker{recycle:Some(s.clone()),thread:None}).collect(),
                packets:Some(packets),jobs:entries.clone(),offsets:vec![0;entries.len()],verified:vec![false;entries.len()],remaining:entries.len(),control:None,one_use:false,retained:None,copy_retained:None,chunk_bytes:config.chunk_bytes,complete:false,shared:Some(shared.clone())});
        }
        let jobs=Arc::new(component_jobs(orders,schedule));let next=Arc::new(AtomicUsize::new(0));
        drop(plans_span);
        let _threads_span=crate::startup_trace::Span::new("startup_reader_prepare","start shared reader threads",0);
        for (worker,free) in free.into_iter().enumerate(){
            let (jobs,next,senders,cancel)=(jobs.clone(),next.clone(),senders.clone(),shared.cancel.clone());
            let handle=thread::Builder::new().name(format!("component-reader-{worker}")).spawn(move||{
                let mut files=OpenedFiles::default();
                while let Some((stage,index,job))=jobs.get(next.fetch_add(1,Ordering::Relaxed)){
                    if cancel.load(Ordering::Acquire){break;}
                    let result=read_job_files(worker,*index,job,config.chunk_bytes,&free,&senders[*stage],verify_sha,guarded,false,file_io,Some(&cancel),None,None,Some(&mut files));
                    let failed=result.is_err();
                    if senders[*stage].send(Packet::Done{job:*index,result}).is_err() || failed {break;}
                }
            }).map_err(|e|format!("component reader: {e}"))?;
            shared.threads.lock().unwrap().push(handle);
        }
        Ok(pools)
    }
    /// One worker budget for all admitted resident stages. Readers may proceed
    /// to a later stage as soon as earlier CPU jobs are assigned; GPU stages
    /// still complete in dependency order. No additional model-specific pool.
    pub fn start_retained_stages(stages:Vec<Vec<Job>>,config:Config,verify_sha:bool,guarded:bool)->Result<Vec<Self>> {
        Self::start_retained_schedule(stages,config,verify_sha,guarded,None)
    }
    /// Foreground independent reads fill their final immutable host storage.
    /// Each component uploads through its existing bounded staging ring;
    /// final host registration is deferred until the owner requests residency.
    pub(crate) fn start_retained_components(stages:Vec<Vec<Job>>,config:Config,verify_sha:bool,guarded:bool)->Result<Vec<Self>> {
        if !snapshot_file_policy()? {
            return Err("retained component snapshots require buffered file reads".into());
        }
        Self::start_retained_components_scheduled(stages,config,verify_sha,guarded,ReadSchedule::ByteFair)
    }
    pub(crate) fn start_retained_components_scheduled(stages:Vec<Vec<Job>>,config:Config,verify_sha:bool,guarded:bool,schedule:ReadSchedule)->Result<Vec<Self>> {
        if !snapshot_file_policy()?{return Err("retained component snapshots require buffered file reads".into());}
        Self::start_retained_schedule(stages,config,verify_sha,guarded,Some(schedule))
    }
    fn start_retained_schedule(stages:Vec<Vec<Job>>,config:Config,verify_sha:bool,guarded:bool,schedule:Option<ReadSchedule>)->Result<Vec<Self>> {
        config.validate()?;
        let file_io=FileIo::from_env(FileIo::Buffered)?;
        let sizes:Vec<_>=stages.iter().flatten().map(|j|j.bytes).collect();
        if stages.iter().flatten().any(|j|j.finite_bf16 && (j.bytes%2!=0 || config.chunk_bytes%2!=0)) {
            return Err("finite BF16 validation requires even tensor and chunk lengths".into());
        }
        if read_ahead_bound(&sizes,config,true,stages.len())?>config.budget_bytes {
            return Err("resident stage reads exceed admitted RAM".into());
        }
        let count=config.workers.min(sizes.len());
        let shared=Arc::new(SharedReaders{threads:std::sync::Mutex::new(Vec::new()),cancel:Arc::new(std::sync::atomic::AtomicBool::new(false))});
        let mut pools=Vec::new();let mut senders=Vec::new();let mut orders=Vec::new();
        for (stage,entries) in stages.into_iter().enumerate() {
            if entries.iter().any(|j|!verify_sha && !guarded && j.file_bytes.is_some()) {
                return Err("packed tensors require foreground SHA".into());
            }
            let entries=Arc::new(entries);
            let order=scheduled_job_order(&entries,schedule.unwrap_or(ReadSchedule::ByteFair));
            orders.push(order.into_iter().map(|index|(stage,index,entries[index].clone())).collect::<Vec<_>>().into_iter());
            let (send,packets)=mpsc::sync_channel(entries.iter().map(|j|j.bytes.div_ceil(config.chunk_bytes)+1).sum::<usize>().max(1));
            senders.push(send);
            let mut workers=Vec::new();
            for _ in 0..count {let (recycle,closed)=mpsc::sync_channel(0);drop(closed);workers.push(Worker{recycle:Some(recycle),thread:None});}
            let mut pool=Self{workers,packets:Some(packets),jobs:entries.clone(),offsets:vec![0;entries.len()],verified:vec![false;entries.len()],remaining:entries.len(),control:None,one_use:true,retained:None,copy_retained:None,chunk_bytes:config.chunk_bytes,complete:false,shared:Some(shared.clone())};
            pool.retain_outputs()?;pools.push(pool);
        }
        let mut jobs=Vec::new();
        if let Some(schedule)=schedule {jobs=component_jobs(orders,schedule);}
        else {for order in orders {jobs.extend(order);}}
        let jobs=Arc::new(jobs);let next=Arc::new(AtomicUsize::new(0));
        for worker in 0..count {
            let (jobs,next,senders,cancel)=(jobs.clone(),next.clone(),senders.clone(),shared.cancel.clone());
            let handle=thread::Builder::new().name(format!("resident-reader-{worker}")).spawn(move||{
                if schedule.is_none() {let _=crate::cpu_priority::background();}
                while let Some((stage,index,job))=jobs.get(next.fetch_add(1,Ordering::Relaxed)) {
                    if cancel.load(Ordering::Acquire){break;}
                    let send=&senders[*stage];
                    let result=(||->Result<()> {
                        let (ready,free)=mpsc::sync_channel(job.bytes.div_ceil(config.chunk_bytes).max(1));
                        for offset in (0..job.bytes).step_by(config.chunk_bytes) {
                            ready.send(ReadChunk::new(config.chunk_bytes.min(job.bytes-offset),true)?).map_err(|_|"resident reader cancelled")?;
                        }
                        drop(ready);
                        read_job_cancel(worker,*index,job,config.chunk_bytes,&free,send,verify_sha,guarded,false,file_io,Some(&cancel),None,None)
                    })();
                    let failed=result.is_err();
                    if send.send(Packet::Done{job:*index,result}).is_err() || failed {break;}
                }
            }).map_err(|e|format!("resident reader spawn: {e}"))?;
            shared.threads.lock().unwrap().push(handle);
        }
        Ok(pools)
    }
    /// Only admitted one-use read-ahead may retain its buffers: a reusable
    /// worker must never overwrite bytes that an immutable owner can observe.
    pub fn retain_outputs(&mut self)->Result<()> {
        if !self.one_use || self.retained.is_some(){return Err("retention requires one-use readers".into());}
        self.retained=Some(RetainedChunks{chunks:(0..self.jobs.len()).map(|_|Vec::new()).collect()});Ok(())
    }
    /// Keep independent immutable copies while the original bounded reader
    /// buffers continue to recycle for DMA. Admit every aligned output before
    /// any consumption; failed integrity checks never publish this storage.
    pub fn retain_copies(&mut self,budget:usize)->Result<()> {
        if self.one_use || self.complete || self.offsets.iter().any(|&n|n!=0)
            || self.copy_retained.is_some_and(|(used,_)|used!=0)
            || (self.retained.is_some() && self.copy_retained.is_none()) {
            return Err("retained copies require unconsumed reusable readers".into());
        }
        let bytes=retained_copy_bound(&self.jobs.iter().map(|j|j.bytes).collect::<Vec<_>>(),self.chunk_bytes)?;
        if bytes>budget {return Err("retained copies exceed admitted storage".into());}
        self.copy_retained=Some((0,budget));
        self.retained=Some(RetainedChunks{chunks:(0..self.jobs.len()).map(|_|Vec::new()).collect()});Ok(())
    }
    pub(crate) fn retains_copies(&self)->bool {self.copy_retained.is_some()}
    pub(crate) fn owns_retained_read_buffers(&self)->bool {self.one_use && self.retained.is_some()}
    fn retain_chunk(&mut self,job:usize,offset:usize,data:&ReadChunk)->Result<()> {
        if let Some(retained)=&mut self.retained {
            if let Some((used,budget))=&mut self.copy_retained {
                let bytes=data.len().checked_add(4095).map(|n|n/4096*4096).ok_or("retained chunk overflow")?;
                *used=used.checked_add(bytes).filter(|n|*n<=*budget).ok_or("retained copy exceeds admitted storage")?;
                let mut copy=ReadChunk::new(data.len(),true)?;
                copy.copy_from_slice(data);
                retained.chunks[job].push((offset,copy.aligned.take().ok_or("aligned retained storage missing")?,data.len()));
                return Ok(());
            }
            if data.prefix!=0{return Err("retention requires an unshifted one-use chunk".into());}
            let storage=data.aligned.as_ref().ok_or("retention requires aligned reader storage")?;
            retained.chunks[job].push((offset,storage.clone(),data.len()));
        }
        Ok(())
    }
    pub fn take_retained(&mut self)->Result<Option<RetainedChunks>> {
        if !self.complete{return Err("cannot retain an incomplete or unverified load".into());}
        Ok(self.retained.take())
    }
    /// Finish the ORIGINAL reusable read schedule into separately admitted
    /// immutable storage. Recycle each reader chunk only after its bounded
    /// CPU copy; never retain an alias that a reader can overwrite. Queued
    /// bytes, open files, offsets and guards are consumed without restarting.
    /// No storage is published unless every original integrity check passes.
    pub fn finish_retained_copy(mut self,budget:usize)->Result<RetainedChunks> {
        self.retain_copies(budget)?;
        self.consume_all(|_,_,_|Ok(()))?;
        self.take_retained()?.ok_or_else(||"retained copies missing".into())
    }
    /// Admitted stages can finish CPU reads while their GPU owner initializes.
    /// Exact-sized chunks are retained once, under the existing explicit byte limit.
    /// The upload event still owns each lease; no chunk waits for GPU recycling.
    pub fn start_read_ahead(jobs:Vec<Job>,config:Config)->Result<Self> {
        Self::start_read_ahead_options(jobs,config,false,true,false)
    }
    pub fn start_read_ahead_options(jobs:Vec<Job>,config:Config,direct:bool,verify_sha:bool,guarded:bool)->Result<Self> {
        config.validate()?;
        let mut capacity=jobs.len();
        let required=read_ahead_bound(&jobs.iter().map(|j|j.bytes).collect::<Vec<_>>(),config,direct,1)?;
        for job in &jobs {
            if job.finite_bf16 && (job.bytes%2!=0 || config.chunk_bytes%2!=0) {return Err("finite BF16 validation requires even tensor and chunk lengths".into());}
            if !verify_sha && !guarded && job.file_bytes.is_some(){return Err("packed tensors require foreground SHA".into());}
            let chunks=job.bytes/config.chunk_bytes+usize::from(job.bytes%config.chunk_bytes!=0);
            capacity=capacity.checked_add(chunks).ok_or("encoder read-ahead packet overflow")?;
        }
        if required>config.budget_bytes {return Err(format!("encoder read-ahead needs {required} bytes, budget is {}",config.budget_bytes));}
        eprintln!("native full read-ahead: tensors={} bound_bytes={required} workers={} chunk_bytes={} pinned_slots={} direct={direct}",
            jobs.len(),config.workers,config.chunk_bytes,config.pinned_slots);
        let jobs=Arc::new(jobs);let count=config.workers.min(jobs.len());
        let mut order:Vec<_>=(0..jobs.len()).collect();
        order.sort_by(|a,b|jobs[*b].bytes.cmp(&jobs[*a].bytes).then(a.cmp(b)));
        let order=Arc::new(order);let next=Arc::new(AtomicUsize::new(0));
        let (send,packets)=mpsc::sync_channel(capacity.max(1));
        let mut pool=Self{workers:Vec::new(),packets:Some(packets),jobs:jobs.clone(),one_use:true,retained:None,copy_retained:None,chunk_bytes:config.chunk_bytes,complete:false,shared:None,offsets:vec![0;jobs.len()],verified:vec![false;jobs.len()],remaining:jobs.len(),control:None};
        for worker in 0..count {
            // Reclaimed one-use leases are discarded after completion. Keeping
            // a closed recycle receiver makes send nonblocking on every path.
            let (recycle,retired)=mpsc::sync_channel(0);drop(retired);
            let (send,jobs,order,next)=(send.clone(),jobs.clone(),order.clone(),next.clone());
            let thread=thread::Builder::new().name(format!("read-ahead-{worker}")).spawn(move||{
                // Read-ahead must not starve tokenization or the GPU submitter.
                let _=crate::cpu_priority::background();
                while let Some(&index)=order.get(next.fetch_add(1,Ordering::Relaxed)) {
                    let job=&jobs[index];
                    let result=(||->Result<()> {
                        let chunks=job.bytes/config.chunk_bytes+usize::from(job.bytes%config.chunk_bytes!=0);
                        let (ready,free)=mpsc::sync_channel(chunks.max(1));
                        crate::startup_trace::measure("reader_allocate","retained-read-ahead",job.bytes,||->Result<()> {
                            for offset in (0..job.bytes).step_by(config.chunk_bytes) {
                                ready.send(ReadChunk::new(config.chunk_bytes.min(job.bytes-offset),direct)?).map_err(|_|"encoder buffer initialization")?;
                            }
                            Ok(())
                        })?;
                        drop(ready);
                        read_job(worker,index,job,config.chunk_bytes,&free,&send,verify_sha,guarded,false,None,None)
                    })();
                    let failed=result.is_err();
                    if send.send(Packet::Done{job:index,result}).is_err() || failed {break;}
                }
            }).map_err(|e|format!("encoder reader spawn: {e}"))?;
            pool.workers.push(Worker{recycle:Some(recycle),thread:Some(thread)});
        }
        Ok(pool)
    }
    /// Start before CUDA context creation to overlap the first reads with it.
    pub fn start(jobs:Vec<Job>,config:Config)->Result<Self> {
        Self::start_policy(jobs,config,false,true,false)
    }
    pub fn start_options(jobs:Vec<Job>,config:Config,direct:bool,verify_sha:bool)->Result<Self> {
        Self::start_policy(jobs,config,direct,verify_sha,false)
    }
    /// Dedicated DiT cache policy: a trusted light guard admits packed payloads
    /// while a separate owner audits the immutable retained bytes in full.
    pub fn start_guarded(jobs:Vec<Job>,config:Config)->Result<Self> {
        Self::start_policy(jobs,config,false,false,true)
    }
    fn start_policy(jobs:Vec<Job>,config:Config,direct:bool,verify_sha:bool,guarded:bool)->Result<Self> {
        Self::start_scheduled_mode(jobs,config,direct,verify_sha,guarded,false,false)
    }
    #[cfg(test)]
    fn start_scheduled(jobs:Vec<Job>,config:Config,direct:bool,verify_sha:bool,guarded:bool,balanced:bool)->Result<Self> {
        Self::start_scheduled_mode(jobs,config,direct,verify_sha,guarded,balanced,false)
    }
    pub fn start_scheduled_mode(jobs:Vec<Job>,config:Config,direct:bool,verify_sha:bool,guarded:bool,balanced:bool,mapped:bool)->Result<Self> {
        Self::start_scheduled_io(jobs,config,direct,verify_sha,guarded,balanced,mapped,FileIo::Buffered)
    }
    /// Declared checkpoints opt into the automatic packed-file policy. Keep
    /// the established low-level reader's default behavior for legacy adapters.
    pub fn start_checkpoint_mode(jobs:Vec<Job>,config:Config,direct:bool,verify_sha:bool,guarded:bool,balanced:bool)->Result<Self>{
        Self::start_scheduled_io(jobs,config,direct,verify_sha,guarded,balanced,false,FileIo::Auto)
    }
    fn start_scheduled_io(jobs:Vec<Job>,config:Config,direct:bool,verify_sha:bool,guarded:bool,balanced:bool,mapped:bool,default_io:FileIo)->Result<Self>{
        Self::start_destinations(jobs,config,direct,verify_sha,guarded,balanced,mapped,default_io,None)
    }
    /// CPU-only retained host loads: workers read into disjoint final ranges.
    /// Chunk acknowledgements, light guards and cooperative priority are the
    /// same as staged reads, without a serial copy through the consumer.
    pub fn start_into_guarded(jobs:Vec<Job>,config:Config,targets:Vec<HostTarget>)->Result<Self>{
        if targets.len()!=jobs.len()||targets.iter().zip(&jobs).any(|(t,j)|t.bytes!=j.bytes){
            return Err("host reader destination extents differ".into());
        }
        let mut ranges=targets.iter().map(|t|(t.pointer as usize,t.bytes)).collect::<Vec<_>>();
        ranges.sort_unstable();
        if ranges.windows(2).any(|p|p[0].0+p[0].1>p[1].0){return Err("host reader destinations overlap".into());}
        Self::start_destinations(jobs,config,false,false,true,true,false,
            FileIo::Buffered,Some(Arc::new(targets.into_iter().map(Arc::new).collect())))
    }
    fn start_destinations(jobs:Vec<Job>,config:Config,direct:bool,verify_sha:bool,guarded:bool,balanced:bool,mapped:bool,default_io:FileIo,
        targets:Option<Arc<Vec<Arc<HostTarget>>>>)->Result<Self>{
        config.validate()?;
        let file_io=if mapped || targets.is_some(){FileIo::Buffered}else{FileIo::from_env(default_io)?.for_pool(jobs.iter(),config,direct)};
        if file_io==FileIo::Direct && (!direct || config.chunk_bytes<8192 || config.chunk_bytes%4096!=0){
            return Err("direct file reads require exclusive aligned direct buffers".into());
        }
        if jobs.iter().any(|j|j.finite_bf16 && (j.bytes%2!=0 || config.chunk_bytes%2!=0)) {
            return Err("finite BF16 validation requires even tensor and chunk lengths".into());
        }
        if mapped {
            if !direct{return Err("mapped model reads require direct upload".into());}
            #[cfg(not(target_os="linux"))]
            return Err("mapped model reads require Linux".into());
            #[cfg(target_os="linux")]
            if mapping::budget(config,&jobs)?>config.budget_bytes {return Err("mapped reader allocations exceed budget".into());}
        }
        if !verify_sha && !guarded && jobs.iter().any(|j|j.file_bytes.is_some()) {return Err("packed tensors require foreground SHA".into());}
        if direct {
            let extra=usize::from(file_io!=FileIo::Buffered)*4096;
            let rounded=config.chunk_bytes.checked_add(extra).and_then(|n|n.checked_add(4095)).ok_or("aligned chunk overflow")?/4096*4096;
            let actual=config.workers.checked_mul(2).and_then(|n|n.checked_mul(rounded))
                .and_then(|n|n.checked_add(if file_io!=FileIo::Buffered{config.workers*2*(config.chunk_bytes/4096+1)}else{0}))
                .ok_or("aligned read budget overflow")?;
            if actual>config.budget_bytes{return Err("aligned reader allocations exceed budget".into());}
        }
        let jobs=Arc::new(jobs);
        let count=config.workers.min(jobs.len());
        // GPU allocation and tensor indices stay in manifest order. Only file
        // assignment changes. Start large files first, then let a free reader
        // take the next job instead of waiting behind a fixed modulo partition.
        let mut order:Vec<usize>=(0..jobs.len()).collect();
        if balanced {order.sort_by(|a,b|jobs[*b].bytes.cmp(&jobs[*a].bytes).then(a.cmp(b)));}
        let order=Arc::new(order);let next=Arc::new(AtomicUsize::new(0));
        let capacity=count.max(1);
        let (send,packets)=mpsc::sync_channel(capacity);
        let mut pool=Self {workers:Vec::new(),packets:Some(packets),jobs:jobs.clone(),one_use:false,retained:None,copy_retained:None,chunk_bytes:config.chunk_bytes,complete:false,shared:None,offsets:vec![0;jobs.len()],verified:vec![false;jobs.len()],remaining:jobs.len(),control:crate::background_io::Control::current()};
        for worker in 0..count {
            let send=send.clone();
            let (recycle,free)=mpsc::sync_channel(2);
            for _ in 0..2 {
                let data=if mapped || targets.is_some(){ReadChunk{ordinary:Vec::new(),aligned:None,mapped:None,prefix:0,length:0,target:None}}
                    else{ReadChunk::reader(config.chunk_bytes,direct,file_io)?};
                recycle.send(data).map_err(|_|"model chunk initialization")?;
            }
            let jobs=jobs.clone();
            let order=order.clone();let next=next.clone();
            let control=pool.control.clone();
            let targets=targets.clone();
            let thread=thread::Builder::new().name(format!("model-reader-{worker}"))
                .spawn(move|| {
                    let mut assigned=(worker..jobs.len()).step_by(count);
                    loop {
                        let index=if balanced {order.get(next.fetch_add(1,Ordering::Relaxed)).copied()}else{assigned.next()};
                        let Some(index)=index else{break;};
                        let result=read_job_cancel(worker,index,&jobs[index],config.chunk_bytes,&free,&send,verify_sha,guarded,mapped,file_io,None,control.as_deref(),targets.as_ref().map(|values|&values[index]));
                        let failed=result.is_err();
                        if crate::startup_trace::measure("reader_done_send",jobs[index].path.to_string_lossy(),0,||
                            send.send(Packet::Done {job:index,result})).is_err() || failed {break;}
                    }
                }).map_err(|e|format!("model reader spawn: {e}"))?;
            pool.workers.push(Worker {recycle:Some(recycle),thread:Some(thread)});
        }
        Ok(pool)
    }
    /// GPU allocations stay in manifest order. Consume whichever worker has a
    /// chunk ready, preserving each tensor's byte order. Strict tensor-by-tensor
    /// consumption would stall all other hash workers behind a large tensor.
    pub fn consume_all(&mut self,mut upload:impl FnMut(usize,usize,&[u8])->Result<()>)->Result<()> {
        while self.remaining>0 {
            self.checkpoint()?;
            let packet=match self.packets.as_ref().ok_or("model loader cancelled")?.recv_timeout(std::time::Duration::from_millis(5)){
                Ok(packet)=>packet,Err(mpsc::RecvTimeoutError::Timeout)=>continue,
                Err(_)=>return Err("model reader terminated before verification".into()),
            };
            match packet {
                Packet::Chunk {worker,job,offset,data}=>{
                    if worker>=self.workers.len() || job>=self.jobs.len() || self.verified[job] || offset!=self.offsets[job] {return Err("model chunk order mismatch".into());}
                    self.offsets[job]=offset.checked_add(data.len()).filter(|n|*n<=self.jobs[job].bytes).ok_or("model chunk extent mismatch")?;
                    self.retain_chunk(job,offset,&data)?;
                    let result=upload(job,offset,&data);
                    // The last chunk can race the worker's verified Done packet
                    // and exit. Its completion packet remains authoritative.
                    let _=self.workers[worker].recycle.as_ref().ok_or("model loader cancelled")?.send(data);
                    result?;
                },
                Packet::Done {job,result}=>{
                    if job>=self.jobs.len() || self.verified[job] {return Err("model verification order mismatch".into());}
                    result?;
                    if self.offsets[job]!=self.jobs[job].bytes {return Err("model upload byte count mismatch".into());}
                    self.verified[job]=true;self.remaining-=1;
                },
            }
        }
        self.complete=true;Ok(())
    }
    pub fn foreground(&self,value:bool){if let Some(control)=&self.control{control.foreground(value);}}
    pub fn initialized_bytes(&self)->usize{self.offsets.iter().sum()}
    pub fn checkpoint(&self)->Result<()>{self.control.as_ref().map_or(Ok(()),|control|control.checkpoint())}
}
impl Drop for ReaderPool {
    fn drop(&mut self) {
        if !self.complete {if let Some(shared)=&self.shared{shared.cancel.store(true,Ordering::Release);}}
        if let Some(control)=&self.control{control.close();}
        // Disconnect BOTH directions before joining: a worker may be blocked
        // waiting for a recycled chunk or sending a full look-ahead channel.
        let workers=std::mem::take(&mut self.workers);
        self.packets.take();
        let mut joins=Vec::new();
        for mut worker in workers {
            worker.recycle.take();
            if let Some(thread)=worker.thread.take() {joins.push(thread);}
        }
        for thread in joins {let _=thread.join();}
        self.shared.take(); // last stage joins the shared CPU readers
    }
}
fn read_job(worker:usize,index:usize,job:&Job,chunk:usize,free:&mpsc::Receiver<ReadChunk>,send:&mpsc::SyncSender<Packet>,verify_sha:bool,guarded:bool,mapped:bool,
    control:Option<&crate::background_io::Control>,target:Option<&Arc<HostTarget>>)->Result<()> {
    // One-use retained/read-ahead leases keep their original pointer layout.
    read_job_cancel(worker,index,job,chunk,free,send,verify_sha,guarded,mapped,FileIo::Buffered,None,control,target)
}
// One buffered and optional direct descriptor per packed file on each lane.
// Cursors are never shared between workers; every tensor still checks the
// current descriptor size, its exact extent and its foreground SHA/guard.
struct OpenedFile {path:PathBuf,file:File,direct:Option<File>}
impl OpenedFile {
    fn open(path:&PathBuf)->Result<Self>{
        let file=crate::startup_trace::measure("reader_open",path.to_string_lossy(),0,||File::open(path))
            .map_err(|e|format!("{}: {e}",path.display()))?;
        Ok(Self{path:path.clone(),file,direct:None})
    }
}
#[derive(Default)]
struct OpenedFiles {entries:Vec<OpenedFile>}
impl OpenedFiles {
    fn get(&mut self,path:&PathBuf)->Result<&mut OpenedFile>{
        if let Some(index)=self.entries.iter().position(|entry|entry.path==*path){return Ok(&mut self.entries[index]);}
        // At most eight descriptors per lane, including optional O_DIRECT.
        if self.entries.len()==4{self.entries.clear();}
        self.entries.push(OpenedFile::open(path)?);Ok(self.entries.last_mut().unwrap())
    }
}
fn read_job_cancel(worker:usize,index:usize,job:&Job,chunk:usize,free:&mpsc::Receiver<ReadChunk>,send:&mpsc::SyncSender<Packet>,verify_sha:bool,guarded:bool,mapped:bool,file_io:FileIo,cancel:Option<&std::sync::atomic::AtomicBool>,
    control:Option<&crate::background_io::Control>,target:Option<&Arc<HostTarget>>)->Result<()> {
    read_job_files(worker,index,job,chunk,free,send,verify_sha,guarded,mapped,file_io,cancel,control,target,None)
}
fn read_job_files(worker:usize,index:usize,job:&Job,chunk:usize,free:&mpsc::Receiver<ReadChunk>,send:&mpsc::SyncSender<Packet>,verify_sha:bool,guarded:bool,mapped:bool,file_io:FileIo,cancel:Option<&std::sync::atomic::AtomicBool>,
    control:Option<&crate::background_io::Control>,target:Option<&Arc<HostTarget>>,files:Option<&mut OpenedFiles>)->Result<()> {
    if let Some(control)=control{control.wait()?;}
    let mut file_io=file_io.for_job(job,chunk,mapped);
    let mut direct_disabled=false;
    let mut cache_bitmap:Vec<u8>=Vec::new();
    let mut owned;
    let opened=if job.file_bytes.is_some()&&files.is_some(){files.unwrap().get(&job.path)?}
        else{owned=OpenedFile::open(&job.path)?;&mut owned};
    let file=&mut opened.file;let direct_file=&mut opened.direct;
    let file_bytes=job.file_bytes.unwrap_or(job.bytes as u64);
    if crate::startup_trace::measure("reader_metadata",job.path.to_string_lossy(),0,||file.metadata()).map_err(|e|e.to_string())?.len()!=file_bytes {return Err(format!("wrong model size: {}",job.path.display()));}
    if job.file_offset.checked_add(job.bytes as u64).is_none_or(|end|end>file_bytes) {return Err("invalid packed model range".into());}
    crate::startup_trace::measure("reader_seek",job.path.to_string_lossy(),0,||file.seek(SeekFrom::Start(job.file_offset))).map_err(|e|e.to_string())?;
    let mut guard=if guarded {Some(light::ReaderGuard::new(job.guard.clone().ok_or("trusted light SHA guard missing")?))}else{None};
    let mut hash=Sha256::new();
    let mut offset=0;
    while offset<job.bytes {
        let mut data=crate::startup_trace::measure("reader_backpressure",job.path.to_string_lossy(),0,||->Result<ReadChunk>{
            if let Some(cancel)=cancel {
                loop {
                    if cancel.load(Ordering::Acquire){return Err("component load cancelled".into());}
                    match free.recv_timeout(std::time::Duration::from_millis(25)) {
                        Ok(data)=>return Ok(data),Err(mpsc::RecvTimeoutError::Timeout)=>{},
                        Err(mpsc::RecvTimeoutError::Disconnected)=>return Err("component reader retired".into()),
                    }
                }
            }else{free.recv().map_err(|_|"model loader cancelled".into())}
        })?;
        if let Some(control)=control{control.wait()?;}
        if file_io==FileIo::Auto && data.aligned.is_none(){file_io=FileIo::Buffered;}
        let bytes=chunk.min(job.bytes-offset);
        let absolute=job.file_offset.checked_add(offset as u64).ok_or("model read offset overflow")?;
        if let Some(target)=target{
            if offset.checked_add(bytes).is_none_or(|end|end>target.bytes){return Err("host reader escapes its destination".into());}
            data.target=Some((target.clone(),offset));data.prefix=0;data.length=bytes;
            crate::startup_trace::measure("read_file",job.path.to_string_lossy(),bytes,||file.read_exact(&mut data))
                .map_err(|e|format!("host model read {}: {e}",job.path.display()))?;
        }
        let mut use_direct=file_io==FileIo::Direct;
        if file_io==FileIo::Auto && !direct_disabled {
            #[cfg(target_os="linux")]
            {use_direct=!crate::startup_trace::measure("read_cache_probe",job.path.to_string_lossy(),0,||
                mapping::resident(&file,absolute,bytes,&mut cache_bitmap)).unwrap_or(false);}
        }
        let mut direct_done=target.is_some();
        if use_direct {
            let prefix=(absolute%4096) as usize;
            let needed=prefix.checked_add(bytes).ok_or("direct file read overflow")?;
            let io_bytes=needed.div_ceil(4096)*4096;
            if data.aligned.as_ref().is_none_or(|storage|io_bytes>storage.layout.size()){
                return Err("direct file read exceeds the admitted buffer".into());
            }
            data.resize(io_bytes,0);
            let attempt=crate::startup_trace::measure("direct_file_read",job.path.to_string_lossy(),bytes,||->std::io::Result<()>{
                if direct_file.is_none(){
                    #[cfg(target_os="linux")]
                    {use std::os::unix::fs::OpenOptionsExt;*direct_file=Some(std::fs::OpenOptions::new().read(true).custom_flags(0x4000).open(&job.path)?);}
                    #[cfg(not(target_os="linux"))]
                    return Err(std::io::Error::new(std::io::ErrorKind::Unsupported,"direct file reads require Linux"));
                }
                let source=direct_file.as_mut().unwrap();source.seek(SeekFrom::Start(absolute-prefix as u64))?;
                let mut read=0;
                while read<needed {
                    let count=source.read(&mut data[read..io_bytes])?;
                    if count==0{return Err(std::io::Error::new(std::io::ErrorKind::UnexpectedEof,"short direct model read"));}
                    read+=count;
                    if read<needed && read%4096!=0{return Err(std::io::Error::new(std::io::ErrorKind::InvalidData,"unaligned short direct model read"));}
                }
                Ok(())
            });
            match attempt {
                Ok(())=>{data.prefix=prefix;data.length=bytes;direct_done=true;},
                Err(error) if file_io==FileIo::Auto && direct_unavailable(&error)=>{direct_disabled=true;direct_file.take();},
                Err(error)=>return Err(format!("direct model read {}: {error}",job.path.display())),
            }
        }
        if !direct_done && mapped {
            // The uploader unregisters before recycling this lease. Release
            // its previous mapping before faulting another bounded chunk.
            data.mapped.take();
            #[cfg(target_os="linux")]
            {data.mapped=Some(crate::startup_trace::measure("mmap_prefetch",job.path.to_string_lossy(),bytes,||
                mapping::read(&file,job.file_offset.checked_add(offset as u64).ok_or("mapped model range overflow")?,bytes))?);}
            #[cfg(not(target_os="linux"))]
            return Err("mapped model reads require Linux".into());
            #[cfg(target_os="linux")]
            {data.length=bytes;}
        }else if !direct_done {
            data.resize(bytes,0);
            if file_io!=FileIo::Buffered{file.seek(SeekFrom::Start(absolute)).map_err(|e|e.to_string())?;}
            crate::startup_trace::measure("read_file",job.path.to_string_lossy(),data.len(),||file.read_exact(&mut data)).map_err(|e|format!("model read {}: {e}",job.path.display()))?;
        }
        if job.finite_bf16 && crate::startup_trace::measure("reader_finite_check",job.path.to_string_lossy(),data.len(),||
            data.chunks_exact(2).any(|v|u16::from_le_bytes([v[0],v[1]])&0x7f80==0x7f80)) {
            return Err("nonfinite post-training weight".into());
        }
        if let Some(guard)=&mut guard{guard.observe(offset,&data);}
        // All bytes are hashed once, including small final chunks.
        if verify_sha{crate::startup_trace::measure("sha256",job.path.to_string_lossy(),data.len(),||hash.update(&*data));}
        let size=data.len();
        crate::startup_trace::measure("reader_packet_send",job.path.to_string_lossy(),size,||
            send.send(Packet::Chunk {worker,job:index,offset,data})).map_err(|_|"model loader cancelled")?;
        offset+=size;
    }
    if job.file_bytes.is_some() || mapped || file_io!=FileIo::Buffered {
        if file.metadata().map_err(|e|e.to_string())?.len()!=file_bytes {return Err("packed model file size changed during read".into());}
    }else if file.read(&mut [0u8;1]).map_err(|e|e.to_string())?!=0 {return Err("model file grew during read".into());}
    if let Some(guard)=guard{crate::startup_trace::measure("reader_guard_finish",job.path.to_string_lossy(),0,||guard.finish()).map_err(|e|format!("{}: {e}",job.path.display()))?;}
    if let Some(expected)=job.sha256.as_ref().filter(|_|verify_sha) {
        if format!("{:x}",hash.finalize())!=*expected {return Err(format!("model SHA256 mismatch: {}",job.path.display()));}
    }
    Ok(())
}

/// CUDA state is owned by the calling thread, never the CPU readers.
#[cfg(feature="cuda")]
pub mod device {
    use super::{configured_upload_slot_count,AlignedStorage,Config,Packet,ReadChunk,ReaderPool,RetainedChunks,Result};
    use std::{ffi::{c_int,c_void,CStr,c_char},ptr};
    use std::sync::{mpsc,Arc};
    type Stream=*mut c_void;
    unsafe extern "C" {
        fn cudaHostAlloc(out:*mut *mut c_void,bytes:usize,flags:u32)->c_int;
        fn cudaFreeHost(ptr:*mut c_void)->c_int;
        fn cudaHostRegister(ptr:*mut c_void,bytes:usize,flags:u32)->c_int;
        fn cudaHostUnregister(ptr:*mut c_void)->c_int;
        fn cudaGetDevice(device:*mut c_int)->c_int;
        fn cudaDeviceGetAttribute(value:*mut c_int,attribute:c_int,device:c_int)->c_int;
        fn cudaEventCreateWithFlags(out:*mut *mut c_void,flags:u32)->c_int;
        fn cudaEventRecord(event:*mut c_void,stream:Stream)->c_int;
        fn cudaEventSynchronize(event:*mut c_void)->c_int;
        fn cudaEventDestroy(event:*mut c_void)->c_int;
        fn cudaStreamSynchronize(stream:Stream)->c_int;
        fn cudaMemcpyAsync(dst:*mut c_void,src:*const c_void,bytes:usize,kind:c_int,stream:Stream)->c_int;
        fn cudaGetErrorString(code:c_int)->*const c_char;
    }
    fn check(code:c_int)->Result<()> {
        if code==0 {Ok(())}else{Err(format!("CUDA {code}: {}",unsafe{CStr::from_ptr(cudaGetErrorString(code))}.to_string_lossy()))}
    }
    struct Slot {host:*mut c_void,event:*mut c_void,busy:bool,mapped_registered:bool,pending:Option<(ReadChunk,mpsc::SyncSender<ReadChunk>)>}
    /// A resident cache can retain the original registrations with its RAM
    /// buffers. Evicting device tensors then needs only H2D, not re-pinning.
    pub struct Registrations {storage:Vec<Arc<AlignedStorage>>,context:usize}
    impl Registrations {
        pub fn bytes(&self)->usize{self.storage.iter().map(|storage|storage.layout.size()).sum()}
        pub(crate) fn context_identity(&self)->usize {self.context}
        pub fn pin(source:&RetainedChunks)->Result<Arc<Self>> {
            let context=crate::device_context::current_identity().map_err(|e|e.to_string())?;
            let mut registrations=Self{storage:Vec::new(),context};
            for (_,storage,_) in source.chunks.iter().flatten() {
                check(unsafe{cudaHostRegister(storage.pointer.cast(),storage.layout.size(),0)})?;
                registrations.storage.push(storage.clone());
            }
            Ok(Arc::new(registrations))
        }
    }
    impl Drop for Registrations {
        fn drop(&mut self) {
            for storage in self.storage.drain(..) {
                if unsafe{cudaHostUnregister(storage.pointer.cast())}!=0 {
                    // A failed unregister must not free a still registered page.
                    std::mem::forget(storage);
                }
            }
        }
    }
    /// Reusable buffers cross component streams. Register each allocation
    /// once, and retain it until every component upload has drained.
    pub struct SharedRegistrations {entries:std::sync::Mutex<Registrations>}
    impl SharedRegistrations {
        pub fn new()->Result<Arc<Self>> {
            let context=crate::device_context::current_identity().map_err(|e|e.to_string())?;
            Ok(Arc::new(Self{entries:std::sync::Mutex::new(Registrations{storage:Vec::new(),context})}))
        }
        fn register(&self,storage:&Arc<AlignedStorage>)->Result<()> {
            let mut entries=crate::startup_trace::measure("pinned_registration_lock","shared-reader",0,||
                self.entries.lock()).map_err(|_|"component registrations poisoned")?;
            if !entries.storage.iter().any(|s|Arc::ptr_eq(s,storage)) {
                crate::startup_trace::measure("pinned_register","shared-component-reader",storage.layout.size(),||
                    check(unsafe{cudaHostRegister(storage.pointer.cast(),storage.layout.size(),0)}))?;
                entries.storage.push(storage.clone());
            }
            Ok(())
        }
    }
    pub struct Upload {slots:Vec<Slot>,next:usize,chunk:usize,stream:Stream,direct:bool,registered:Vec<Arc<AlignedStorage>>,
        retained_registrations:Option<Arc<Registrations>>,shared_registrations:Option<Arc<SharedRegistrations>>,
        prefetch_stop:Option<std::path::PathBuf>}
    impl Upload {
        pub fn new(stream:Stream,config:Config)->Result<Self> {
            Self::with_policy(stream,config,false,false,None)
        }
        pub fn with_policy(stream:Stream,config:Config,mut direct:bool,mapped:bool,
            prefetch_stop:Option<std::path::PathBuf>)->Result<Self> {
            if mapped {
                let mut device=0;let mut supported=0;
                check(unsafe{cudaGetDevice(&mut device)})?;
                check(unsafe{cudaDeviceGetAttribute(&mut supported,113,device)})?;
                direct &= supported!=0;
                let policy=if direct{"readonly-direct"}else{"pinned-copy"};
                let _span=crate::startup_trace::Span::new("mapped_upload_policy",policy,0);
                eprintln!("native mapped upload: {policy}; readonly_supported={supported}; device={device}");
            }
            Self::new_options(stream,config,direct,prefetch_stop)
        }
        pub fn new_options(stream:Stream,config:Config,direct:bool,prefetch_stop:Option<std::path::PathBuf>)->Result<Self> {
            let slots=configured_upload_slot_count(config,direct)?;
            let _span=crate::startup_trace::Span::new("upload_queue",format!("direct={direct};inflight={slots};copy_slots={}",config.pinned_slots),0);
            let mut out=Self {slots:Vec::new(),next:0,chunk:config.chunk_bytes,stream,direct,registered:Vec::new(),
                retained_registrations:None,shared_registrations:None,prefetch_stop};
            for _ in 0..slots {
                let mut slot=Slot {host:ptr::null_mut(),event:ptr::null_mut(),busy:false,mapped_registered:false,pending:None};
                if !direct{check(unsafe{cudaHostAlloc(&mut slot.host,config.chunk_bytes,0)})?;}
                // Track allocation before event creation, including its error path.
                out.slots.push(slot);
                let slot=out.slots.last_mut().unwrap();
                crate::startup_trace::measure("cuda_event_create","upload",0,||check(unsafe{cudaEventCreateWithFlags(&mut slot.event,2)}))?;
            }
            Ok(out)
        }
        fn stop_prefetch(&mut self) {
            if let Some(path)=self.prefetch_stop.take() {
                // One small local notification after successful H2D submission.
                // Never join cache helpers or make their outcome gate uploads.
                crate::startup_trace::measure("file_cache_handoff",path.to_string_lossy(),0,|| {
                    match std::fs::OpenOptions::new().write(true).create_new(true).open(&path) {
                        Ok(_)=>{},
                        Err(e) if e.kind()==std::io::ErrorKind::AlreadyExists=>{},
                        Err(e)=>eprintln!("model cache prefetch stop notification failed: {e}"),
                    }
                });
            }
        }
        /// Caller guarantees dst contains at least data.len() writable bytes.
        pub unsafe fn copy(&mut self,dst:*mut c_void,data:&[u8])->Result<()> {
            if self.direct{return Err("direct upload requires an owned reader lease".into());}
            if data.len()>self.chunk {return Err("model upload exceeds chunk".into());}
            let slot=&mut self.slots[self.next];
            if slot.busy {crate::startup_trace::measure("copy_wait","event",0,||check(unsafe{cudaEventSynchronize(slot.event)}))?;slot.busy=false;}
            crate::startup_trace::measure("pinned_staging","",data.len(),||unsafe{ptr::copy_nonoverlapping(data.as_ptr(),slot.host.cast(),data.len());});
            // Even failure after a partially queued operation requires a stream
            // drain before the pinned allocation is released (Drop below).
            crate::startup_trace::measure("h2d_api","pinned-staging",data.len(),||check(unsafe{cudaMemcpyAsync(dst,slot.host,data.len(),1,self.stream)}))?;
            crate::startup_trace::measure("cuda_event_record","upload",0,||check(unsafe{cudaEventRecord(slot.event,self.stream)}))?;
            slot.busy=true;
            self.next=(self.next+1)%self.slots.len();
            self.stop_prefetch();
            Ok(())
        }
        pub fn finish(&self)->Result<()> {crate::startup_trace::measure("copy_wait","finish",0,||check(unsafe{cudaStreamSynchronize(self.stream)}))}
        pub fn retain_registrations(&mut self)->Result<Arc<Registrations>> {
            self.finish()?;
            let context=crate::device_context::current_identity().map_err(|e|e.to_string())?;
            let registrations=Arc::new(Registrations{storage:std::mem::take(&mut self.registered),context});
            self.retained_registrations=Some(registrations.clone());Ok(registrations)
        }
        pub fn use_registrations(&mut self,registrations:Arc<Registrations>)->Result<()> {
            if crate::device_context::current_identity().map_err(|e|e.to_string())?!=registrations.context {
                return Err("retained host registrations belong to another CUDA context".into());
            }
            self.retained_registrations=Some(registrations);Ok(())
        }
        pub fn use_shared_registrations(&mut self,registrations:Arc<SharedRegistrations>)->Result<()> {
            let context=registrations.entries.lock().map_err(|_|"component registrations poisoned")?.context;
            if crate::device_context::current_identity().map_err(|e|e.to_string())?!=context {
                return Err("component registrations belong to another context".into());
            }
            self.shared_registrations=Some(registrations);Ok(())
        }
        /// # Safety
        /// The destination range must be live on this upload's stream. Host
        /// leases remain owned through the last DMA event, including errors.
        pub unsafe fn copy_retained(&mut self,dst:*mut c_void,source:&RetainedChunks,job:usize)->Result<()> {
            for (offset,storage,bytes) in source.chunks.get(job).ok_or("retained tensor index")? {
                let data=ReadChunk{ordinary:Vec::new(),aligned:Some(storage.clone()),mapped:None,prefix:0,length:*bytes,target:None};
                let (recycle,closed)=mpsc::sync_channel(0);drop(closed);
                unsafe{self.enqueue(dst.cast::<u8>().add(*offset).cast(),data,0,*bytes,recycle)}?;
            }
            Ok(())
        }
        fn reclaim(&mut self,index:usize)->Result<bool> {
            let slot=&mut self.slots[index];
            if !slot.busy{return Ok(false);}
            crate::startup_trace::measure("copy_wait","event",0,||check(unsafe{cudaEventSynchronize(slot.event)}))?;slot.busy=false;
            if slot.mapped_registered {
                let storage=slot.pending.as_ref().and_then(|(data,_)|data.mapped.as_ref()).ok_or("registered model mapping lost its lease")?;
                crate::startup_trace::measure("pinned_unregister","mapped-reader",storage.bytes,||check(unsafe{cudaHostUnregister(storage.pointer.cast())}))?;
                slot.mapped_registered=false;
            }
            if let Some((data,recycle))=slot.pending.take(){let _=crate::startup_trace::measure("upload_recycle","reader-lease",0,||recycle.send(data));}
            Ok(true)
        }
        fn reclaim_one(&mut self)->Result<bool> {
            for i in 0..self.slots.len(){let index=(self.next+i)%self.slots.len();if self.reclaim(index)?{return Ok(true);}}
            Ok(false)
        }
        /// Upload `bytes` of `data` starting at `skip`; the lease lives until that copy completes.
        unsafe fn enqueue(&mut self,dst:*mut c_void,data:ReadChunk,skip:usize,bytes:usize,recycle:mpsc::SyncSender<ReadChunk>)->Result<()> {
            if skip.checked_add(bytes).is_none_or(|end|end>data.len()) || bytes==0 {
                let _=recycle.send(data);return Err("model upload range exceeds its chunk".into());
            }
            if !self.direct {
                let result=unsafe{self.copy(dst,&data[skip..skip+bytes])};let _=recycle.send(data);return result;
            }
            if data.len()>self.chunk{return Err("model upload exceeds chunk".into());}
            self.reclaim(self.next)?;
            let mapped=data.mapped.is_some();
            if let Some(storage)=&data.mapped {
                // cudaHostRegisterReadOnly. Never ask CUDA to write-pin a
                // read-only page-cache mapping or silently fall back to a copy.
                crate::startup_trace::measure("pinned_register","mapped-reader",storage.bytes,||check(unsafe{cudaHostRegister(storage.pointer.cast(),storage.bytes,8)}))?;
            }else{
                let storage=data.aligned.as_ref().ok_or("direct reader allocation required")?;
                if let Some(shared)=&self.shared_registrations {shared.register(storage)?;}
                else if !self.registered.iter().any(|p|Arc::ptr_eq(p,storage)) && !self.retained_registrations.as_ref()
                    .is_some_and(|r|r.storage.iter().any(|p|Arc::ptr_eq(p,storage))) {
                    crate::startup_trace::measure("pinned_register","direct-reader",storage.layout.size(),||check(unsafe{cudaHostRegister(storage.pointer.cast(),storage.layout.size(),0)}))?;
                    self.registered.push(storage.clone());
                }
            }
            let source=unsafe{data.as_ptr().add(skip)}.cast();
            let slot=&mut self.slots[self.next];
            // Transfer ownership before CUDA can retain the host pointer, even
            // if copy or event recording subsequently fails.
            slot.pending=Some((data,recycle));slot.busy=true;slot.mapped_registered=mapped;
            crate::startup_trace::measure("h2d_api",if mapped{"mapped-pinned"}else{"direct-pinned"},bytes,||check(unsafe{cudaMemcpyAsync(dst,source,bytes,1,self.stream)}))?;
            crate::startup_trace::measure("cuda_event_record","upload",0,||check(unsafe{cudaEventRecord(slot.event,self.stream)}))?;
            self.next=(self.next+1)%self.slots.len();self.stop_prefetch();Ok(())
        }
        /// Validation-only full D2H hash; excluded from timed performance arms.
        pub unsafe fn verify(&self,src:*const c_void,bytes:usize,expected:&str)->Result<()> {
            use sha2::{Digest,Sha256};
            self.finish()?;
            let mut data=vec![0;self.chunk.min(bytes)];
            let mut hash=Sha256::new();
            let mut offset=0;
            while offset<bytes {
                let size=data.len().min(bytes-offset);
                let result=check(unsafe{cudaMemcpyAsync(data.as_mut_ptr().cast(),src.cast::<u8>().add(offset).cast(),size,2,self.stream)});
                let synced=self.finish();result?;synced?;
                hash.update(&data[..size]);offset+=size;
            }
            if format!("{:x}",hash.finalize())!=expected {return Err("uploaded model SHA256 mismatch".into());}
            Ok(())
        }
    }
    impl Drop for Upload {
        fn drop(&mut self) {
            // cudaFreeHost also synchronizes pending use; drain explicitly first
            // on every success/error path, while the arena stream is still alive.
            unsafe{cudaStreamSynchronize(self.stream);}
            for slot in &mut self.slots {unsafe {
                if slot.mapped_registered {
                    if let Some(storage)=slot.pending.as_ref().and_then(|(data,_)|data.mapped.as_ref()) {
                        cudaHostUnregister(storage.pointer.cast());
                    }
                }
                slot.pending.take();
                if !slot.event.is_null(){cudaEventDestroy(slot.event);}
                if !slot.host.is_null(){cudaFreeHost(slot.host);}
            }}
            for storage in &self.registered{unsafe{cudaHostUnregister(storage.pointer.cast());}}
        }
    }
    impl ReaderPool {
        /// Own the reader lease until the corresponding CUDA event completes.
        /// The destination callback only computes an already allocated address.
        pub unsafe fn consume_upload(&mut self,upload:&mut Upload,mut destination:impl FnMut(usize,usize)->*mut c_void)->Result<()> {
            unsafe{self.consume_upload_ranges(upload,|job,offset,bytes|Some((destination(job,offset),0,bytes)))}
        }
        /// For owners that keep only part of some files: `destination(job, offset, chunk_bytes)`
        /// gives the device address, the first chunk byte and the byte count to copy, or None
        /// to upload nothing from that chunk. Every byte is still read and verified.
        pub unsafe fn consume_upload_ranges(&mut self,upload:&mut Upload,
            mut destination:impl FnMut(usize,usize,usize)->Option<(*mut c_void,usize,usize)>)->Result<()> {
            if !upload.direct{return self.consume_all(|job,offset,data|match destination(job,offset,data.len()) {
                Some((dst,skip,bytes))=>{
                    if skip.checked_add(bytes).is_none_or(|end|end>data.len()) || bytes==0 {return Err("model upload range exceeds its chunk".into());}
                    unsafe{upload.copy(dst,&data[skip..skip+bytes])}
                },
                None=>Ok(()),
            });}
            let mut offsets=vec![0usize;self.jobs.len()];let mut verified=vec![false;self.jobs.len()];
            let mut remaining=self.jobs.len();
            while remaining>0 {
                let receiver=self.packets.as_ref().ok_or("model loader cancelled")?;
                let packet=match receiver.try_recv() {
                    Ok(packet)=>packet,
                    Err(mpsc::TryRecvError::Empty)=>{
                        // With one reader and more event slots than buffers,
                        // waiting for another packet would deadlock without this.
                        if upload.reclaim_one()?{continue;}
                        crate::startup_trace::measure("upload_wait_reader","direct",0,||receiver.recv()).map_err(|_|"model reader terminated before completion")?
                    },
                    Err(mpsc::TryRecvError::Disconnected)=>return Err("model reader terminated before completion".into()),
                };
                match packet {
                    Packet::Chunk{worker,job,offset,data}=>{
                        if worker>=self.workers.len()||job>=self.jobs.len()||verified[job]||offset!=offsets[job]{return Err("model chunk order mismatch".into());}
                        offsets[job]=offset.checked_add(data.len()).filter(|n|*n<=self.jobs[job].bytes).ok_or("model chunk extent mismatch")?;
                        self.retain_chunk(job,offset,&data)?;
                        let recycle=self.workers[worker].recycle.as_ref().ok_or("model loader cancelled")?.clone();
                        match destination(job,offset,data.len()) {
                            Some((dst,skip,bytes))=>unsafe{upload.enqueue(dst,data,skip,bytes,recycle)}?,
                            None=>{let _=recycle.send(data);},
                        }
                    },
                    Packet::Done{job,result}=>{
                        if job>=self.jobs.len()||verified[job]{return Err("model completion order mismatch".into());}
                        result?;
                        if offsets[job]!=self.jobs[job].bytes{return Err("model upload byte count mismatch".into());}
                        verified[job]=true;remaining-=1;
                    },
                }
            }
            upload.finish()?;
            for index in 0..upload.slots.len(){upload.reclaim(index)?;}
            self.complete=true;Ok(())
        }
        /// consume_upload, then `after(job, offset, bytes)` once each chunk's copy is queued on the
        /// upload stream (e.g. a kernel that converts the destination into its final tensor). Stream
        /// order keeps a reused destination behind its previous consumer. The loop is kept separate
        /// from consume_upload so existing callers run exactly the code they were accepted with.
        pub unsafe fn consume_upload_then(&mut self,upload:&mut Upload,mut destination:impl FnMut(usize,usize)->*mut c_void,
            mut after:impl FnMut(usize,usize,usize)->Result<()>)->Result<()> {
            if !upload.direct{return self.consume_all(|job,offset,data|{
                unsafe{upload.copy(destination(job,offset),data)}?;after(job,offset,data.len())});}
            let mut offsets=vec![0usize;self.jobs.len()];let mut verified=vec![false;self.jobs.len()];
            let mut remaining=self.jobs.len();
            while remaining>0 {
                let receiver=self.packets.as_ref().ok_or("model loader cancelled")?;
                let packet=match receiver.try_recv() {
                    Ok(packet)=>packet,
                    Err(mpsc::TryRecvError::Empty)=>{
                        if upload.reclaim_one()?{continue;}
                        crate::startup_trace::measure("upload_wait_reader","direct",0,||receiver.recv()).map_err(|_|"model reader terminated before completion")?
                    },
                    Err(mpsc::TryRecvError::Disconnected)=>return Err("model reader terminated before completion".into()),
                };
                match packet {
                    Packet::Chunk{worker,job,offset,data}=>{
                        if worker>=self.workers.len()||job>=self.jobs.len()||verified[job]||offset!=offsets[job]{return Err("model chunk order mismatch".into());}
                        offsets[job]=offset.checked_add(data.len()).filter(|n|*n<=self.jobs[job].bytes).ok_or("model chunk extent mismatch")?;
                        self.retain_chunk(job,offset,&data)?;
                        let recycle=self.workers[worker].recycle.as_ref().ok_or("model loader cancelled")?.clone();
                        let bytes=data.len();
                        unsafe{upload.enqueue(destination(job,offset),data,0,bytes,recycle)}?;
                        after(job,offset,bytes)?;
                    },
                    Packet::Done{job,result}=>{
                        if job>=self.jobs.len()||verified[job]{return Err("model completion order mismatch".into());}
                        result?;
                        if offsets[job]!=self.jobs[job].bytes{return Err("model upload byte count mismatch".into());}
                        verified[job]=true;remaining-=1;
                    },
                }
            }
            upload.finish()?;
            for index in 0..upload.slots.len(){upload.reclaim(index)?;}
            self.complete=true;Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn direct_dma_depth_does_not_allocate_a_second_host_ring(){
        let config=Config{workers:6,chunk_bytes:32<<20,pinned_slots:1,budget_bytes:416<<20};
        assert_eq!(upload_slot_count(config,true,None).unwrap(),1);
        assert_eq!(upload_slot_count(config,false,None).unwrap(),1);
        assert_eq!(upload_slot_count(config,true,Some(4)).unwrap(),4);
        assert_eq!(config.budget_bytes,416<<20);
        assert!(upload_slot_count(config,false,Some(4)).is_err());
        assert!(upload_slot_count(config,true,Some(0)).is_err());
        assert!(upload_slot_count(config,true,Some(17)).is_err());
        let too_small=Config{budget_bytes:(416<<20)-1,..config};
        assert!(upload_slot_count(too_small,true,Some(4)).is_err());
    }
    #[test] fn retained_component_readers_fill_the_final_owned_storage_with_one_worker_budget(){
        let f=fixture();let data:Vec<u8>=(0..17003).map(|n|(n%251) as u8).collect();
        let mut config=Config{workers:2,chunk_bytes:4096,pinned_slots:1,budget_bytes:1<<20};
        let jobs=(0..5).map(|stage|vec![job(&f,&format!("snapshot-{stage}"),&data)]).collect::<Vec<_>>();
        let bound=read_ahead_bound(&vec![data.len();5],config,true,5).unwrap();
        config.budget_bytes=bound-1;
        assert!(ReaderPool::start_retained_components(jobs.clone(),config,true,false).is_err());
        config.budget_bytes=bound;
        let pools=ReaderPool::start_retained_components(jobs,config,true,false).unwrap();
        assert_eq!(pools[0].shared.as_ref().unwrap().threads.lock().unwrap().len(),2);
        for pool in &pools {assert!(Arc::ptr_eq(pool.shared.as_ref().unwrap(),pools[0].shared.as_ref().unwrap()));}
        let results=thread::scope(|scope|pools.into_iter().map(|mut pool|scope.spawn(move||{
            let mut addresses=Vec::new();
            pool.consume_all(|_,_,bytes|{addresses.push(bytes.as_ptr() as usize);Ok(())}).unwrap();
            let host=pool.take_retained().unwrap().unwrap();
            assert_eq!(addresses,host.chunks[0].iter().map(|(_,storage,_)|storage.pointer as usize).collect::<Vec<_>>());
            host
        })).collect::<Vec<_>>().into_iter().map(|handle|handle.join().unwrap()).collect::<Vec<_>>());
        for host in results {assert_eq!(host.read(0).unwrap(),data);}
    }
    #[test] fn retained_takeover_bound_includes_each_aligned_tail(){
        assert_eq!(retained_copy_bound(&[5001,0,8192],4096).unwrap(),16384);
        assert_eq!(retained_copy_bound(&[6001],3000).unwrap(),12288);
        assert!(retained_copy_bound(&[1],0).is_err());
        assert!(retained_copy_bound(&[usize::MAX],4096).is_err());
    }
    #[test] fn retained_takeover_consumes_the_existing_open_file_without_restarting(){
        let f=fixture();let data:Vec<u8>=(0..65037).map(|n|(n%251) as u8).collect();
        let task=job(&f,"takeover",&data);
        let cfg=Config{workers:1,chunk_bytes:4096,pinned_slots:1,budget_bytes:16384};
        let mut pool=ReaderPool::start_options(vec![task.clone()],cfg,true,true).unwrap();
        // Receipt of its first chunk proves this exact file is already open.
        // Keep that packet in order while unlinking the source. Any restart
        // or second file open would now fail instead of passing this test.
        let packets=pool.packets.take().unwrap();
        let first=packets.recv_timeout(std::time::Duration::from_secs(3)).unwrap();
        assert!(matches!(&first,Packet::Chunk{offset:0,..}));
        std::fs::remove_file(&task.path).unwrap();
        let (send,receive)=mpsc::sync_channel(1);pool.packets=Some(receive);
        let forward=thread::spawn(move||{
            if send.send(first).is_ok(){for packet in packets{if send.send(packet).is_err(){break;}}}
        });
        let bound=retained_copy_bound(&[data.len()],cfg.chunk_bytes).unwrap();
        let host=pool.finish_retained_copy(bound).unwrap();forward.join().unwrap();
        assert_eq!(host.read(0).unwrap(),data);
    }
    #[test] fn retained_takeover_drains_all_shared_components_and_rejects_corruption(){
        let f=fixture();let data:Vec<u8>=(0..17003).map(|n|(n%251) as u8).collect();
        let cfg=Config{workers:2,chunk_bytes:4096,pinned_slots:1,budget_bytes:32768};
        let stages=(0..5).map(|i|vec![job(&f,&format!("takeover-{i}"),&data)]).collect();
        let pools=ReaderPool::start_components(stages,cfg,true,true,false).unwrap();
        let bound=retained_copy_bound(&[data.len()],cfg.chunk_bytes).unwrap();
        let hosts=thread::scope(|scope|{
            let consumers=pools.into_iter().map(|pool|scope.spawn(move||pool.finish_retained_copy(bound))).collect::<Vec<_>>();
            consumers.into_iter().map(|h|h.join().unwrap().unwrap()).collect::<Vec<_>>()
        });
        for host in hosts{assert_eq!(host.read(0).unwrap(),data);}
        let task=job(&f,"takeover-bad",&data);let mut bad=data.clone();bad[3000]^=1;std::fs::write(&task.path,bad).unwrap();
        let pool=ReaderPool::start_options(vec![task],cfg,true,true).unwrap();
        assert!(pool.finish_retained_copy(bound).is_err());
        let task=job(&f,"takeover-budget",&data);
        let pool=ReaderPool::start_options(vec![task],cfg,true,true).unwrap();
        assert!(pool.finish_retained_copy(4095).is_err());
    }
    #[test] fn first_upload_retains_independent_copies_after_reader_buffers_recycle(){
        let f=fixture();let data:Vec<u8>=(0..129037).map(|n|((n*17+n/4096)%251) as u8).collect();
        let cfg=Config{workers:1,chunk_bytes:4096,pinned_slots:1,budget_bytes:16384};
        for aligned in [false,true] {
            let task=job(&f,&format!("upload-retain-{aligned}"),&data);
            let mut pool=ReaderPool::start_options(vec![task.clone()],cfg,aligned,true).unwrap();
            let bound=retained_copy_bound(&[data.len()],cfg.chunk_bytes).unwrap();
            assert!(pool.retain_copies(bound-1).is_err());
            pool.retain_copies(bound).unwrap();
            assert!(pool.take_retained().is_err());
            let mut received=Vec::new();let mut unlinked=false;
            pool.consume_all(|job,offset,bytes| {
                assert_eq!(job,0);assert_eq!(offset,received.len());
                if !unlinked {std::fs::remove_file(&task.path).unwrap();unlinked=true;}
                received.extend_from_slice(bytes);Ok(())
            }).unwrap();
            assert_eq!(received,data);
            let host=pool.take_retained().unwrap().unwrap();
            // More than thirty chunks reused a two-buffer reader ring. An
            // alias of that ring would contain the final chunks here.
            assert_eq!(host.read(0).unwrap(),data);
            assert!(pool.take_retained().unwrap().is_none());
        }
    }
    #[test] fn failed_first_upload_never_publishes_retained_copies(){
        let f=fixture();let data=vec![7u8;17003];let task=job(&f,"upload-retain-bad",&data);
        let mut bad=data.clone();bad[8193]^=1;std::fs::write(&task.path,bad).unwrap();
        let cfg=Config{workers:1,chunk_bytes:4096,pinned_slots:1,budget_bytes:16384};
        let mut pool=ReaderPool::start_options(vec![task],cfg,true,true).unwrap();
        pool.retain_copies(retained_copy_bound(&[data.len()],cfg.chunk_bytes).unwrap()).unwrap();
        assert!(pool.consume_all(|_,_,_|Ok(())).is_err());
        assert!(pool.take_retained().is_err());
    }
    #[cfg(target_os="linux")]
    #[test]fn direct_file_reads_preserve_unaligned_tensor_boundaries_and_eof(){
        let f=fixture();let payload:Vec<u8>=(0..65037).map(|n|(n%251) as u8).collect();
        let mut file=vec![9;37];file.extend_from_slice(&payload);
        std::fs::write(f.0.join("direct-packed"),&file).unwrap();
        let task=Job::slice(&f.0,"direct-packed",37,payload.len(),format!("{:x}",Sha256::digest(&payload)),file.len() as u64).unwrap();
        let (free,receive)=mpsc::sync_channel(2);let (send,packets)=mpsc::sync_channel(2);
        for _ in 0..2{free.send(ReadChunk::reader(8192,true,FileIo::Direct).unwrap()).unwrap();}
        let mut worker=Some(std::thread::spawn(move||{
            let result=read_job_cancel(0,0,&task,8192,&receive,&send,true,false,false,FileIo::Direct,None,None,None);
            send.send(Packet::Done{job:0,result}).unwrap();
        }));
        let mut actual=Vec::new();
        loop{match packets.recv().unwrap(){
            Packet::Chunk{offset,data,..}=>{
                assert_eq!(offset,actual.len());assert_eq!(data.len(),8192.min(65037-offset));actual.extend_from_slice(&data);
                if actual.len()==payload.len(){
                    // Force the normal terminal race: Done is queued and the
                    // reader has exited before this unused final lease returns.
                    worker.take().unwrap().join().unwrap();
                    assert!(free.send(data).is_err());
                }else{free.send(data).unwrap();}
            },
            Packet::Done{result,..}=>{result.unwrap();break;},
        }}
        assert!(worker.is_none());assert_eq!(actual,payload);
    }
    #[cfg(target_os="linux")]
    #[test]fn cache_probe_handles_unaligned_ranges_without_touching_payloads(){
        let f=fixture();let bytes:Vec<u8>=(0..65037).map(|n|(n%251) as u8).collect();
        let path=f.0.join("cached-packed");std::fs::write(&path,&bytes).unwrap();
        let source=File::open(&path).unwrap();let mut bitmap=Vec::new();
        assert!(mapping::resident(&source,37,bytes.len()-37,&mut bitmap).unwrap());
        assert_eq!(std::fs::read(path).unwrap(),bytes);
    }
    #[test]fn component_reader_quotas_cannot_multiply_the_aggregate_budget(){
        let parent=Config{workers:8,chunk_bytes:32<<20,pinned_slots:4,budget_bytes:4352<<20};
        let configs=component_reader_configs(parent,&[24,4,24,4,2],true,FileReadPolicy::Direct).unwrap();
        assert_eq!(configs.iter().map(|c|c.budget_bytes).sum::<usize>(),parent.budget_bytes);
        assert_eq!(configs.iter().map(|c|c.workers).sum::<usize>(),58);
        assert!(component_reader_configs(Config{budget_bytes:4351<<20,..parent},&[24,4,24,4,2],true,FileReadPolicy::Direct).is_err());
        assert!(component_reader_configs(parent,&[0,24],true,FileReadPolicy::Direct).is_err());
        assert!(component_reader_configs(parent,&[65],true,FileReadPolicy::Direct).is_err());
        assert!(component_reader_configs(parent,&[],true,FileReadPolicy::Direct).is_err());
    }
    #[test]fn selectable_reader_schedules_keep_indices_guards_and_one_worker_budget(){
        assert!(ReadSchedule::parse("unknown").is_err());
        let f=fixture();let bytes:Vec<u8>=(0..65037).map(|n|(n%251)as u8).collect();
        let path=f.0.join("scheduled-packed");std::fs::write(&path,&bytes).unwrap();
        let slices=[vec![(37usize,19037usize),(107,703)],vec![(11037usize,29007usize),(507,4003)]];
        let expected=slices.iter().map(|ranges|ranges.iter().map(|&(offset,count)|bytes[offset..offset+count].to_vec()).collect::<Vec<_>>()).collect::<Vec<_>>();
        let jobs=slices.iter().map(|ranges|ranges.iter().map(|&(offset,count)|
            Job::slice(&f.0,"scheduled-packed",offset as u64,count,format!("{:x}",Sha256::digest(&bytes[offset..offset+count])),bytes.len()as u64).unwrap())
            .collect::<Vec<_>>()).collect::<Vec<_>>();
        let cfg=Config{workers:2,chunk_bytes:8192,pinned_slots:1,budget_bytes:8*8192};
        for schedule in [ReadSchedule::ByteFair,ReadSchedule::RoundRobin,ReadSchedule::FileOffset] {
            for policy in [FileReadPolicy::Buffered,FileReadPolicy::Direct] {
            for corrupt in [false,true] {
                let mut input=bytes.clone();if corrupt{input[11037+8193]^=1;}std::fs::write(&path,&input).unwrap();
                let pools=ReaderPool::start_components_scheduled(jobs.clone(),cfg,true,true,false,policy,schedule).unwrap();
                let results=std::thread::scope(|scope|{
                    let workers=pools.into_iter().map(|mut pool|scope.spawn(move||{
                        let mut actual=vec![Vec::new();2];pool.consume_all(|index,offset,data|{
                            assert_eq!(offset,actual[index].len());actual[index].extend_from_slice(data);Ok(())
                        }).map(|_|actual)
                    })).collect::<Vec<_>>();
                    workers.into_iter().map(|worker|worker.join().unwrap()).collect::<Vec<_>>()
                });
                if corrupt{assert!(results.iter().any(Result::is_err));}
                else{assert_eq!(results.into_iter().collect::<Result<Vec<_>>>().unwrap(),expected);}
            }
            }
        }
    }
    #[test]fn packed_file_offset_schedule_reads_physical_ranges_in_order() {
        let f=fixture();let bytes:Vec<u8>=(0..65037).map(|n|(n%251)as u8).collect();
        std::fs::write(f.0.join("physical-order"),&bytes).unwrap();
        let ranges=[(50000usize,8192usize),(7,12288),(25000,16384)];
        let jobs=ranges.iter().map(|&(offset,count)|Job::slice(&f.0,"physical-order",offset as u64,count,
            format!("{:x}",Sha256::digest(&bytes[offset..offset+count])),bytes.len()as u64).unwrap()).collect();
        let config=Config{workers:1,chunk_bytes:8192,pinned_slots:1,budget_bytes:32768};
        let mut pool=ReaderPool::start_components_scheduled(vec![jobs],config,true,true,false,FileReadPolicy::Direct,ReadSchedule::FileOffset).unwrap().pop().unwrap();
        let mut observed=Vec::new();pool.consume_all(|index,offset,data|{
            if offset==0{observed.push(index);}
            let (start,count)=ranges[index];assert!(offset+data.len()<=count);assert_eq!(data,&bytes[start+offset..start+offset+data.len()]);Ok(())
        }).unwrap();assert_eq!(observed,vec![1,2,0]);
    }
    #[test]fn declared_buffered_components_preserve_ranges_and_reject_corrupt_bytes(){
        let f=fixture();let payload:Vec<u8>=(0..65037).map(|n|(n%251)as u8).collect();
        let mut raw=vec![9;37];raw.extend_from_slice(&payload);raw.extend_from_slice(&[17;91]);
        std::fs::write(f.0.join("declared-buffered"),&raw).unwrap();
        let task=Job::slice(&f.0,"declared-buffered",37,payload.len(),format!("{:x}",Sha256::digest(&payload)),raw.len()as u64).unwrap();
        let cfg=Config{workers:1,chunk_bytes:8192,pinned_slots:1,budget_bytes:32768};
        let payloads=vec![payload.clone(),payload[8193..49013].to_vec(),payload[51..707].to_vec()];
        let tasks=vec![task.clone(),
            Job::slice(&f.0,"declared-buffered",37+8193,payloads[1].len(),format!("{:x}",Sha256::digest(&payloads[1])),raw.len()as u64).unwrap(),
            Job::slice(&f.0,"declared-buffered",37+51,payloads[2].len(),format!("{:x}",Sha256::digest(&payloads[2])),raw.len()as u64).unwrap()];
        let mut pool=ReaderPool::start_components_with_policy(vec![tasks],cfg,true,true,false,FileReadPolicy::Buffered).unwrap().pop().unwrap();
        let mut actual=vec![Vec::new();payloads.len()];pool.consume_all(|job,offset,data|{
            assert_eq!(offset,actual[job].len());assert_eq!(data.len(),8192.min(payloads[job].len()-offset));
            actual[job].extend_from_slice(data);Ok(())
        }).unwrap();assert_eq!(actual,payloads);
        raw[37+8193]^=1;std::fs::write(&task.path,&raw).unwrap();
        let mut pool=ReaderPool::start_components_with_policy(vec![vec![task]],cfg,true,true,false,FileReadPolicy::Buffered).unwrap().pop().unwrap();
        assert!(pool.consume_all(|_,_,_|Ok(())).is_err());
    }
    #[test]fn legacy_packed_readers_preserve_configured_chunk_boundaries(){
        let f=fixture();let payload:Vec<u8>=(0..19037).map(|n|(n%251) as u8).collect();
        let mut file=vec![9;37];file.extend_from_slice(&payload);std::fs::write(f.0.join("legacy-packed"),&file).unwrap();
        let task=Job::slice(&f.0,"legacy-packed",37,payload.len(),format!("{:x}",Sha256::digest(&payload)),file.len() as u64).unwrap();
        let cfg=Config{workers:1,chunk_bytes:8192,pinned_slots:1,budget_bytes:32768};
        let mut pool=ReaderPool::start_scheduled_mode(vec![task],cfg,true,true,false,true,false).unwrap();
        let mut actual=Vec::new();
        pool.consume_all(|_,offset,data|{assert_eq!(offset,actual.len());assert_eq!(data.len(),8192.min(payload.len()-offset));actual.extend_from_slice(data);Ok(())}).unwrap();
        assert_eq!(actual,payload);
    }
    #[test]fn independent_components_share_bounded_workers_and_make_concurrent_progress(){
        let f=fixture();let data:Vec<u8>=(0..50137).map(|n|(n%251) as u8).collect();
        let cfg=Config{workers:3,chunk_bytes:4096,pinned_slots:2,budget_bytes:32768};
        let stages=(0..3).map(|s|(0..2).map(|i|job(&f,&format!("component-{s}-{i}"),&data)).collect()).collect();
        let pools=ReaderPool::start_components(stages,cfg,true,true,false).unwrap();
        assert_eq!(pools[0].shared.as_ref().unwrap().threads.lock().unwrap().len(),3);
        assert!(pools.iter().all(|p|Arc::ptr_eq(p.shared.as_ref().unwrap(),pools[0].shared.as_ref().unwrap())));
        std::thread::scope(|scope|{
            let expected=&data;
            let workers=pools.into_iter().map(|mut pool|scope.spawn(move||{
                let mut actual=vec![Vec::new(),Vec::new()];
                pool.consume_all(|job,_,data|{actual[job].extend_from_slice(data);Ok(())}).unwrap();
                for value in actual{assert_eq!(&value,expected);}
            })).collect::<Vec<_>>();
            for worker in workers{worker.join().unwrap();}
        });
    }
    #[test]fn small_component_finishes_while_large_consumer_is_paused_with_one_shared_budget(){
        let f=fixture();let large=vec![7;1<<20];let small=vec![11;4096];
        let stages=vec![(0..32).map(|i|job(&f,&format!("large-{i}"),&large)).collect(),
            (0..32).map(|i|job(&f,&format!("small-{i}"),&small)).collect()];
        let cfg=Config{workers:2,chunk_bytes:4096,pinned_slots:1,budget_bytes:20480};
        let mut pools=ReaderPool::start_components(stages,cfg,true,true,false).unwrap();
        assert_eq!(pools[0].shared.as_ref().unwrap().threads.lock().unwrap().len(),2);
        let mut small_pool=pools.pop().unwrap();let mut large_pool=pools.pop().unwrap();
        let (release,wait)=mpsc::channel();let (complete,done)=mpsc::channel();
        std::thread::scope(|scope| {
            let large_worker=scope.spawn(move || {
                let mut wait=Some(wait);
                large_pool.consume_all(|_,_,_|{if let Some(wait)=wait.take(){wait.recv().map_err(|e|e.to_string())?;}Ok(())})
            });
            let small_worker=scope.spawn(move || {
                let mut actual=vec![Vec::new();32];
                let result=small_pool.consume_all(|index,_,data|{actual[index].extend_from_slice(data);Ok(())});
                complete.send(result.clone()).unwrap();
                result.unwrap();for value in actual {assert_eq!(value,small);}
            });
            let progress=done.recv_timeout(std::time::Duration::from_secs(2));
            // Always release and join both owners, including on a failed test.
            release.send(()).unwrap();large_worker.join().unwrap().unwrap();small_worker.join().unwrap();
            progress.expect("small component starved behind unrelated large tensors").unwrap();
        });
    }
    #[test]fn cancelling_one_component_releases_shared_recycle_waits(){
        let f=fixture();let data=vec![7;500137];
        let cfg=Config{workers:3,chunk_bytes:4096,pinned_slots:2,budget_bytes:32768};
        let stages=(0..3).map(|s|vec![job(&f,&format!("cancel-component-{s}"),&data)]).collect();
        let mut pools=ReaderPool::start_components(stages,cfg,true,true,false).unwrap();
        drop(pools.remove(0));
        for mut pool in pools{assert!(pool.consume_all(|_,_,_|Ok(())).is_err());}
    }
    #[test] fn staged_read_ahead_admits_every_chunk_and_ring_once() {
        let cfg=Config{workers:2,chunk_bytes:3,pinned_slots:2,budget_bytes:32768};
        assert_eq!(read_ahead_bound(&[7,2],cfg,false,2).unwrap(),21);
        assert_eq!(read_ahead_bound(&[7,2],cfg,true,2).unwrap(),4*4096+12);
        assert!(read_ahead_bound(&[usize::MAX],cfg,true,2).is_err());
        assert!(read_ahead_bound(&[7],cfg,false,usize::MAX).is_err());
        assert!(read_ahead_bound(&[7],cfg,false,0).is_err());
    }
    #[cfg(target_os="linux")]
    #[test] fn anonymous_reader_storage_is_initialized_aligned_and_keeps_its_lease() {
        for bytes in [1usize,4096,65543,(2<<20)+97] {
            let size=bytes.div_ceil(4096)*4096;
            let layout=std::alloc::Layout::from_size_align(size,4096).unwrap();
            let mut chunk=ReadChunk::aligned(bytes,layout,true).unwrap();
            assert_eq!(chunk.as_ptr() as usize%4096,0);
            assert!(chunk.iter().all(|&v|v==0));
            for (i,value) in chunk.iter_mut().enumerate(){*value=(i%251) as u8;}
            let lease=chunk.aligned.clone().unwrap();
            assert!(lease.anonymous);drop(chunk);
            let retained=unsafe{std::slice::from_raw_parts(lease.pointer,bytes)};
            assert!(retained.iter().enumerate().all(|(i,&v)|v==(i%251) as u8));
            drop(lease);
        }
    }
    #[test] fn buffered_reader_storage_starts_zeroed_and_preserves_small_allocation_bounds() {
        for bytes in [7usize,4096,2<<20] {
            let mut chunk=ReadChunk::new(bytes,false).unwrap();assert_eq!(chunk.len(),bytes);
            assert!(chunk.iter().all(|&value|value==0));chunk[bytes-1]=19;assert_eq!(chunk[bytes-1],19);
            if bytes<4096{assert!(chunk.aligned.is_none());assert_eq!(chunk.ordinary.len(),bytes);}
            if cfg!(target_os="linux") && bytes>=4096 && std::env::var_os("NATIVE_CORE_READ_LAZY_ZEROED").is_none(){
                assert!(chunk.aligned.as_ref().unwrap().anonymous);
            }
        }
    }
    #[test] fn tensor_views_are_aligned_disjoint_and_within_the_owned_allocation() {
        let sizes=[1,256,3,1048577];
        let (offsets,bytes)=tensor_allocation_layout(&sizes).unwrap();
        assert_eq!(offsets,vec![0,256,512,768]);
        assert_eq!(bytes,768+1048577);
        for (i,(&offset,&size)) in offsets.iter().zip(&sizes).enumerate() {
            assert_eq!(offset%256,0);
            assert!(offset+size<=bytes);
            if i>0 {assert!(offsets[i-1]+sizes[i-1]<=offset);}
        }
        assert_eq!(tensor_allocation_layout(&[]).unwrap(),(vec![],0));
        assert!(tensor_allocation_layout(&[0]).is_err());
        assert!(tensor_allocation_layout(&[usize::MAX,1]).is_err());
        assert!(tensor_allocation_layout(&[1,usize::MAX]).is_err());
    }
    #[test] fn tensor_ownership_accepts_slab_views_and_rejects_workspace_or_overflow() {
        let owned=[(4096,1024),(16384,256)];
        validate_tensor_allocations(&owned,&[(4096,256),(4352,512),(16384,256)]).unwrap();
        assert!(validate_tensor_allocations(&owned,&[(5000,200)]).is_err());
        assert!(validate_tensor_allocations(&owned,&[(32768,256)]).is_err());
        assert!(validate_tensor_allocations(&owned,&[(4096,0)]).is_err());
        assert!(validate_tensor_allocations(&owned,&[(usize::MAX,2)]).is_err());
        assert!(validate_tensor_allocations(&[(usize::MAX,2)],&[]).is_err());
    }
    #[test] fn tensor_views_keep_their_offsets_when_owned_allocations_move() {
        let owned=[(4096,1024),(16384,256)];
        let views=[(4096,256),(4352,512),(16384,256),(4096,256)];
        let locations=tensor_allocation_locations(&owned,&views).unwrap();
        assert_eq!(locations,vec![(0,0),(0,256),(1,0),(0,0)]);
        let restored=[(32768,1024),(65536,256)];
        let relocated:Vec<_>=locations.iter().zip(views).map(|(&(index,offset),(_,bytes))|
            (restored[index].0+offset,bytes)).collect();
        assert_eq!(relocated,vec![(32768,256),(33024,512),(65536,256),(32768,256)]);
        assert_eq!(tensor_allocation_locations(&restored,&relocated).unwrap(),locations);
        assert!(tensor_allocation_locations(&owned,&[(5000,200)]).is_err());
        assert!(tensor_allocation_locations(&[(usize::MAX,2)],&views).is_err());
    }
    use std::sync::atomic::{AtomicUsize,Ordering};
    static ID:AtomicUsize=AtomicUsize::new(0);
    struct Fixture(PathBuf);
    impl Drop for Fixture {fn drop(&mut self){let _=std::fs::remove_dir_all(&self.0);}}
    fn fixture()->Fixture {
        let p=std::env::temp_dir().join(format!("native-reader-{}-{}",std::process::id(),ID.fetch_add(1,Ordering::Relaxed)));
        std::fs::create_dir(&p).unwrap();Fixture(p)
    }
    fn config()->Config {Config{workers:3,chunk_bytes:257,pinned_slots:2,budget_bytes:2056}}
    fn job(f:&Fixture,name:&str,data:&[u8])->Job {
        std::fs::write(f.0.join(name),data).unwrap();
        Job::new(&f.0,name,data.len(),Some(format!("{:x}",Sha256::digest(data)))).unwrap()
    }
    struct HostBytes(Vec<std::cell::UnsafeCell<u8>>);
    unsafe impl Send for HostBytes {}
    unsafe impl Sync for HostBytes {}
    impl HostBytes {
        fn new(bytes:usize)->Arc<Self>{Arc::new(Self((0..bytes).map(|_|std::cell::UnsafeCell::new(0)).collect()))}
        fn pointer(&self)->*mut u8{self.0.as_ptr().cast::<u8>().cast_mut()}
        fn target(self:&Arc<Self>,offset:usize,bytes:usize)->HostTarget{
            assert!(offset+bytes<=self.0.len());unsafe{HostTarget::new(self.clone(),self.pointer().add(offset),bytes).unwrap()}
        }
    }
    fn guarded_job(f:&Fixture,name:&str,data:&[u8])->Job{
        let mut source=job(f,name,data);let edge=data.len().min(1024);
        source.guard=Some(light::Expected{bytes:data.len(),head_sha256:format!("{:x}",Sha256::digest(&data[..edge])),
            tail_sha256:format!("{:x}",Sha256::digest(&data[data.len()-edge..]))});source
    }
    #[test]fn workers_initialize_disjoint_final_host_ranges_without_staging_copies(){
        let f=fixture();let a=vec![17;8193];let b=vec![83;2049];let gap=17;let second=a.len()+gap;
        let storage=HostBytes::new(second+b.len());
        let mut pool=ReaderPool::start_into_guarded(vec![guarded_job(&f,"a",&a),guarded_job(&f,"b",&b)],config(),
            vec![storage.target(0,a.len()),storage.target(second,b.len())]).unwrap();
        pool.consume_all(|index,offset,data|{
            let base=if index==0{0}else{second};
            assert_eq!(data.as_ptr(),unsafe{storage.pointer().add(base+offset)});Ok(())
        }).unwrap();drop(pool);
        let actual=unsafe{std::slice::from_raw_parts(storage.pointer(),storage.0.len())};
        assert_eq!(&actual[..a.len()],a);assert!(actual[a.len()..second].iter().all(|b|*b==0));assert_eq!(&actual[second..],b);
        assert_eq!(Arc::strong_count(&storage),1);
    }
    #[test]fn direct_host_reader_yields_and_foreground_reuses_its_progress_and_guards(){
        let f=fixture();let gate=f.0.join("gate");std::fs::write(&gate,b"").unwrap();
        let scope=crate::background_io::Scope::enter(crate::background_io::Policy{gate:gate.clone(),cancel:f.0.join("cancel"),stop:None,pin_gate:None});
        let data=vec![29;4097];let storage=HostBytes::new(data.len());
        let mut pool=ReaderPool::start_into_guarded(vec![guarded_job(&f,"a",&data)],config(),vec![storage.target(0,data.len())]).unwrap();drop(scope);
        let mut consumed=0;
        assert_eq!(pool.consume_all(|_,offset,bytes|{assert_eq!(offset,consumed);consumed+=bytes.len();std::fs::remove_file(&gate).unwrap();Ok(())}).unwrap_err(),crate::background_io::PAUSED);
        pool.foreground(true);
        pool.consume_all(|_,offset,bytes|{assert_eq!(offset,consumed);consumed+=bytes.len();Ok(())}).unwrap();drop(pool);
        assert_eq!(consumed,data.len());assert_eq!(unsafe{std::slice::from_raw_parts(storage.pointer(),data.len())},data);
        assert_eq!(Arc::strong_count(&storage),1);
    }
    #[test]fn direct_host_reader_rejects_aliases_and_failed_guards(){
        let f=fixture();let data=vec![31;4097];let storage=HostBytes::new(data.len());
        let source=guarded_job(&f,"a",&data);
        assert!(ReaderPool::start_into_guarded(vec![source.clone(),source.clone()],config(),
            vec![storage.target(0,data.len()),storage.target(0,data.len())]).is_err());
        let mut broken=source;broken.guard.as_mut().unwrap().tail_sha256="0".repeat(64);
        let mut pool=ReaderPool::start_into_guarded(vec![broken],config(),vec![storage.target(0,data.len())]).unwrap();
        assert!(pool.consume_all(|_,_,_|Ok(())).is_err());drop(pool);assert_eq!(Arc::strong_count(&storage),1);
    }
    #[test]fn paused_pool_retains_offsets_and_foreground_finishes_without_repeating_chunks(){
        let f=fixture();let gate=f.0.join("gate");std::fs::write(&gate,b"").unwrap();
        let scope=crate::background_io::Scope::enter(crate::background_io::Policy{
            gate:gate.clone(),cancel:f.0.join("cancel"),stop:None,pin_gate:None});
        let data=vec![37u8;4097];let source=job(&f,"weights",&data);
        let mut pool=ReaderPool::start(vec![source],config()).unwrap();drop(scope);
        let mut actual=Vec::new();
        let result=pool.consume_all(|_,offset,chunk|{
            assert_eq!(offset,actual.len());actual.extend_from_slice(chunk);
            std::fs::remove_file(&gate).unwrap();Ok(())
        });
        assert_eq!(result.unwrap_err(),crate::background_io::PAUSED);assert!(!actual.is_empty());
        pool.foreground(true);
        pool.consume_all(|_,offset,chunk|{assert_eq!(offset,actual.len());actual.extend_from_slice(chunk);Ok(())}).unwrap();
        assert_eq!(actual,data);assert!(!gate.exists());
    }
    #[test]fn dropping_a_paused_pool_joins_its_workers(){
        let f=fixture();let _scope=crate::background_io::Scope::enter(crate::background_io::Policy{
            gate:f.0.join("closed"),cancel:f.0.join("cancel"),stop:None,pin_gate:None});
        let source=job(&f,"weights",&vec![12;4097]);
        drop(ReaderPool::start(vec![source],config()).unwrap());
    }
    #[test] fn preserves_every_byte_and_per_tensor_order_across_boundaries() {
        let f=fixture();let data:Vec<Vec<u8>>=(0..17).map(|i|(0..i*119+1).map(|n|(n%251) as u8).collect()).collect();
        let jobs=data.iter().enumerate().map(|(i,d)|job(&f,&i.to_string(),d)).collect();
        let mut pool=ReaderPool::start(jobs,config()).unwrap();
        let mut actual=vec![Vec::<u8>::new();data.len()];
        pool.consume_all(|i,offset,chunk|{assert_eq!(offset,actual[i].len());actual[i].extend(chunk);Ok(())}).unwrap();
        assert_eq!(actual,data);
    }
    #[test] fn balanced_readers_preserve_manifest_indices_and_chunk_leases() {
        let f=fixture();
        let data:Vec<Vec<u8>>=(0..19).map(|i|{
            let size=if i%3==0 {7001+i*307}else{1+i*23};
            (0..size).map(|n|((n+i)%251) as u8).collect()
        }).collect();
        let jobs:Vec<Job>=data.iter().enumerate().map(|(i,d)|job(&f,&i.to_string(),d)).collect();
        // Exercise ordinary and aligned owned leases with uneven tensor sizes.
        // An incorrect job-modulo recycle can strand a dynamically assigned reader.
        let cfg=Config {budget_bytes:32768,..config()}; // six page-aligned chunks
        for direct in [false,true] {
            let mut pool=ReaderPool::start_scheduled(jobs.clone(),cfg,direct,true,false,true).unwrap();
            let mut actual=vec![Vec::<u8>::new();data.len()];
            pool.consume_all(|i,offset,chunk|{
                assert_eq!(offset,actual[i].len());actual[i].extend(chunk);Ok(())
            }).unwrap();
            assert_eq!(actual,data);
        }
    }
    #[test] fn balanced_reader_failure_disconnects_all_workers() {
        let f=fixture();let data=vec![7;5001];
        let jobs:Vec<Job>=(0..13).map(|i|job(&f,&i.to_string(),&data)).collect();
        let mut changed=data;changed[1997]^=1;std::fs::write(&jobs[7].path,changed).unwrap();
        let cfg=Config {budget_bytes:32768,..config()};
        let mut pool=ReaderPool::start_scheduled(jobs,cfg,true,true,false,true).unwrap();
        assert!(pool.consume_all(|_,_,_|Ok(())).unwrap_err().contains("SHA256 mismatch"));
        drop(pool);
    }
    #[test] fn retained_one_use_chunks_survive_pool_drop_and_reject_failed_verification() {
        let f=fixture();let data:Vec<u8>=(0..5001).map(|n|(n%251) as u8).collect();
        let cfg=Config{workers:2,chunk_bytes:4096,pinned_slots:2,budget_bytes:32768};
        let j=job(&f,"retained",&data);
        let mut reusable=ReaderPool::start_options(vec![j.clone()],cfg,true,true).unwrap();
        assert!(reusable.retain_outputs().is_err());drop(reusable);
        let mut pool=ReaderPool::start_read_ahead_options(vec![j.clone()],cfg,true,true,false).unwrap();
        pool.retain_outputs().unwrap();assert!(pool.take_retained().is_err());
        pool.consume_all(|_,_,_|Ok(())).unwrap();
        let retained=pool.take_retained().unwrap().unwrap();drop(pool);
        std::fs::remove_file(&j.path).unwrap();
        assert_eq!(retained.bytes(),data.len());assert_eq!(retained.tensor_bytes(0).unwrap(),data.len());
        assert!(retained.tensor_bytes(1).is_err());assert_eq!(retained.read(0).unwrap(),data);
        assert!(retained.read(1).is_err());drop(retained);
        let mut corrupt=data;corrupt[1997]^=1;std::fs::write(&j.path,corrupt).unwrap();
        let mut pool=ReaderPool::start_read_ahead_options(vec![j],cfg,true,true,false).unwrap();
        pool.retain_outputs().unwrap();assert!(pool.consume_all(|_,_,_|Ok(())).is_err());
        assert!(pool.take_retained().is_err());drop(pool);
    }
    #[test] fn resident_stages_share_workers_and_keep_later_reads_owned_until_consumed() {
        let f=fixture();let data:Vec<u8>=(0..5001).map(|n|(n%251) as u8).collect();
        let cfg=Config{workers:3,chunk_bytes:4096,pinned_slots:2,budget_bytes:1<<20};
        let stages=(0..4).map(|stage|(0..5).map(|i|job(&f,&format!("{stage}-{i}"),&data)).collect()).collect();
        let mut pools=ReaderPool::start_retained_stages(stages,cfg,true,false).unwrap();
        for pool in &pools {assert_eq!(Arc::strong_count(pool.shared.as_ref().unwrap()),4);assert_eq!(pool.workers.len(),3);}
        for mut pool in pools.drain(..) {
            pool.consume_all(|_,_,_|Ok(())).unwrap();
            let host=pool.take_retained().unwrap().unwrap();drop(pool);
            for i in 0..5 {assert_eq!(host.read(i).unwrap(),data);}
        }
        let stages=vec![vec![job(&f,"cancel-a",&data)],vec![job(&f,"cancel-b",&data)]];
        drop(ReaderPool::start_retained_stages(stages,cfg,true,false).unwrap());
    }
    #[test] fn rejects_interior_corruption_and_wrong_sizes() {
        let f=fixture();let data=vec![7;5001];
        for (i,changed) in [193,997,4801].into_iter().enumerate() {
            let j=job(&f,&i.to_string(),&data);let mut bad=data.clone();bad[changed]^=1;
            std::fs::write(&j.path,bad).unwrap();
            let mut pool=ReaderPool::start(vec![j],config()).unwrap();
            assert!(pool.consume_all(|_,_,_|Ok(())).unwrap_err().contains("SHA256"));
        }
        for size in [5000,5002] {
            let j=job(&f,"size",&data);std::fs::write(&j.path,vec![7;size]).unwrap();
            let mut pool=ReaderPool::start(vec![j],config()).unwrap();assert!(pool.consume_all(|_,_,_|Ok(())).is_err());
        }
    }
    #[test] fn encoder_read_ahead_finishes_before_consumption_and_preserves_exact_bytes() {
        let f=fixture();let data:Vec<Vec<u8>>=(0..19).map(|i|(0..9001+i*197).map(|n|((i+n)%251) as u8).collect()).collect();
        let jobs:Vec<Job>=data.iter().enumerate().map(|(i,d)|job(&f,&i.to_string(),d)).collect();
        let cfg=Config{chunk_bytes:4096,budget_bytes:512<<10,..config()};
        for direct in [false,true] {
            let mut pool=ReaderPool::start_read_ahead_options(jobs.clone(),cfg,direct,true,false).unwrap();
            // No upload/recycle occurs until every reader has completed. A
            // rolling two-chunk pool would block here; the exact byte bound
            // admits every complete payload without an unbounded queue.
            for worker in &mut pool.workers {worker.thread.take().unwrap().join().unwrap();}
            let mut actual=vec![Vec::<u8>::new();data.len()];
            pool.consume_all(|i,offset,bytes|{assert_eq!(actual[i].len(),offset);actual[i].extend_from_slice(bytes);Ok(())}).unwrap();
            assert_eq!(actual,data);
            let mut failed=ReaderPool::start_read_ahead_options(jobs.clone(),cfg,direct,true,false).unwrap();
            assert_eq!(failed.consume_all(|_,_,_|Err("upload failed".into())).unwrap_err(),"upload failed");drop(failed);
            drop(ReaderPool::start_read_ahead_options(jobs.clone(),cfg,direct,true,false).unwrap());
        }
        assert!(ReaderPool::start_read_ahead_options(jobs.clone(),Config{budget_bytes:32768,..cfg},true,true,false).is_err());
        let mut changed=data[7].clone();changed[619]^=1;std::fs::write(&jobs[7].path,changed).unwrap();
        let mut pool=ReaderPool::start_read_ahead_options(jobs,cfg,true,true,false).unwrap();
        assert!(pool.consume_all(|_,_,_|Ok(())).unwrap_err().contains("SHA256 mismatch"));
    }
    #[test] fn cancels_full_queues_after_uploader_failure_or_early_drop() {
        let f=fixture();let jobs=(0..12).map(|i|job(&f,&i.to_string(),&vec![9;10000])).collect::<Vec<_>>();
        let start=std::time::Instant::now();
        {
            let mut pool=ReaderPool::start(jobs.clone(),config()).unwrap();
            assert_eq!(pool.consume_all(|_,_,_|Err("upload failed".into())).unwrap_err(),"upload failed");
        }
        drop(ReaderPool::start(jobs,config()).unwrap());
        assert!(start.elapsed().as_secs()<5,"reader cancellation deadlocked");
    }
    #[test] fn thirty_two_readers_require_the_expanded_buffer_budget() {
        let mut c=Config{workers:32,chunk_bytes:32<<20,pinned_slots:4,budget_bytes:2048<<20};
        assert!(c.validate().is_err());
        c.budget_bytes=2176<<20;
        assert!(c.validate().is_ok());
        c.budget_bytes-=1;
        assert!(c.validate().is_err());
    }
    #[test] fn wide_reader_sweep_requires_each_exact_buffer_bound() {
        for (workers,mib) in [(24,1664),(32,2176),(48,3200),(64,4224)] {
            let mut c=Config {workers,chunk_bytes:32<<20,pinned_slots:4,budget_bytes:mib<<20};
            assert!(c.validate().is_ok());
            c.budget_bytes-=1;
            assert!(c.validate().is_err());
        }
    }
    #[test] fn rejects_budget_overflow_and_paths() {
        let mut c=config();c.budget_bytes-=1;assert!(c.validate().is_err());
        c.workers=usize::MAX;assert!(c.validate().is_err());
        for file in ["", "../a", "/a", "a/../b"] {assert!(Job::new(std::path::Path::new("/x"),file,1,None).is_err());}
    }
    #[cfg(target_os="linux")]
    #[test] fn mapped_reads_preserve_unaligned_packed_slices_and_reject_corruption() {
        let f=fixture();let data:Vec<u8>=(0..18001).map(|n|(n%251) as u8).collect();
        std::fs::write(f.0.join("packed"),&data).unwrap();
        let ranges=[(13usize,5001usize),(5117,8193),(17001,17)];
        let jobs=ranges.iter().map(|&(start,bytes)|Job::slice(&f.0,"packed",start as u64,bytes,
            format!("{:x}",Sha256::digest(&data[start..start+bytes])),data.len() as u64).unwrap()).collect::<Vec<_>>();
        let cfg=Config{budget_bytes:65536,..config()};
        let mut pool=ReaderPool::start_scheduled_mode(jobs.clone(),cfg,true,true,false,true,true).unwrap();
        let mut actual=vec![Vec::new();ranges.len()];
        pool.consume_all(|i,offset,chunk|{assert_eq!(offset,actual[i].len());actual[i].extend_from_slice(chunk);Ok(())}).unwrap();
        for (i,&(start,bytes)) in ranges.iter().enumerate(){assert_eq!(actual[i],data[start..start+bytes]);}
        drop(pool);
        let mut changed=data;changed[3001]^=1;std::fs::write(f.0.join("packed"),changed).unwrap();
        let mut pool=ReaderPool::start_scheduled_mode(jobs,cfg,true,true,false,true,true).unwrap();
        assert!(pool.consume_all(|_,_,_|Ok(())).unwrap_err().contains("SHA256 mismatch"));
    }
    #[cfg(target_os="linux")]
    #[test] fn mapped_reads_are_bounded_and_cancel_full_queues() {
        let f=fixture();let jobs=(0..13).map(|i|job(&f,&i.to_string(),&vec![9;10001])).collect::<Vec<_>>();
        assert!(ReaderPool::start_scheduled_mode(jobs.clone(),config(),true,true,false,true,true).is_err());
        assert!(ReaderPool::start_scheduled_mode(jobs.clone(),config(),false,true,false,true,true).is_err());
        assert!(mapping::extent(usize::MAX).is_err());
        // Page-aligned direct-file chunks fit the same buffer budget, including
        // the pinned-copy fallback. Packed unaligned ranges need extra pages.
        let aligned=Config{chunk_bytes:4096,budget_bytes:32768,..config()};
        assert_eq!(mapping::budget(aligned,&jobs).unwrap(),32768);
        let mut unaligned=jobs[0].clone();unaligned.file_offset=1;
        assert!(mapping::budget(aligned,&[unaligned]).unwrap()>aligned.budget_bytes);
        let cfg=Config{budget_bytes:65536,..config()};let begin=std::time::Instant::now();
        {
            let mut pool=ReaderPool::start_scheduled_mode(jobs.clone(),cfg,true,true,false,true,true).unwrap();
            assert_eq!(pool.consume_all(|_,_,_|Err("upload failed".into())).unwrap_err(),"upload failed");
        }
        drop(ReaderPool::start_scheduled_mode(jobs,cfg,true,true,false,true,true).unwrap());
        assert!(begin.elapsed().as_secs()<5,"mapped reader cancellation deadlocked");
    }
}
