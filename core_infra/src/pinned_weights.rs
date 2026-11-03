//! Verified, immutable host weights and event-owned asynchronous H2D sources.
//! No tensor arithmetic or model names. Reservations include complete files;
//! cancelling a consumer cannot free host bytes still referenced by a copy.
use sha2::{Digest,Sha256};
use std::{cell::{Cell,RefCell},collections::HashMap,ffi::{c_int,c_void},fs::File,io::Read,ops::Range,path::Path,ptr,rc::Rc,sync::Arc};
type Result<T> = std::result::Result<T,Box<dyn std::error::Error>>;
type Stream=*mut c_void;
unsafe extern "C" {
    fn cudaHostAlloc(out:*mut *mut c_void,bytes:usize,flags:u32)->c_int;
    fn cudaFreeHost(pointer:*mut c_void)->c_int;
    fn cudaMemcpyAsync(dst:*mut c_void,src:*const c_void,bytes:usize,kind:c_int,stream:Stream)->c_int;
    fn cudaEventCreateWithFlags(event:*mut *mut c_void,flags:u32)->c_int;
    fn cudaEventRecord(event:*mut c_void,stream:Stream)->c_int;
    fn cudaEventQuery(event:*mut c_void)->c_int;
    fn cudaEventSynchronize(event:*mut c_void)->c_int;
    fn cudaEventDestroy(event:*mut c_void)->c_int;
    fn cudaStreamSynchronize(stream:Stream)->c_int;
}
fn check(code:c_int,operation:&str)->Result<()>{
    if code==0{Ok(())}else{Err(format!("{operation}: CUDA error {code}").into())}
}
struct HostRecord {registration:Arc<crate::host_buffer::HostRegistration>,reserved:Cell<bool>}
struct Budget {limit:usize,used:Cell<usize>,records:RefCell<Vec<Rc<HostRecord>>>}
/// Allocation metadata only. Readiness and full payload verification remain
/// separate; retained bytes include private readers and paused TLS loads.
#[derive(Clone,Copy,Debug,PartialEq,Eq)]
pub struct HostSnapshot {pub reserved_bytes:usize,pub retained_bytes:usize,pub pinned_bytes:usize}
#[derive(Clone)]
pub struct HostBudget(Rc<Budget>);
impl HostBudget {
    pub fn new(limit:usize)->Self{Self(Rc::new(Budget{limit,used:Cell::new(0),records:RefCell::new(Vec::new())}))}
    /// Outstanding admission leases; this is not a registration count.
    pub fn used(&self)->usize{self.0.used.get()}
    pub fn snapshot(&self)->Result<HostSnapshot>{
        let mut records=self.0.records.borrow_mut();let(mut retained,mut pinned)=(0usize,0usize);
        for record in records.iter(){
            let bytes=record.registration.allocated_bytes();let registered=record.registration.registered_bytes()?;
            retained=retained.checked_add(bytes).ok_or("host retained extent overflow")?;
            pinned=pinned.checked_add(registered).ok_or("host registered extent overflow")?;
        }
        records.retain(|record|record.reserved.get()||record.registration.allocated_bytes()>0);
        Ok(HostSnapshot{reserved_bytes:self.used(),retained_bytes:retained,pinned_bytes:pinned})
    }
    fn reserve(&self,bytes:usize)->Result<Reservation>{
        let used=self.used().checked_add(bytes).ok_or("host weight budget overflow")?;
        // A deferred registration may retain the real allocation after its
        // PinnedWeights lease is dropped. It still occupies this same budget.
        self.snapshot()?;
        let orphaned=self.0.records.borrow().iter().filter(|record|!record.reserved.get())
            .try_fold(0usize,|sum,record|sum.checked_add(record.registration.allocated_bytes()).ok_or("host retained extent overflow"))?;
        if used.checked_add(orphaned).is_none_or(|total|total>self.0.limit){return Err("pinned host weight budget exceeded".into());}
        self.0.used.set(used);Ok(Reservation{budget:self.clone(),bytes,record:None})
    }
}
struct Reservation {budget:HostBudget,bytes:usize,record:Option<Rc<HostRecord>>}
impl Reservation {
    fn track(&mut self,registration:Arc<crate::host_buffer::HostRegistration>)->Result<()>{
        if self.record.is_some()||registration.allocated_bytes()>self.bytes{return Err("host allocation exceeds its reservation".into());}
        let mut records=self.budget.0.records.borrow_mut();
        if records.iter().any(|record|Arc::ptr_eq(&record.registration,&registration)){
            return Err("host allocation already belongs to this budget".into());
        }
        let record=Rc::new(HostRecord{registration,reserved:Cell::new(true)});
        records.push(record.clone());self.record=Some(record);Ok(())
    }
}
impl Drop for Reservation {fn drop(&mut self){
    if let Some(record)=&self.record{record.reserved.set(false);}
    self.budget.0.used.set(self.budget.used()-self.bytes);
}}

/// Thread-owned until the native backend shuts down; immutable after load().
pub struct PinnedWeights {pointer:*mut u8,bytes:usize,sha256:String,_reservation:Reservation,
    cpu:Option<std::sync::Arc<crate::host_buffer::HostBuffer>>}
struct PendingLoad {allocation:PinnedWeights,readers:crate::weight_loader::ReaderPool,
    jobs:Vec<crate::weight_loader::Job>,begun:std::time::Instant,mapped:bool,
    read_seconds:f64,copy_seconds:f64,pin_seconds:f64}
thread_local! {static PENDING:RefCell<HashMap<String,PendingLoad>>=RefCell::new(HashMap::new());}
impl PinnedWeights {
    /// Load an admitted tensor archive through the common bounded readers.
    /// Its SHA-pinned manifest supplies per-tensor light guards. Full payload
    /// auditing is separate; an explicit NATIVE_LOAD_VERIFY=full still wins.
    pub fn load_tensors(path:&Path,expected:&str,manifest:&crate::checkpoint::Manifest,
                        file_key:&str,budget:&HostBudget)->Result<Rc<Self>>{
        if !crate::weight_loader::flag("NATIVE_RESIDENT_GUARDED_LOAD")?{
            return Self::load(path,expected,budget);
        }
        let file=manifest.files.get(file_key).ok_or("archive payload missing")?;
        if file.sha256!=expected || std::fs::metadata(path)?.len()!=file.bytes{
            return Err("archive payload identity/extent differs".into());
        }
        let full=match std::env::var("NATIVE_LOAD_VERIFY").as_deref(){
            Ok("full")=>true,Err(_)|Ok("light")=>false,_=>return Err("NATIVE_LOAD_VERIFY must be full or light".into()),
        };
        if full{return Self::load(path,expected,budget);}
        let config=crate::weight_loader::Config::from_env()?;
        let mut root=path;
        for _ in Path::new(&file.path).components(){root=root.parent().ok_or("archive payload directory missing")?;}
        if root.join(&file.path)!=path{return Err("archive payload path differs".into());}
        let roots=HashMap::from([(file.root.clone(),root.to_path_buf())]);
        let mut jobs=Vec::new();
        for (name,tensor) in &manifest.tensors{
            if tensor.file!=file_key{return Err("archive tensor refers to another payload".into());}
            jobs.push(manifest.job(name,&roots)?);
        }
        if jobs.is_empty(){return Err("empty tensor archive".into());}
        let mut ranges=jobs.iter().map(|job|(job.file_offset as usize,job.bytes)).collect::<Vec<_>>();
        ranges.sort_unstable();let bytes=usize::try_from(file.bytes)?;let mut end=0usize;
        for &(offset,count) in &ranges{
            if offset<end || offset.checked_add(count).is_none_or(|n|n>bytes){
                return Err("archive tensor ranges overlap or escape payload".into());
            }
            end=offset+count;
        }
        let key=format!("{}:{expected}",path.display());
        let retained=PENDING.with(|pending|pending.borrow_mut().remove(&key));
        if let Some(pending)=&retained{eprintln!("{}",serde_json::json!({"event":"resident.host_weights_resumed",
            "path":path,"initialized_bytes":pending.readers.initialized_bytes(),"foreground":crate::background_io::current().is_none()}));}
        let mut pending=if let Some(pending)=retained{
            if !Rc::ptr_eq(&pending.allocation._reservation.budget.0,&budget.0){
                return Err("retained host load belongs to another budget".into());
            }
            pending
        }else{
            let begun=std::time::Instant::now();
            let requested=crate::weight_loader::flag("NATIVE_RESIDENT_MAPPED_HOST")?;
            let mapped=requested && crate::host_buffer::HostBuffer::can_map_read_only()?;
            let extent=if mapped{crate::host_buffer::HostBuffer::mapping_bytes(0,bytes)?}
                else{crate::host_buffer::HostBuffer::allocation_bytes(bytes)?};
            let mut reservation=budget.reserve(extent)?;
            let cpu=std::sync::Arc::new(if mapped{crate::host_buffer::HostBuffer::map_file(path,0,bytes)?}
                else{crate::host_buffer::HostBuffer::new(bytes)?});
            reservation.track(cpu.registration_snapshot())?;
            let readers=if mapped{crate::weight_loader::ReaderPool::start_scheduled_mode(
                jobs.clone(),config,false,false,true,true,false)?}else{
                let targets=jobs.iter().map(|job|unsafe{crate::weight_loader::HostTarget::new(cpu.clone(),
                    cpu.pointer().add(job.file_offset as usize),job.bytes)}).collect::<std::result::Result<Vec<_>,_>>()?;
                crate::weight_loader::ReaderPool::start_into_guarded(jobs.clone(),config,targets)?
            };
            let allocation=Self{pointer:cpu.pointer(),bytes,sha256:expected.to_owned(),_reservation:reservation,cpu:Some(cpu)};
            // Preserve archive alignment gaps, which are covered by the final
            // full GPU audit even though only tensors have light read guards.
            use std::os::unix::fs::FileExt;
            let source=File::open(path)?;end=0;
            if !mapped{for &(offset,count) in &ranges{
                if offset>end{unsafe{
                    source.read_exact_at(std::slice::from_raw_parts_mut(allocation.pointer.add(end),offset-end),end as u64)?;
                }}
                end=offset+count;
            }
            if bytes>end{unsafe{
                source.read_exact_at(std::slice::from_raw_parts_mut(allocation.pointer.add(end),bytes-end),end as u64)?;
            }}}
            PendingLoad{allocation,readers,jobs,begun,mapped,read_seconds:0.0,copy_seconds:0.0,pin_seconds:0.0}
        };
        // A foreground request adopts the same allocation, byte offsets and
        // reader guards. It does not wait for a speculative model to finish,
        // discard partial reads, or read the archive a second time.
        pending.readers.foreground(crate::background_io::current().is_none());
        let result=(||->Result<()>{
            let pointer=pending.allocation.pointer;
            let read=std::time::Instant::now();
            let read_result=pending.readers.consume_all(|index,offset,data|{
                let start=usize::try_from(pending.jobs[index].file_offset).map_err(|_|"archive offset overflow")?;
                let position=start.checked_add(offset).ok_or("archive copy overflow")?;
                if position.checked_add(data.len()).is_none_or(|n|n>bytes){return Err("archive copy escapes payload".into());}
                let copy=std::time::Instant::now();
                if !pending.mapped && data.as_ptr()!=unsafe{pointer.add(position)}{
                    return Err("host reader destination changed".into());
                }
                pending.copy_seconds+=copy.elapsed().as_secs_f64();Ok(())
            });
            pending.read_seconds+=read.elapsed().as_secs_f64();read_result?;
            let pin=std::time::Instant::now();
            let pin_result=if crate::background_io::current().is_some(){
                crate::background_io::defer_pin(pending.allocation.cpu.as_ref().unwrap().clone());Ok(())
            }else{pending.allocation.cpu.as_ref().unwrap().pin_with(||pending.readers.checkpoint().map_err(Into::into))};
            pending.pin_seconds+=pin.elapsed().as_secs_f64();pin_result?;
            Ok(())
        })();
        if let Err(error)=result{
            if error.to_string()==crate::background_io::PAUSED{
                eprintln!("{}",serde_json::json!({"event":"resident.host_weights_paused","path":path,
                    "initialized_bytes":pending.readers.initialized_bytes(),"total_bytes":bytes}));
                PENDING.with(|retained|retained.borrow_mut().insert(key,pending));
            }
            return Err(error);
        }
        eprintln!("{}",serde_json::json!({"event":"resident.host_weights_loaded","path":path,"bytes":bytes,
            "seconds":pending.begun.elapsed().as_secs_f64(),"read_guard_seconds":pending.read_seconds,
            "cpu_copy_seconds":pending.copy_seconds,"pin_seconds":pending.pin_seconds,"retained_reads":true}));
        Ok(Rc::new(pending.allocation))
    }
    /// A cold component cohort shares one reader budget and publishes each
    /// guarded immutable allocation independently. Retained/background loads
    /// keep their existing pause/takeover state machine.
    pub fn load_cohort(specs:&[(&Path,&str,&crate::checkpoint::Manifest,&str)],budget:&HostBudget,
                       ready:impl FnMut(usize,Rc<Self>)->Result<()>)->Result<()>{
        Self::load_cohort_with_prepare(specs,budget,||Ok(()),ready)
    }
    pub fn load_cohort_with_prepare(specs:&[(&Path,&str,&crate::checkpoint::Manifest,&str)],budget:&HostBudget,
                       prepare:impl FnOnce()->Result<()>,mut ready:impl FnMut(usize,Rc<Self>)->Result<()>)->Result<()>{
        Self::load_cohort_prioritized_with_prepare(specs,&vec![0;specs.len()],budget,prepare,ready)
    }
    pub fn load_cohort_prioritized_with_prepare(specs:&[(&Path,&str,&crate::checkpoint::Manifest,&str)],priorities:&[u8],budget:&HostBudget,
                       prepare:impl FnOnce()->Result<()>,mut ready:impl FnMut(usize,Rc<Self>)->Result<()>)->Result<()>{
        if priorities.len()!=specs.len(){return Err("weight priority declaration differs from its components".into());}
        match std::env::var("NATIVE_LOAD_VERIFY").as_deref(){Err(_)|Ok("light")|Ok("full")=>{},_=>return Err("NATIVE_LOAD_VERIFY must be full or light".into())}
        let eligible=crate::background_io::current().is_none()
            && crate::weight_loader::flag("NATIVE_RESIDENT_GUARDED_LOAD")?
            && !crate::weight_loader::flag("NATIVE_RESIDENT_MAPPED_HOST")?
            && std::env::var("NATIVE_LOAD_VERIFY").as_deref()!=Ok("full")
            && !PENDING.with(|pending|specs.iter().any(|(path,sha,_,_)|pending.borrow().contains_key(&format!("{}:{sha}",path.display()))));
        if !eligible{
            prepare()?;
            let mut order=(0..specs.len()).collect::<Vec<_>>();order.sort_by_key(|&i|priorities[i]);
            for index in order{let(path,sha,manifest,key)=&specs[index];ready(index,Self::load_tensors(path,sha,manifest,key,budget)?)?;}
            return Ok(());
        }
        let config=crate::weight_loader::Config::from_env()?;
        let mut allocations=Vec::new();let mut archives=Vec::new();
        for(path,sha,manifest,key)in specs{
            let file=manifest.files.get(*key).ok_or("archive payload missing")?;
            if file.sha256!=*sha || sha.len()!=64 || !sha.bytes().all(|b|b.is_ascii_hexdigit()&&!b.is_ascii_uppercase()){
                return Err("archive payload identity differs".into());
            }
            let bytes=usize::try_from(file.bytes)?;
            let mut reservation=budget.reserve(crate::host_buffer::HostBuffer::allocation_bytes(bytes)?)?;
            let cpu=std::sync::Arc::new(crate::host_buffer::HostBuffer::new(bytes)?);
            reservation.track(cpu.registration_snapshot())?;
            let mut guards=Vec::new();
            for tensor in manifest.tensors.values(){
                if tensor.file!=*key{return Err("archive tensor refers to another payload".into());}
                guards.push((usize::try_from(tensor.offset)?,tensor.guard.clone()));
            }
            guards.sort_by_key(|(offset,_)|*offset);
            archives.push(crate::archive_reader::Archive{path:path.to_path_buf(),bytes,storage:cpu.clone(),guards});
            allocations.push(Rc::new(Self{pointer:cpu.pointer(),bytes,sha256:sha.to_string(),_reservation:reservation,cpu:Some(cpu)}));
        }
        let begun=std::time::Instant::now();
        crate::archive_reader::read_prioritized_with_prepare(&archives,priorities,config,prepare,|index,stats|{
            let pin=std::time::Instant::now();allocations[index].cpu.as_ref().unwrap().pin()?;
            let pin_seconds=pin.elapsed().as_secs_f64();
            eprintln!("{}",serde_json::json!({"event":"resident.host_weights_loaded","path":archives[index].path,
                "bytes":stats.bytes,"seconds":begun.elapsed().as_secs_f64(),"read_guard_seconds":stats.read_complete_seconds+stats.guard_seconds,
                "pin_seconds":pin_seconds,"cpu_copy_seconds":0.0,"reader_mode":"archive-cohort","shared_workers":config.workers,
                "overlapped_components":specs.len(),"priority":priorities[index],"direct_read_bytes":stats.direct_read_bytes,"retained_reads":true}));
            ready(index,allocations[index].clone())
        })
    }
    pub fn load(path:&Path,expected:&str,budget:&HostBudget)->Result<Rc<Self>>{
        crate::startup_trace::configure_from_env();
        if expected.len()!=64 || !expected.bytes().all(|v|v.is_ascii_hexdigit() && !v.is_ascii_uppercase()){
            return Err("full weight SHA256 required".into());
        }
        let mut file=File::open(path)?;let bytes:usize=file.metadata()?.len().try_into()?;
        if bytes==0{return Err("empty weight payload".into());}
        let label=path.to_string_lossy();
        let _load=crate::startup_trace::Span::new("pinned_weight_load",&label,bytes);
        let reservation=budget.reserve(bytes)?;let mut pointer=ptr::null_mut();
        let status=crate::startup_trace::measure("pinned_host_allocate",&label,bytes,
            ||unsafe{cudaHostAlloc(&mut pointer,bytes,1)}); // portable pinned host memory
        if pointer.is_null(){check(status,"cudaHostAlloc")?;return Err("null pinned host allocation".into());}
        let mut allocation=Self{pointer:pointer.cast(),bytes,sha256:String::new(),_reservation:reservation,cpu:None};
        allocation._reservation.track(crate::host_buffer::HostRegistration::pinned(bytes))?;
        check(status,"cudaHostAlloc")?;
        // Read requires initialized u8 storage. This touch is startup work,
        // never repeated for a warm immutable checkpoint.
        crate::startup_trace::measure("pinned_host_zero",&label,bytes,
            ||unsafe{ptr::write_bytes(allocation.pointer,0,bytes);});
        let mut digest=Sha256::new();let mut offset=0;
        while offset<bytes{
            let count=(bytes-offset).min(16<<20);
            let block=unsafe{std::slice::from_raw_parts_mut(allocation.pointer.add(offset),count)};
            crate::startup_trace::measure("pinned_weight_read",&label,count,||file.read_exact(block))?;
            crate::startup_trace::measure("pinned_weight_sha256",&label,count,||digest.update(&*block));offset+=count;
        }
        let mut extra=[0u8;1];
        if file.read(&mut extra)?!=0{return Err("weight file grew during verification".into());}
        allocation.sha256=format!("{:x}",digest.finalize());
        if allocation.sha256!=expected{return Err("pinned weight SHA256 mismatch".into());}
        Ok(Rc::new(allocation))
    }
    pub fn sha256(&self)->&str{&self.sha256}
    pub fn slice(&self,range:Range<usize>)->Result<&[u8]>{
        if range.start>range.end || range.end>self.bytes{return Err("weight view escapes verified payload".into());}
        Ok(unsafe{std::slice::from_raw_parts(self.pointer.add(range.start),range.len())})
    }
    /// # Safety
    /// dst must stay alive until this copy completes; the caller owns the CUDA
    /// context and must order all uses/frees of dst after its copy stream.
    pub unsafe fn upload(self:&Rc<Self>,range:Range<usize>,dst:*mut c_void,dst_bytes:usize,stream:Stream)->Result<CopyLease>{
        let start=range.start;
        let source=self.slice(range)?;
        if source.len()>dst_bytes{return Err("weight upload exceeds its device allocation".into());}
        if let Some(cpu)=&self.cpu{cpu.pin()?;}
        let mut event=ptr::null_mut();check(cudaEventCreateWithFlags(&mut event,2),"cudaEventCreate")?;
        let mut lease=CopyLease{event,stream,recorded:false,source:Some(self.clone())};
        if let Some(cpu)=&self.cpu{cpu.enqueue(start,dst,source.len(),stream)?;}
        else{check(cudaMemcpyAsync(dst,source.as_ptr().cast(),source.len(),1,stream),"weight H2D")?;}
        check(cudaEventRecord(event,stream),"weight copy event")?;lease.recorded=true;Ok(lease)
    }
}
impl Drop for PinnedWeights {fn drop(&mut self){if self.cpu.is_none(){
    let released=unsafe{cudaFreeHost(self.pointer.cast())}==0;
    if let Some(record)=&self._reservation.record{record.registration.release_pinned(released);}
}}}

pub struct CopyLease {event:*mut c_void,stream:Stream,recorded:bool,source:Option<Rc<PinnedWeights>>}
impl CopyLease {
    pub fn ready(&self)->Result<bool>{
        if !self.recorded{return Ok(false);}
        match unsafe{cudaEventQuery(self.event)}{0=>Ok(true),34=>Ok(false),code=>{
            check(code,"cudaEventQuery")?;unreachable!()
        }}
    }
    pub fn wait(&self)->Result<()>{
        if self.recorded{check(unsafe{cudaEventSynchronize(self.event)},"cudaEventSynchronize")}
        else{check(unsafe{cudaStreamSynchronize(self.stream)},"failed copy drain")}
    }
}
impl Drop for CopyLease {
    fn drop(&mut self){
        if self.wait().is_err(){
            // An indeterminate CUDA failure retires the owner. Keep DMA
            // sources allocated until process exit rather than reusing them.
            if let Some(source)=self.source.take(){std::mem::forget(source);}
        }
        unsafe{cudaEventDestroy(self.event);}
    }
}

#[cfg(test)]mod host_budget_tests {
    use super::*;
    use crate::host_buffer::{HostBuffer,HostRegistration};
    #[test]fn retained_unregistered_storage_stays_charged_after_its_admission_lease_drops(){
        let budget=HostBudget::new(200);let registration=HostRegistration::unregistered(80);
        let mut lease=budget.reserve(100).unwrap();lease.track(registration.clone()).unwrap();
        assert_eq!(budget.snapshot().unwrap(),HostSnapshot{reserved_bytes:100,retained_bytes:80,pinned_bytes:0});
        assert_eq!(budget.clone().snapshot().unwrap().retained_bytes,80);
        drop(lease); // A deferred-pin/PENDING owner may still retain the storage.
        assert_eq!(budget.snapshot().unwrap(),HostSnapshot{reserved_bytes:0,retained_bytes:80,pinned_bytes:0});
        assert!(budget.reserve(121).is_err());
        let remaining=budget.reserve(120).unwrap();assert!(budget.reserve(1).is_err());
        registration.release_pinned(true); // Actual resource retirement releases this metadata.
        assert_eq!(budget.snapshot().unwrap().retained_bytes,0);drop(remaining);
        assert!(budget.reserve(200).is_ok());
    }
    #[test]fn legacy_pinned_leases_and_aliases_are_counted_once_and_fail_closed(){
        let budget=HostBudget::new(160);let registration=HostRegistration::pinned(80);
        let mut lease=budget.reserve(80).unwrap();lease.track(registration.clone()).unwrap();
        let mut duplicate=budget.reserve(80).unwrap();assert!(duplicate.track(registration.clone()).is_err());drop(duplicate);
        assert_eq!(budget.snapshot().unwrap(),HostSnapshot{reserved_bytes:80,retained_bytes:80,pinned_bytes:80});
        registration.release_pinned(false);
        assert!(budget.snapshot().is_err());assert!(budget.reserve(1).is_err());
    }
    #[test]fn known_allocation_padding_is_admitted_before_storage_is_created(){
        let budget=HostBudget::new(4095);
        assert!(budget.reserve(HostBuffer::allocation_bytes(4095).unwrap()).is_err());
        assert!(budget.reserve(HostBuffer::mapping_bytes(3,4094).unwrap()).is_err());
        assert_eq!(budget.used(),0);assert_eq!(budget.snapshot().unwrap().retained_bytes,0);
        assert!(HostBuffer::allocation_bytes(usize::MAX).is_err());
        assert!(HostBuffer::mapping_bytes(3,usize::MAX).is_err());
    }
    #[test]
    #[cfg(target_os="linux")]
    fn paused_private_buffer_and_its_deferred_owner_have_truthful_budget_metadata(){
        let bytes=8193;let extent=HostBuffer::allocation_bytes(bytes).unwrap();let budget=HostBudget::new(extent);
        let mut reservation=budget.reserve(extent).unwrap();
        let cpu=Arc::new(HostBuffer::new(bytes).unwrap());reservation.track(cpu.registration_snapshot()).unwrap();
        let allocation=PinnedWeights{pointer:cpu.pointer(),bytes,sha256:"a".repeat(64),_reservation:reservation,cpu:Some(cpu.clone())};
        assert_eq!(cpu.pin_with(||Err(crate::background_io::PAUSED.into())).unwrap_err().to_string(),crate::background_io::PAUSED);
        assert_eq!(budget.snapshot().unwrap(),HostSnapshot{reserved_bytes:extent,retained_bytes:extent,pinned_bytes:0});
        drop(allocation);assert_eq!(budget.snapshot().unwrap().retained_bytes,extent);
        assert!(budget.reserve(1).is_err());drop(cpu);
        assert_eq!(budget.snapshot().unwrap(),HostSnapshot{reserved_bytes:0,retained_bytes:0,pinned_bytes:0});
    }
}
