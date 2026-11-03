//! CPU-owned immutable weight storage, pinned only when an H2D consumer needs it.
use crate::Result;
use std::{ffi::{c_int,c_void},sync::{Arc,Mutex,atomic::{AtomicBool,AtomicUsize,Ordering}}};
use std::{fs::File,path::{Path,PathBuf},os::{fd::AsRawFd,unix::fs::MetadataExt}};
unsafe extern "C" {
    fn cudaHostRegister(pointer:*mut c_void,bytes:usize,flags:u32)->c_int;
    fn cudaHostUnregister(pointer:*mut c_void)->c_int;
    fn cudaGetDevice(device:*mut c_int)->c_int;
    fn cudaDeviceGetAttribute(value:*mut c_int,attribute:c_int,device:c_int)->c_int;
    fn cudaMemcpyAsync(dst:*mut c_void,src:*const c_void,bytes:usize,kind:c_int,stream:*mut c_void)->c_int;
    fn mmap(address:*mut c_void,bytes:usize,protection:c_int,flags:c_int,fd:c_int,offset:i64)->*mut c_void;
    fn munmap(address:*mut c_void,bytes:usize)->c_int;
    fn madvise(address:*mut c_void,bytes:usize,advice:c_int)->c_int;
}
struct Source {path:PathBuf,file:File,stamp:(u64,u64,u64,i64,i64,i64,i64)}
fn stamp(meta:&std::fs::Metadata)->(u64,u64,u64,i64,i64,i64,i64){
    (meta.dev(),meta.ino(),meta.len(),meta.mtime(),meta.mtime_nsec(),meta.ctime(),meta.ctime_nsec())
}
struct PinState {bytes:usize,indeterminate:bool}
/// Metadata lease only. Holding this never retains a host allocation, CUDA
/// registration or device resource; status cannot become a resource's owner.
pub struct HostRegistration {bytes:usize,live:AtomicBool,registered:AtomicUsize,indeterminate:AtomicBool}
impl HostRegistration {
    pub(crate) fn unregistered(bytes:usize)->Arc<Self>{Arc::new(Self{bytes,live:AtomicBool::new(true),
        registered:AtomicUsize::new(0),indeterminate:AtomicBool::new(false)})}
    pub(crate) fn pinned(bytes:usize)->Arc<Self>{Arc::new(Self{bytes,live:AtomicBool::new(true),
        registered:AtomicUsize::new(bytes),indeterminate:AtomicBool::new(false)})}
    pub(crate) fn release_pinned(&self,released:bool){
        if released{self.registered.store(0,Ordering::Release);self.live.store(false,Ordering::Release);}
        else{self.indeterminate.store(true,Ordering::Release);}
    }
    pub fn allocated_bytes(&self)->usize{if self.live.load(Ordering::Acquire){self.bytes}else{0}}
    pub fn registered_bytes(&self)->Result<usize>{
        if self.indeterminate.load(Ordering::Acquire){return Err("host weight registration is indeterminate; retire owner".into());}
        Ok(if self.live.load(Ordering::Acquire){self.registered.load(Ordering::Acquire)}else{0})
    }
}
const PIN_CHUNK:usize=128<<20;
fn copy_extent(source:usize,remaining:usize)->usize{remaining.min(PIN_CHUNK-source%PIN_CHUNK)}
pub struct HostBuffer {pointer:*mut u8,bytes:usize,offset:usize,source:Option<Source>,pinned:Mutex<PinState>,
    registration:Arc<HostRegistration>}
// A loader exclusively initializes its disjoint ranges before publication.
// Thereafter consumers retain immutable storage until their DMA has completed.
unsafe impl Send for HostBuffer {}
unsafe impl Sync for HostBuffer {}
impl HostBuffer {
    pub fn can_map_read_only()->Result<bool>{
        let(mut device,mut supported)=(0,0);
        for code in [unsafe{cudaGetDevice(&mut device)},unsafe{cudaDeviceGetAttribute(&mut supported,113,device)}]{
            if code!=0{return Err(format!("host registration capability: CUDA {code}").into());}
        }Ok(supported!=0)
    }
    pub fn new(bytes:usize)->Result<Self>{
        let alignment=if bytes>=2<<20{2usize<<20}else{4096};
        let bytes=Self::allocation_bytes(bytes)?;
        // Anonymous mmap is initialized storage without a serial userspace
        // memset of the whole checkpoint before bounded readers can begin.
        let extent=bytes.checked_add(alignment).ok_or("host allocation extent overflow")?;
        let raw=unsafe{mmap(std::ptr::null_mut(),extent,3,2|0x20,-1,0)};
        if raw as isize == -1{return Err(std::io::Error::last_os_error().into());}
        let start=(raw as usize+alignment-1)&!(alignment-1);let prefix=start-raw as usize;
        unsafe{
            if prefix>0{munmap(raw,prefix);}
            let suffix=extent-prefix-bytes;
            if suffix>0{munmap((start+bytes) as *mut c_void,suffix);}
            if alignment>4096{madvise(start as *mut c_void,bytes,14);} // MADV_HUGEPAGE; optional kernel policy
        }
        Ok(Self{pointer:start as *mut u8,bytes,offset:0,source:None,pinned:Mutex::new(PinState{bytes:0,indeterminate:false}),
            registration:HostRegistration::unregistered(bytes)})
    }
    pub fn allocation_bytes(bytes:usize)->Result<usize>{
        let alignment=if bytes>=2<<20{2usize<<20}else{4096};
        Ok(bytes.max(1).checked_add(alignment-1).ok_or("host allocation overflow")?&!(alignment-1))
    }
    pub fn allocated_bytes(&self)->usize{self.bytes}
    pub fn registration_snapshot(&self)->Arc<HostRegistration>{self.registration.clone()}
    /// Read-only accounting; never registers memory or changes CUDA context.
    pub fn registered_bytes(&self)->Result<usize>{
        self.registration.registered_bytes()
    }
    /// The caller must admit an immutable read-only checkpoint and validate
    /// every tensor's trusted guards before publishing this read-only mapping.
    pub fn map_file(path:&Path,offset:u64,bytes:usize)->Result<Self>{
        if bytes==0 || !path.is_absolute(){return Err("nonempty absolute host mapping required".into());}
        let file=File::open(path)?;let metadata=file.metadata()?;
        if offset.checked_add(bytes as u64).is_none_or(|n|n>metadata.len()){
            return Err("host mapping escapes checkpoint".into());
        }
        let start=offset&!4095;let prefix=usize::try_from(offset-start)?;
        let extent=Self::mapping_bytes(offset,bytes)?;
        let pointer=unsafe{mmap(std::ptr::null_mut(),extent,1,2,file.as_raw_fd(),i64::try_from(start)?)};
        if pointer as isize == -1{return Err(std::io::Error::last_os_error().into());}
        Ok(Self{pointer:pointer.cast(),bytes:extent,offset:prefix,
            source:Some(Source{path:path.to_path_buf(),file,stamp:stamp(&metadata)}),pinned:Mutex::new(PinState{bytes:0,indeterminate:false}),
            registration:HostRegistration::unregistered(extent)})
    }
    pub(crate) fn mapping_bytes(offset:u64,bytes:usize)->Result<usize>{
        let prefix=usize::try_from(offset&4095)?;
        Ok(prefix.checked_add(bytes).and_then(|n|n.checked_add(4095)).ok_or("host mapping overflow")?&!4095)
    }
    /// Mutable reader capacity; file-backed immutable mappings are never destinations.
    pub fn writable_bytes(&self)->Option<usize>{self.source.is_none().then_some(self.bytes-self.offset)}
    pub fn pointer(&self)->*mut u8{unsafe{self.pointer.add(self.offset)}}
    pub fn validate_source(&self)->Result<()>{
        if let Some(source)=&self.source{
            if stamp(&source.file.metadata()?)!=source.stamp || stamp(&std::fs::metadata(&source.path)?)!=source.stamp{
                return Err("host mapping checkpoint identity changed".into());
            }
        }Ok(())
    }
    pub fn pin(&self)->Result<()>{
        self.pin_with(||Ok(()))
    }
    pub fn pin_with(&self,mut checkpoint:impl FnMut()->Result<()>)->Result<()>{
        let mut pinned=self.pinned.lock().map_err(|_|"host weight pin state poisoned")?;
        if pinned.indeterminate{return Err("host weight registration is indeterminate; retire owner".into());}
        if pinned.bytes<self.bytes{
            self.validate_source()?;
            // Read-only registration keeps file-backed page-cache pages as
            // the H2D source; never write-pin and COW a whole checkpoint.
            let flags=if self.source.is_some(){8}else{1};
            while pinned.bytes<self.bytes{
                checkpoint()?;
                let bytes=PIN_CHUNK.min(self.bytes-pinned.bytes);
                let code=unsafe{cudaHostRegister(self.pointer.add(pinned.bytes).cast(),bytes,flags)};
                if code!=0{pinned.indeterminate=true;self.registration.indeterminate.store(true,Ordering::Release);
                    return Err(format!("host weight registration: CUDA {code}").into());}
                pinned.bytes+=bytes;
                self.registration.registered.store(pinned.bytes,Ordering::Release);
            }
        }Ok(())
    }
    /// The caller retains this buffer and the destination through stream
    /// completion. A CUDA copy cannot span separate host registrations.
    pub unsafe fn enqueue(&self,offset:usize,destination:*mut c_void,bytes:usize,stream:*mut c_void)->Result<()>{
        let source=self.offset.checked_add(offset).ok_or("host copy offset overflow")?;
        if source.checked_add(bytes).is_none_or(|end|end>self.bytes){return Err("host copy escapes allocation".into());}
        self.pin()?;
        let mut copied=0;
        while copied<bytes{
            let count=copy_extent(source+copied,bytes-copied);
            let code=unsafe{cudaMemcpyAsync(destination.cast::<u8>().add(copied).cast(),
                self.pointer.add(source+copied).cast(),count,1,stream)};
            if code!=0{return Err(format!("registered host H2D: CUDA {code}").into());}
            copied+=count;
        }Ok(())
    }
}

#[cfg(test)]mod tests{
    use super::*;
    #[test]fn file_mapping_preserves_unaligned_ranges_and_rejects_replacement(){
        let path=std::env::temp_dir().join(format!("mapped-host-{}-{}",std::process::id(),
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
        std::fs::write(&path,b"abcdefghij").unwrap();
        let mapped=HostBuffer::map_file(&path,3,5).unwrap();mapped.validate_source().unwrap();
        assert_eq!(unsafe{std::slice::from_raw_parts(mapped.pointer(),5)},b"defgh");
        assert!(HostBuffer::map_file(&path,8,5).is_err());
        let replacement=path.with_extension("replacement");std::fs::write(&replacement,b"ABCDEFGHIJ").unwrap();
        std::fs::rename(replacement,&path).unwrap();assert!(mapped.validate_source().is_err());
        drop(mapped);std::fs::remove_file(path).unwrap();
    }
    #[test]fn anonymous_storage_is_initialized(){
        let buffer=HostBuffer::new(8193).unwrap();
        assert_eq!(buffer.registered_bytes().unwrap(),0);
        assert_eq!(buffer.allocated_bytes(),HostBuffer::allocation_bytes(8193).unwrap());
        assert!(unsafe{std::slice::from_raw_parts(buffer.pointer(),8193)}.iter().all(|b|*b==0));
    }
    #[test]fn registration_status_does_not_wait_for_the_idle_gate(){
        use std::{sync::{Arc,mpsc},thread,time::Duration};
        let buffer=Arc::new(HostBuffer::new(8193).unwrap());
        let(entered,waiting)=mpsc::channel();let(release,resume)=mpsc::channel();let worker=buffer.clone();
        let task=thread::spawn(move||worker.pin_with(||{
            entered.send(()).unwrap();resume.recv().unwrap();Err("registration cancelled before CUDA".into())
        }).map_err(|e|e.to_string()));
        waiting.recv_timeout(Duration::from_secs(2)).unwrap();
        let status=buffer.registration_snapshot();let(send,receive)=mpsc::channel();
        let observer=thread::spawn(move||send.send(status.registered_bytes().map_err(|e|e.to_string())).unwrap());
        let snapshot=receive.recv_timeout(Duration::from_millis(100));
        release.send(()).unwrap();task.join().unwrap().unwrap_err();observer.join().unwrap();
        assert_eq!(snapshot.unwrap().unwrap(),0);assert_eq!(buffer.registered_bytes().unwrap(),0);
    }
    #[test]fn metadata_snapshots_do_not_retain_the_host_resource(){
        let buffer=HostBuffer::new(8193).unwrap();let metadata=buffer.registration_snapshot();
        assert_eq!(metadata.allocated_bytes(),buffer.allocated_bytes());
        drop(buffer);assert_eq!(metadata.allocated_bytes(),0);assert_eq!(metadata.registered_bytes().unwrap(),0);
    }
    #[test]fn copies_stop_at_every_registration_boundary_including_unaligned_views(){
        let start=PIN_CHUNK-256;let bytes=2*PIN_CHUNK+1000;
        let mut copied=0;let mut extents=Vec::new();
        while copied<bytes{
            let count=copy_extent(start+copied,bytes-copied);
            assert!(count>0);assert_eq!((start+copied)/PIN_CHUNK,(start+copied+count-1)/PIN_CHUNK);
            extents.push(count);copied+=count;
        }
        assert_eq!(extents,vec![256,PIN_CHUNK,PIN_CHUNK,744]);assert_eq!(copied,bytes);
    }
}
impl Drop for HostBuffer {
    fn drop(&mut self){
        // The last copy lease is drained before the last buffer owner drops.
        // On an indeterminate unregister, keep the allocation until exit.
        let state=self.pinned.get_mut().unwrap_or_else(|e|e.into_inner());
        if state.indeterminate{return;}
        for offset in (0..state.bytes).step_by(PIN_CHUNK){
            if unsafe{cudaHostUnregister(self.pointer.add(offset).cast())}!=0{
                self.registration.indeterminate.store(true,Ordering::Release);return;
            }
            self.registration.registered.fetch_sub(PIN_CHUNK.min(state.bytes-offset),Ordering::Release);
        }
        if unsafe{munmap(self.pointer.cast(),self.bytes)}==0{self.registration.live.store(false,Ordering::Release);}
    }
}
