//! Model-independent residency for an aligned, verified tensor archive.
//! Immutable pinned host bytes survive device eviction. A device allocation
//! retains its copy lease until completion and its tensors share one owner.
use crate::{checkpoint::{Dtype,Manifest},native_backend::Pin,
    pinned_weights::{HostBudget,PinnedWeights},Result};
use std::{ffi::{c_int,c_void},path::{Component,Path},rc::Rc};

unsafe extern "C" {
    fn cudaMalloc(pointer:*mut *mut c_void,bytes:usize)->c_int;
    fn cudaFree(pointer:*mut c_void)->c_int;
    fn cudaStreamSynchronize(stream:*mut c_void)->c_int;
    fn cudaMemcpyAsync(dst:*mut c_void,src:*const c_void,bytes:usize,kind:c_int,stream:*mut c_void)->c_int;
}
fn check(code:c_int,operation:&str)->Result<()>{
    if code==0{Ok(())}else{Err(format!("{operation}: CUDA error {code}").into())}
}

pub struct HostArchive {
    pub manifest:Manifest,
    pub manifest_sha256:String,
    pub source_manifest_sha256:String,
    payload:Rc<PinnedWeights>,
    bytes:usize,
}
impl HostArchive {
    pub fn load(pin:&Pin,budget:&HostBudget)->Result<Rc<Self>>{
        pin.verify()?;
        let raw=std::fs::read(&pin.path)?;
        let metadata:serde_json::Value=serde_json::from_slice(&raw)?;
        let source=metadata["source_manifest_sha256"].as_str().ok_or("source tensor identity missing")?;
        if source.len()!=64 || !source.bytes().all(|b|b.is_ascii_hexdigit() && !b.is_ascii_uppercase()){
            return Err("invalid source tensor identity".into());
        }
        let manifest=Manifest::parse(&raw,Some(&pin.sha256))?;
        if manifest.files.len()!=1 || manifest.tensors.is_empty(){return Err("one nonempty tensor archive required".into());}
        let (file_key,file)=manifest.files.iter().next().unwrap();
        if file.root!="archive" || Path::new(&file.path).components().any(|part|!matches!(part,Component::Normal(_))){
            return Err("tensor archive payload escapes its root".into());
        }
        let path=pin.path.parent().ok_or("archive directory missing")?.join(&file.path);
        let bytes=usize::try_from(file.bytes)?;
        if std::fs::metadata(&path)?.len()!=file.bytes{return Err("archive payload extent differs".into());}
        let payload=PinnedWeights::load_tensors(&path,&file.sha256,&manifest,file_key,budget)?;
        Ok(Rc::new(Self{manifest,manifest_sha256:pin.sha256.clone(),
            source_manifest_sha256:source.into(),payload,bytes}))
    }
    /// Framework-owned cold archive pipeline. Metadata is admitted before
    /// readers start. Completion callbacks run on the CUDA owner, allowing a
    /// ready encoder's upload to overlap the remaining component reads.
    pub fn load_cohort(pins:&[Pin],budget:&HostBudget,ready:impl FnMut(usize,Rc<Self>)->Result<()>)->Result<()>{
        Self::load_cohort_with_prepare(pins,budget,||Ok(()),ready)
    }
    pub fn load_cohort_with_prepare(pins:&[Pin],budget:&HostBudget,prepare:impl FnOnce()->Result<()>,mut ready:impl FnMut(usize,Rc<Self>)->Result<()>)->Result<()>{
        Self::load_cohort_prioritized_with_prepare(pins,&vec![0;pins.len()],budget,prepare,ready)
    }
    pub fn load_cohort_prioritized_with_prepare(pins:&[Pin],priorities:&[u8],budget:&HostBudget,
        prepare:impl FnOnce()->Result<()>,mut ready:impl FnMut(usize,Rc<Self>)->Result<()>)->Result<()>{
        let mut metadata=Vec::new();
        for pin in pins{
            pin.verify()?;let raw=std::fs::read(&pin.path)?;
            let value:serde_json::Value=serde_json::from_slice(&raw)?;
            let source=value["source_manifest_sha256"].as_str().ok_or("source tensor identity missing")?.to_string();
            if source.len()!=64||!source.bytes().all(|b|b.is_ascii_hexdigit()&&!b.is_ascii_uppercase()){return Err("invalid source tensor identity".into());}
            let manifest=Manifest::parse(&raw,Some(&pin.sha256))?;
            if manifest.files.len()!=1||manifest.tensors.is_empty(){return Err("one nonempty tensor archive required".into());}
            let(key,file)=manifest.files.iter().next().unwrap();
            if file.root!="archive"||Path::new(&file.path).components().any(|p|!matches!(p,Component::Normal(_))){return Err("tensor archive payload escapes its root".into());}
            let path=pin.path.parent().ok_or("archive directory missing")?.join(&file.path);
            metadata.push((path,key.clone(),file.sha256.clone(),usize::try_from(file.bytes)?,source,manifest));
        }
        let specs=metadata.iter().map(|(path,key,sha,_,_,manifest)|(path.as_path(),sha.as_str(),manifest,key.as_str())).collect::<Vec<_>>();
        PinnedWeights::load_cohort_prioritized_with_prepare(&specs,priorities,budget,prepare,|index,payload|{
            let(_,_,_,bytes,source,manifest)=&metadata[index];
            ready(index,Rc::new(Self{manifest:manifest.clone(),manifest_sha256:pins[index].sha256.clone(),
                source_manifest_sha256:source.clone(),payload,bytes:*bytes}))
        })
    }
    pub fn bytes(&self)->usize{self.bytes}
    /// # Safety
    /// `stream` belongs to the calling thread's current device/context. That
    /// context and stream outlive every returned tensor lease.
    pub unsafe fn prefetch(self:&Rc<Self>,stream:*mut c_void,budget:usize)->Result<Rc<DeviceArchive>>{
        if self.bytes>budget{return Err("tensor archive exceeds the admitted device reservation".into());}
        let mut device=DeviceArchive{host:self.clone(),pointer:std::ptr::null_mut(),stream};
        check(cudaMalloc(&mut device.pointer,self.bytes),"archive allocation")?;
        let copy=self.payload.upload(0..self.bytes,device.pointer,self.bytes,stream)?;
        // Error paths drop the lease (which drains DMA) before the allocation.
        copy.wait()?;
        drop(copy);
        Ok(Rc::new(device))
    }
}

pub struct DeviceArchive {host:Rc<HostArchive>,pointer:*mut c_void,stream:*mut c_void}
impl DeviceArchive {
    pub fn bytes(&self)->usize{self.host.bytes}
    pub fn manifest_sha256(&self)->&str{&self.host.manifest_sha256}
    pub fn host_tensor(&self,name:&str)->Result<&[u8]>{
        let entry=self.host.manifest.tensor(name)?;
        let offset=usize::try_from(entry.offset)?;
        self.host.payload.slice(offset..offset.checked_add(entry.bytes).ok_or("host tensor range overflow")?)
    }
    pub fn tensor(self:&Rc<Self>,name:&str,sha256:&str,bytes:usize,shape:&[usize],dtype:Dtype)->Result<TensorLease>{
        let entry=self.host.manifest.tensor(name)?;
        if entry.sha256!=sha256 || entry.bytes!=bytes || entry.shape!=shape || entry.dtype!=dtype{
            return Err(format!("resident tensor differs from its admitted source: {name}").into());
        }
        let offset=usize::try_from(entry.offset)?;
        if offset.checked_add(bytes).is_none_or(|end|end>self.bytes()){
            return Err("resident tensor escapes its allocation".into());
        }
        Ok(TensorLease{address:unsafe{self.pointer.cast::<u8>().add(offset)} as u64,bytes,_owner:self.clone()})
    }
    /// Validation-only full D2H hash of the immutable serving payload.
    pub fn verify(&self)->Result<()>{
        use sha2::{Digest,Sha256};
        let mut hash=Sha256::new();
        let mut block=vec![0u8;(16usize<<20).min(self.bytes())];
        let mut offset=0;
        while offset<self.bytes(){
            let bytes=block.len().min(self.bytes()-offset);
            unsafe{
                let copied=cudaMemcpyAsync(block.as_mut_ptr().cast(),self.pointer.cast::<u8>().add(offset).cast(),bytes,2,self.stream);
                let drained=cudaStreamSynchronize(self.stream);
                check(copied,"archive D2H")?;check(drained,"archive audit completion")?;
            }
            hash.update(&block[..bytes]);offset+=bytes;
        }
        if format!("{:x}",hash.finalize())!=self.host.payload.sha256(){return Err("resident archive payload SHA differs".into());}
        Ok(())
    }
}
impl Drop for DeviceArchive {
    fn drop(&mut self){
        if !self.pointer.is_null(){unsafe{cudaStreamSynchronize(self.stream);cudaFree(self.pointer);}}
    }
}
pub struct TensorLease {pub address:u64,pub bytes:usize,_owner:Rc<DeviceArchive>}
