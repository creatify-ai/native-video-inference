//! Bounded safetensors reads. A component's full SHA is frozen by preparation;
//! each tensor range can additionally be checked against an exported raw SHA.
use crate::{gold::sha, Result};
use serde_json::Value;
use sha2::{Digest,Sha256};
use std::{cell::{Cell,RefCell},fs::File,io::{Read,Seek,SeekFrom},path::{Path,PathBuf},rc::Rc};

pub struct Checkpoint {pub path:PathBuf,pub header:Value,data_start:u64,file_bytes:u64,
    tokenizer:RefCell<Option<Rc<crate::frontend::GemmaTokenizer>>>,tokenizer_seconds:Cell<f64>,
    #[cfg(feature="cuda")] pinned:Option<std::rc::Rc<native_core_infra::pinned_weights::PinnedWeights>>,
    #[cfg(feature="cuda")] admitted:Option<(String,std::sync::Arc<native_core_infra::checkpoint::Manifest>)>,
    #[cfg(feature="cuda")] retained:RefCell<std::collections::BTreeMap<String,(String,std::sync::Arc<native_core_infra::model_load::HostWeights>)>>,
}
impl Checkpoint {
    pub fn verify(&self,expected:&str)->Result<()>{
        let mut file=File::open(&self.path)?;let mut block=vec![0;16<<20];let mut hash=Sha256::new();
        loop{let n=file.read(&mut block)?;if n==0{break;}hash.update(&block[..n]);}
        if format!("{:x}",hash.finalize())!=expected{return Err("component checkpoint SHA mismatch".into());}
        Ok(())
    }
    pub fn open(path:impl AsRef<Path>)->Result<Self>{
        let path=path.as_ref().to_path_buf();let mut file=File::open(&path)?;let file_bytes=file.metadata()?.len();
        let mut prefix=[0;8];file.read_exact(&mut prefix)?;let count=u64::from_le_bytes(prefix);
        if count==0 || count>64<<20 || count.checked_add(8).is_none_or(|n|n>file_bytes){
            return Err("invalid safetensors header length".into());
        }
        let mut header=vec![0;count.try_into()?];file.read_exact(&mut header)?;
        let header:Value=serde_json::from_slice(&header)?;
        if !header.is_object(){return Err("safetensors header must be an object".into());}
        Ok(Self{path,header,data_start:count+8,file_bytes,tokenizer:RefCell::new(None),
            tokenizer_seconds:Cell::new(0.),#[cfg(feature="cuda")] pinned:None,
            #[cfg(feature="cuda")] admitted:None,#[cfg(feature="cuda")] retained:RefCell::new(Default::default())})
    }
    pub fn tensor(&self,name:&str,expected_sha256:Option<&str>)->Result<Vec<u8>>{
        let info=self.header.get(name).ok_or_else(||format!("checkpoint has no tensor {name}"))?;
        let offsets=info["data_offsets"].as_array().ok_or("tensor offsets")?;
        if offsets.len()!=2{return Err("tensor offset pair".into());}
        let start=offsets[0].as_u64().ok_or("tensor range start")?;
        let end=offsets[1].as_u64().ok_or("tensor range end")?;
        if end<start || self.data_start.checked_add(end).is_none_or(|n|n>self.file_bytes){
            return Err("tensor range escapes checkpoint".into());
        }
        #[cfg(feature="cuda")]
        let pinned=self.pinned.as_ref().map(|p|p.slice(usize::try_from(self.data_start+start)?..
            usize::try_from(self.data_start+end)?).map(<[u8]>::to_vec)).transpose()?;
        #[cfg(not(feature="cuda"))]
        let pinned:Option<Vec<u8>>=None;
        let data=if let Some(data)=pinned{data}else{
            let mut data=vec![0;usize::try_from(end-start)?];
            let mut file=File::open(&self.path)?;file.seek(SeekFrom::Start(self.data_start+start))?;file.read_exact(&mut data)?;data
        };
        #[cfg(feature="cuda")]
        if let Some((prefix,manifest))=&self.admitted {
            let entry=manifest.tensor(&format!("{prefix}.{name}"))?;
            if sha(&data)!=entry.sha256{return Err(format!("admitted CPU asset SHA mismatch: {name}").into());}
        }
        if expected_sha256.is_some_and(|expected|sha(&data)!=expected){return Err(format!("tensor SHA mismatch: {name}").into());}
        Ok(data)
    }
    pub fn asset(&self,key:&str)->Result<Vec<u8>>{
        let dtype=self.header[key]["dtype"].as_str().ok_or("asset dtype")?;
        if !matches!(dtype,"U8"|"I8"){return Err("packed asset must be byte-valued".into());}
        self.tensor(key,None)
    }
    /// This is immutable model metadata, never a cached request encoding or
    /// activation. The session already binds this checkpoint to a full SHA.
    pub fn tokenizer(&self)->Result<Rc<crate::frontend::GemmaTokenizer>>{
        if let Some(value)=self.tokenizer.borrow().as_ref(){return Ok(value.clone());}
        let started=std::time::Instant::now();
        let value=Rc::new(crate::frontend::GemmaTokenizer::from_checkpoint(self)?);
        self.tokenizer_seconds.set(started.elapsed().as_secs_f64());
        *self.tokenizer.borrow_mut()=Some(value.clone());Ok(value)
    }
    pub fn tokenizer_loaded(&self)->bool{self.tokenizer.borrow().is_some()}
    #[cfg(feature="cuda")]
    pub(crate) fn install_tokenizer(&self,prepared:native_core_infra::startup_policy::CpuPrepared<crate::frontend::GemmaTokenizer>)->Result<()>{
        if self.tokenizer.borrow().is_some(){return Err("tokenizer preparation already installed".into());}
        self.tokenizer_seconds.set(prepared.seconds);*self.tokenizer.borrow_mut()=Some(Rc::new(prepared.value));Ok(())
    }
    pub fn tokenizer_initialization_seconds(&self)->f64{self.tokenizer_seconds.get()}
    #[cfg(feature="cuda")]
    pub fn pin(&mut self,expected:&str,budget:&native_core_infra::pinned_weights::HostBudget)->Result<()>{
        self.pinned=Some(native_core_infra::pinned_weights::PinnedWeights::load(&self.path,expected,budget)?);Ok(())
    }
    #[cfg(feature="cuda")]
    pub(crate) fn admit(&mut self,prefix:&str,expected:&str,manifest:std::sync::Arc<native_core_infra::checkpoint::Manifest>,
        roots:&std::collections::HashMap<String,PathBuf>)->Result<()> {
        let mut count=0;
        for (key,info) in self.header.as_object().ok_or("checkpoint header")? {
            if key=="__metadata__"{continue;}
            let entry=manifest.tensor(&format!("{prefix}.{key}"))?;
            let file=&manifest.files[&entry.file];
            let offsets=info["data_offsets"].as_array().ok_or("weight offsets")?;
            if offsets.len()!=2 || roots.get(&file.root).ok_or("checkpoint root")?.join(&file.path)!=self.path
                || file.sha256!=expected || file.bytes!=self.file_bytes
                || offsets[0].as_u64().and_then(|n|n.checked_add(self.data_start))!=Some(entry.offset)
                || offsets[1].as_u64().and_then(|n|n.checked_sub(offsets[0].as_u64()?))!=Some(entry.bytes as u64)
                || info["dtype"].as_str()!=Some(entry.dtype.name()) || info["shape"]!=serde_json::json!(entry.shape) {
                return Err(format!("checkpoint header differs from admitted metadata: {prefix}.{key}").into());
            }
            count+=1;
        }
        if manifest.tensors.keys().filter(|key|key.starts_with(&format!("{prefix}."))).count()!=count {
            return Err("checkpoint tensor set differs from admitted metadata".into());
        }
        self.admitted=Some((prefix.into(),manifest));Ok(())
    }
    #[cfg(feature="cuda")]
    pub(crate) fn retain(&self,key:&str,name:&str,host:std::sync::Arc<native_core_infra::model_load::HostWeights>) {
        self.retained.borrow_mut().insert(key.into(),(name.into(),host));
    }
    #[cfg(feature="cuda")]
    pub(crate) fn retained_hosts(&self)->Vec<std::sync::Arc<native_core_infra::model_load::HostWeights>>{
        let mut seen=std::collections::HashSet::new();
        self.retained.borrow().values().filter_map(|(_,host)|
            seen.insert(std::sync::Arc::as_ptr(host) as usize).then(||host.clone())).collect()
    }
    #[cfg(feature="cuda")]
    pub(crate) fn retained_bytes(&self)->usize {
        self.retained.borrow().keys().map(|key|{
            let offsets=self.header[key]["data_offsets"].as_array().unwrap();
            (offsets[1].as_u64().unwrap()-offsets[0].as_u64().unwrap()) as usize
        }).sum()
    }
    #[cfg(feature="cuda")]
    pub(crate) fn weight_layout(&self,key:&str)->Result<(Vec<usize>,String)>{
        let info=&self.header[key];
        let dtype=match info["dtype"].as_str(){Some("BF16")=>"torch.bfloat16",Some("F32")=>"torch.float32",
            _=>return Err(format!("unported checkpoint weight {key}").into())};
        let shape=info["shape"].as_array().ok_or("weight shape")?.iter()
            .map(|v|v.as_u64().and_then(|n|usize::try_from(n).ok()).ok_or("weight dimension"))
            .collect::<std::result::Result<Vec<_>,_>>()?;
        Ok((shape,dtype.into()))
    }
    #[cfg(feature="cuda")]
    pub(crate) fn prefetch_weights(&self,keys:&[String],device:&crate::cuda::Device)->Result<()>{
        use crate::{cuda::Buffer,tensor::{self,Tensor}};
        use std::rc::Rc;
        if keys.is_empty(){return Ok(());}
        let layouts=keys.iter().map(|key|self.weight_layout(key)).collect::<Result<Vec<_>>>()?;
        let sizes=layouts.iter().map(|(shape,dtype)|{
            shape.iter().try_fold(tensor::dtype(dtype)?.1,|n,&d|n.checked_mul(d).ok_or_else(||"weight size overflow".into()))
        }).collect::<Result<Vec<_>>>()?;
        let (offsets,bytes)=native_core_infra::weight_loader::tensor_allocation_layout(&sizes)?;
        // One owned slab per checkpoint. The Gemma checkpoint can still be
        // released independently before AV sampling. Poison once, then copy
        // complete verified tensors; no per-tensor allocator/memset launches.
        let storage=Rc::new(Buffer::poisoned(bytes,device)?);
        let mut retained_uploads=Vec::new();
        for ((key,(shape,dtype)),offset) in keys.iter().zip(layouts).zip(offsets){
            let strides=tensor::contiguous_strides(&shape);
            let value=Tensor{storage:storage.clone(),offset,shape,strides,dtype,reference_offset:0};
            if let Some((name,host))=self.retained.borrow().get(key) {
                retained_uploads.push((name.clone(),host.clone(),value.ptr()));
            }else{self.upload(key,&value,device)?;}
            device.cache_weight(format!("{}\0{key}",self.path.display()),&value);
        }
        let mut groups:Vec<(std::sync::Arc<native_core_infra::model_load::HostWeights>,Vec<(String,u64)>)>=Vec::new();
        for (name,host,dst) in retained_uploads {
            if let Some((_,jobs))=groups.iter_mut().find(|(owner,_)|std::sync::Arc::ptr_eq(owner,&host)){jobs.push((name,dst));}
            else{groups.push((host,vec![(name,dst)]));}
        }
        for (host,jobs) in groups {
            let jobs:Vec<_>=jobs.iter().map(|(name,dst)|(name.as_str(),*dst)).collect();
            unsafe{host.upload(&jobs,device.stream)}?;
        }
        Ok(())
    }
    #[cfg(feature="cuda")]
    pub fn upload(&self,key:&str,value:&crate::tensor::Tensor,device:&crate::cuda::Device)->Result<()>{
        let bytes=value.elements()?.checked_mul(crate::tensor::dtype(&value.dtype)?.1).ok_or("weight tensor extent overflow")?;
        if !value.is_contiguous() || value.offset.checked_add(bytes).is_none_or(|end|end>value.storage.bytes()){
            return Err("checkpoint upload escapes its tensor view".into());
        }
        if let Some((name,host))=self.retained.borrow().get(key) {
            unsafe{host.upload(&[(name.as_str(),value.ptr())],device.stream)}?;Ok(())
        }else if let Some(pinned)=&self.pinned{
            let offsets=self.header[key]["data_offsets"].as_array().ok_or("weight offsets")?;
            if offsets.len()!=2{return Err("weight offset pair".into());}
            let start=self.data_start.checked_add(offsets[0].as_u64().ok_or("weight start")?).ok_or("weight offset overflow")?;
            let end=self.data_start.checked_add(offsets[1].as_u64().ok_or("weight end")?).ok_or("weight offset overflow")?;
            if end.checked_sub(start)!=Some(bytes.try_into()?){return Err("tensor shape/dtype bytes differ from checkpoint payload".into());}
            let lease=unsafe{pinned.upload(usize::try_from(start)?..usize::try_from(end)?,value.ptr() as *mut _,bytes,device.stream)?};
            device.track_upload(lease);Ok(())
        }else{
            let data=self.tensor(key,None)?;
            if data.len()!=bytes || value.offset!=0 || value.storage.bytes()!=bytes{
                return Err("pageable checkpoint upload requires a complete owned tensor".into());
            }
            crate::cuda::upload_into(&value.storage,&data,device)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn refuses_overflowing_or_truncated_tensor_ranges(){
        let path=std::env::temp_dir().join(format!("ltx25-weight-test-{}",std::process::id()));
        let header=br#"{"broken":{"dtype":"U8","shape":[8],"data_offsets":[0,8]}}"#;
        let mut bytes=(header.len() as u64).to_le_bytes().to_vec();bytes.extend_from_slice(header);
        std::fs::write(&path,bytes).unwrap();
        let checkpoint=Checkpoint::open(&path).unwrap();
        assert!(checkpoint.tensor("broken",None).is_err());
        std::fs::remove_file(path).unwrap();
    }
}
