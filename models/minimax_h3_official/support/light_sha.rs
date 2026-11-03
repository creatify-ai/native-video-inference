//! A trusted first/last 1 KiB guard. Full-file audit remains independent.
use sha2::{Digest,Sha256};
use std::{collections::BTreeMap,path::Path,sync::OnceLock};

#[derive(Clone)]
pub struct Expected {pub bytes:usize,pub head_sha256:String,pub tail_sha256:String}
struct Manifest {schema:String,segment_bytes:usize,full_sha256_verified:bool,entries:BTreeMap<String,Expected>}
static TABLE:OnceLock<Result<Option<Manifest>,String>>=OnceLock::new();
/// Only model payloads use this policy. Metadata and live request identities
/// keep their existing checks. An async payload must have a trusted guard.
pub fn async_enabled()->Result<bool,String>{
    match std::env::var("H3_MODEL_LOAD_ASYNC_SHA").as_deref(){
        Err(_)|Ok("0")=>Ok(false),Ok("1")=>{
            if std::env::var("H3_MODEL_LOAD_LIGHT_SHA").as_deref()!=Ok("1"){
                return Err("background model SHA requires the trusted light guard".into());
            }Ok(true)
        },_=>Err("H3_MODEL_LOAD_ASYNC_SHA must be 0 or 1".into()),
    }
}
pub fn verify_model_bytes(data:&[u8],full:&str)->Result<(),String>{
    if async_enabled()? {
        if expected(Some(full),data.len())?.is_none(){return Err("trusted model payload guard missing".into());}
        verify_bytes(data,full)
    }
    else if format!("{:x}",Sha256::digest(data))==full {Ok(())}
    else {Err("model payload SHA256 mismatch".into())}
}
fn load(path:&Path,wanted:&str)->Result<Manifest,String>{
    let raw=std::fs::read(path).map_err(|e|format!("light SHA manifest: {e}"))?;
    if format!("{:x}",Sha256::digest(&raw))!=wanted{return Err("light SHA manifest digest mismatch".into());}
    let value:serde_json::Value=serde_json::from_slice(&raw).map_err(|e|e.to_string())?;
    let mut entries=BTreeMap::new();
    for (key,item) in value["entries"].as_object().ok_or("light SHA entries missing")? {
        entries.insert(key.clone(),Expected{
            bytes:item["bytes"].as_u64().and_then(|n|usize::try_from(n).ok()).ok_or("light SHA size invalid")?,
            head_sha256:item["head_sha256"].as_str().ok_or("light SHA head missing")?.into(),
            tail_sha256:item["tail_sha256"].as_str().ok_or("light SHA tail missing")?.into(),
        });
    }
    let result=Manifest{schema:value["schema"].as_str().unwrap_or("").into(),
        segment_bytes:value["segment_bytes"].as_u64().and_then(|n|usize::try_from(n).ok()).unwrap_or(0),
        full_sha256_verified:value["full_sha256_verified"].as_bool().unwrap_or(false),entries};
    if result.schema!="h3-light-sha-v1"||result.segment_bytes!=1024||!result.full_sha256_verified {
        return Err("untrusted light SHA manifest".into());
    }
    for (digest,item) in &result.entries {
        for text in [digest,&item.head_sha256,&item.tail_sha256] {
            if text.len()!=64||!text.bytes().all(|c|c.is_ascii_hexdigit()&&!c.is_ascii_uppercase()){
                return Err("invalid light SHA digest".into());
            }
        }
    }
    Ok(result)
}
fn lookup(full:Option<&str>,bytes:usize,required:bool)->Result<Option<Expected>,String>{
    let table=TABLE.get_or_init(|| {
        match std::env::var("H3_MODEL_LOAD_LIGHT_SHA").as_deref(){
            Err(_)|Ok("0")=>return Ok(None),Ok("1")=>{},
            _=>return Err("H3_MODEL_LOAD_LIGHT_SHA must be 0 or 1".into()),
        }
        let path=std::env::var("H3_MODEL_LOAD_LIGHT_SHA_MANIFEST").map_err(|_|"light SHA manifest missing")?;
        let wanted=std::env::var("H3_MODEL_LOAD_LIGHT_SHA_MANIFEST_SHA256").map_err(|_|"light SHA manifest identity missing")?;
        load(Path::new(&path),&wanted).map(Some)
    });
    match table {
        Err(error)=>Err(error.clone()),Ok(None)=>Ok(None),Ok(Some(manifest))=>{
            let full=full.ok_or("light SHA requires a full manifest digest")?;
            let Some(entry)=manifest.entries.get(full) else {
                return if required{Err("weight missing from trusted light SHA manifest".into())}else{Ok(None)};
            };
            if entry.bytes!=bytes{return Err("light SHA weight size mismatch".into());}
            Ok(Some(entry.clone()))
        },
    }
}
pub fn expected(full:Option<&str>,bytes:usize)->Result<Option<Expected>,String>{lookup(full,bytes,true)}
fn compare(data:&[u8],wanted:&str,part:&str)->Result<(),String>{
    if format!("{:x}",Sha256::digest(data))!=wanted{return Err(format!("light SHA mismatch: {part} 1 KiB"));}Ok(())
}
pub fn verify_bytes(data:&[u8],full:&str)->Result<(),String>{
    if let Some(expected)=expected(Some(full),data.len())? {
        let n=data.len().min(1024);
        compare(&data[..n],&expected.head_sha256,"first")?;
        compare(&data[data.len()-n..],&expected.tail_sha256,"last")?;
    }
    Ok(())
}
/// Legacy VAE reads also include validation inputs and gold fixtures. Keep a
/// full foreground check for these entries, which are not model weights.
pub fn verify_bytes_or_full(data:&[u8],full:&str)->Result<(),String>{
    if let Some(expected)=lookup(Some(full),data.len(),false)? {
        let n=data.len().min(1024);
        compare(&data[..n],&expected.head_sha256,"first")?;
        compare(&data[data.len()-n..],&expected.tail_sha256,"last")?;
    }else if matches!(TABLE.get(),Some(Ok(Some(_)))) {
        if format!("{:x}",Sha256::digest(data))!=full{return Err("auxiliary tensor SHA mismatch".into());}
    }
    Ok(())
}
pub struct ReaderGuard {expected:Expected,head:Vec<u8>,tail:Vec<u8>}
impl ReaderGuard {
    pub fn new(expected:Expected)->Self{Self{expected,head:Vec::with_capacity(1024),tail:Vec::with_capacity(1024)}}
    pub fn observe(&mut self,offset:usize,data:&[u8]){
        let head_end=self.expected.bytes.min(1024);
        if offset<head_end{self.head.extend_from_slice(&data[..data.len().min(head_end-offset)]);}
        let tail_start=self.expected.bytes.saturating_sub(1024);
        if offset+data.len()>tail_start{self.tail.extend_from_slice(&data[tail_start.saturating_sub(offset)..]);}
    }
    pub fn finish(self)->Result<(),String>{
        if self.head.len()!=self.expected.bytes.min(1024)||self.tail.len()!=self.expected.bytes.min(1024){return Err("incomplete light SHA segments".into());}
        compare(&self.head,&self.expected.head_sha256,"first")?;
        compare(&self.tail,&self.expected.tail_sha256,"last")
    }
}
#[cfg(test)] mod tests {
    use super::*;
    fn specification(data:&[u8])->Expected{let n=data.len().min(1024);Expected{bytes:data.len(),head_sha256:format!("{:x}",Sha256::digest(&data[..n])),tail_sha256:format!("{:x}",Sha256::digest(&data[data.len()-n..]))}}
    fn check(data:&[u8],expected:Expected,chunk:usize)->Result<(),String>{let mut guard=ReaderGuard::new(expected);for(offset,part)in data.chunks(chunk).enumerate(){guard.observe(offset*chunk,part);}guard.finish()}
    #[test]fn covers_tiny_files_overlapping_segments_and_chunk_boundaries(){
        for size in [0,1,17,1023,1024,1025,2047,2048,4097,5001]{let data=(0..size).map(|n|(n%251)as u8).collect::<Vec<_>>();for chunk in [17,257,1024,4096]{assert!(check(&data,specification(&data),chunk).is_ok());}}
    }
    #[test]fn rejects_head_and_tail_and_leaves_middle_to_full_audit(){
        let data=vec![7;5001];for index in [0,1023,3977,5000]{let mut bad=data.clone();bad[index]^=1;assert!(check(&bad,specification(&data),257).is_err());}
        let mut middle=data.clone();middle[2500]^=1;assert!(check(&middle,specification(&data),257).is_ok());assert_ne!(Sha256::digest(&middle),Sha256::digest(&data));
        let mut partial=ReaderGuard::new(specification(&data));partial.observe(0,&data[..17]);assert!(partial.finish().is_err());
    }
    #[test]fn manifest_requires_its_pinned_digest_and_full_verification(){
        let path=std::env::temp_dir().join(format!("h3-light-manifest-test-{}-{}",std::process::id(),module_path!().replace("::","-")));
        let valid=serde_json::json!({"schema":"h3-light-sha-v1","segment_bytes":1024,"full_sha256_verified":true,"entries":{}});
        let raw=serde_json::to_vec(&valid).unwrap();std::fs::write(&path,&raw).unwrap();
        assert!(load(&path,&format!("{:x}",Sha256::digest(&raw))).is_ok());
        assert!(load(&path,&"0".repeat(64)).is_err());
        for (field,value) in [("full_sha256_verified",serde_json::json!(false)),("segment_bytes",serde_json::json!(2048)),("schema",serde_json::json!("unknown"))] {
            let mut bad=valid.clone();bad[field]=value;let raw=serde_json::to_vec(&bad).unwrap();std::fs::write(&path,&raw).unwrap();
            assert!(load(&path,&format!("{:x}",Sha256::digest(&raw))).is_err());
        }
        std::fs::remove_file(path).unwrap();
    }
}
