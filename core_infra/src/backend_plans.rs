//! Bounded publication of SHA-pinned backend assets. This is CPU-only work;
//! numerical operators still check their complete layout/runtime cache keys.
use serde::Deserialize;
use serde_json::{json,Value};
use sha2::{Digest,Sha256};
use std::{collections::BTreeMap,fs,path::{Path,PathBuf},sync::atomic::{AtomicUsize,Ordering}};

type Result<T>=std::result::Result<T,String>;
const MAX_BYTES:usize=128<<20;
const MAX_FILES:usize=128;
#[derive(Clone,Debug)]
pub struct Pin {pub manifest:PathBuf,pub sha256:String}
impl Pin {
    pub fn from_env()->Result<Option<Self>>{
        match (std::env::var_os("NATIVE_BACKEND_PLANS_MANIFEST"),std::env::var("NATIVE_BACKEND_PLANS_SHA256")){
            (None,Err(std::env::VarError::NotPresent))=>Ok(None),
            (Some(manifest),Ok(sha256))=>Ok(Some(Self{manifest:manifest.into(),sha256})),
            _=>Err("backend plan manifest path and SHA256 must be supplied together".into()),
        }
    }
}
#[derive(Deserialize)]
struct Manifest {files:BTreeMap<String,String>}
fn sha(bytes:&[u8])->String{format!("{:x}",Sha256::digest(bytes))}
fn digest(value:&str)->bool{value.len()==64&&value.bytes().all(|c|c.is_ascii_digit()||(b'a'..=b'f').contains(&c))}
fn read(path:&Path,limit:usize)->Result<Vec<u8>>{
    use std::io::Read;
    if !fs::symlink_metadata(path).map_err(|e|e.to_string())?.file_type().is_file(){return Err("backend asset must be a regular file".into());}
    let mut file=fs::File::open(path).map_err(|e|e.to_string())?;
    let bytes=usize::try_from(file.metadata().map_err(|e|e.to_string())?.len()).map_err(|_|"backend asset extent")?;
    if bytes>limit{return Err("backend asset byte budget exceeded".into());}
    let mut data=vec![0;bytes];file.read_exact(&mut data).map_err(|e|e.to_string())?;
    if file.read(&mut [0u8;1]).map_err(|e|e.to_string())?!=0{return Err("backend asset extent changed".into());}Ok(data)
}
struct Directory(PathBuf);
impl Directory {
    fn create()->Result<Self>{
        static NEXT:AtomicUsize=AtomicUsize::new(0);
        for _ in 0..128 {
            let path=std::env::temp_dir().join(format!("native-backend-plans-{}-{}",std::process::id(),NEXT.fetch_add(1,Ordering::Relaxed)));
            match fs::create_dir(&path){Ok(())=>return Ok(Self(path)),Err(e)if e.kind()==std::io::ErrorKind::AlreadyExists=>{},Err(e)=>return Err(e.to_string())}
        }Err("backend asset directory capacity exceeded".into())
    }
}
impl Drop for Directory {fn drop(&mut self){let _=fs::remove_dir_all(&self.0);}}
pub struct Published {directory:Option<Directory>,manifest_sha256:Option<String>,files:usize,bytes:usize}
impl Published {
    pub fn prepare(pin:Option<&Pin>)->Result<Self>{
        let Some(pin)=pin else{return Ok(Self{directory:None,manifest_sha256:None,files:0,bytes:0});};
        if !pin.manifest.is_absolute()||!digest(&pin.sha256){return Err("absolute backend manifest and lowercase SHA256 required".into());}
        let raw=read(&pin.manifest,1<<20)?;
        if sha(&raw)!=pin.sha256{return Err("backend manifest SHA256 mismatch".into());}
        let manifest:Manifest=serde_json::from_slice(&raw).map_err(|e|e.to_string())?;
        if manifest.files.is_empty()||manifest.files.len()>MAX_FILES{return Err("backend asset file budget exceeded".into());}
        let root=pin.manifest.parent().ok_or("backend manifest directory")?;
        let directory=Directory::create()?;let staging=directory.0.join("pending");fs::create_dir(&staging).map_err(|e|e.to_string())?;
        let mut bytes=0usize;
        for (name,expected)in &manifest.files {
            if Path::new(name).file_name().and_then(|v|v.to_str())!=Some(name.as_str())||name=="."||name==".."||!digest(expected){
                return Err("invalid backend asset filename or SHA256".into());
            }
            let data=read(&root.join(name),MAX_BYTES-bytes)?;bytes+=data.len();
            if sha(&data)!=*expected{return Err("backend asset SHA256 mismatch".into());}
            fs::write(staging.join(name),data).map_err(|e|e.to_string())?;
        }
        fs::rename(staging,directory.0.join("plans")).map_err(|e|e.to_string())?;
        Ok(Self{directory:Some(directory),manifest_sha256:Some(pin.sha256.clone()),files:manifest.files.len(),bytes})
    }
    pub fn path(&self)->Option<PathBuf>{self.directory.as_ref().map(|v|v.0.join("plans"))}
    pub fn receipt(&self)->Value{json!({"state":if self.directory.is_some(){"configured"}else{"native_builders"},
        "manifest_sha256":self.manifest_sha256,"files":self.files,"bytes":self.bytes,"forward_executed":false})}
    pub fn validates(&self,value:&Value)->bool{*value==self.receipt()}
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture()->(Directory,Pin){
        let root=Directory::create().unwrap();let bytes=b"unchanged execution plan";
        fs::write(root.0.join("one.plan"),bytes).unwrap();
        let raw=serde_json::to_vec(&json!({"files":{"one.plan":sha(bytes)}})).unwrap();
        let path=root.0.join("manifest.json");fs::write(&path,&raw).unwrap();
        (root,Pin{manifest:path,sha256:sha(&raw)})
    }
    #[test]fn publication_owns_verified_bytes_and_closes_only_its_directory(){
        let(source,pin)=fixture();let published=Published::prepare(Some(&pin)).unwrap();let path=published.path().unwrap();
        assert!(published.validates(&published.receipt()));assert_eq!(published.receipt()["forward_executed"],false);
        fs::write(source.0.join("one.plan"),b"changed").unwrap();
        assert_eq!(fs::read(path.join("one.plan")).unwrap(),b"unchanged execution plan");
        drop(published);assert!(!path.exists());assert!(source.0.exists());
    }
    #[test]fn changed_bundles_and_path_escapes_never_publish_backend_readiness(){
        let(source,mut pin)=fixture();pin.sha256="0".repeat(64);assert!(Published::prepare(Some(&pin)).is_err());
        let raw=serde_json::to_vec(&json!({"files":{"../outside.plan":"0".repeat(64)}})).unwrap();
        fs::write(&pin.manifest,&raw).unwrap();pin.sha256=sha(&raw);assert!(Published::prepare(Some(&pin)).is_err());
        let(_second,pin)=fixture();fs::write(pin.manifest.parent().unwrap().join("one.plan"),b"damaged").unwrap();
        assert!(Published::prepare(Some(&pin)).is_err());assert!(source.0.exists());
        let value=Published::prepare(None).unwrap();assert!(value.path().is_none());assert!(value.validates(&value.receipt()));
    }
}
