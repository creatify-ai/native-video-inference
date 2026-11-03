//! SHA-pinned backend composition inside one native owner process.
use crate::backend_abi::{self,Bytes,Call,Free,Version};
use serde::Deserialize;
use serde_json::Value;
use sha2::{Digest,Sha256};
use std::{collections::BTreeMap,ffi::{c_char,c_int,c_void,CStr,CString},fs::File,io::Read,path::{Path,PathBuf},ptr};
type Result<T> = std::result::Result<T,Box<dyn std::error::Error>>;
unsafe extern "C" {
    fn dlopen(name:*const c_char,flags:c_int)->*mut c_void;
    fn dlsym(handle:*mut c_void,name:*const c_char)->*mut c_void;
    fn dlerror()->*const c_char;
}
#[derive(Clone,Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Pin {pub path:PathBuf,pub sha256:String}
impl Pin {
    pub fn verify(&self)->Result<()>{
        if !self.path.is_absolute() || self.sha256.len()!=64 ||
            !self.sha256.bytes().all(|v|v.is_ascii_hexdigit() && !v.is_ascii_uppercase()){
            return Err("absolute native artifact path and SHA256 required".into());
        }
        let mut file=File::open(&self.path)?;let mut block=vec![0;16<<20];let mut digest=Sha256::new();
        loop{let count=file.read(&mut block)?;if count==0{break;}digest.update(&block[..count]);}
        if format!("{:x}",digest.finalize())!=self.sha256{return Err(format!("native artifact SHA mismatch: {}",self.path.display()).into());}
        Ok(())
    }
}
#[derive(Clone,Deserialize)]
#[serde(deny_unknown_fields)]
struct Definition {library:Pin,initialization:Pin,#[serde(default)] preload:Vec<Pin>,
    #[serde(default)] expected_versions:BTreeMap<String,Value>}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {schema:String,backends:BTreeMap<String,Definition>}
struct Library {call:Call,free:Free}
impl Library {
    fn open(definition:&Definition,initialize:bool)->Result<Self>{
        definition.library.verify()?;definition.initialization.verify()?;
        if definition.preload.len()>64{return Err("too many backend dependencies".into());}
        for pin in &definition.preload{pin.verify()?;}
        fn open(path:&Path)->Result<*mut c_void>{
            let name=CString::new(path.as_os_str().as_encoded_bytes())?;
            let handle=unsafe{dlopen(name.as_ptr(),2|8)}; // NOW | LOCAL | DEEPBIND
            if handle.is_null(){let error=unsafe{dlerror()};return Err(if error.is_null(){"native dlopen failed".into()}
                else{unsafe{CStr::from_ptr(error)}.to_string_lossy().into_owned().into()});}
            // Bounded one library closure per admitted backend, retained for
            // owner lifetime. CUDA registers process exit callbacks into DSOs.
            Ok(handle)
        }
        for pin in &definition.preload{open(&pin.path)?;}
        let handle=open(&definition.library.path)?;
        unsafe fn symbol(handle:*mut c_void,name:&CStr)->Result<*mut c_void>{
            let value=dlsym(handle,name.as_ptr());
            if value.is_null(){return Err(format!("native backend lacks {}",name.to_string_lossy()).into());}Ok(value)
        }
        let version:Version=unsafe{std::mem::transmute(symbol(handle,c"native_backend_version_v1")?)};
        if unsafe{version()}!=backend_abi::VERSION{return Err("native backend ABI differs".into());}
        let library=Self{
            call:unsafe{std::mem::transmute(symbol(handle,c"native_backend_call_v1")?)},
            free:unsafe{std::mem::transmute(symbol(handle,c"native_backend_free_v1")?)},
        };
        if !definition.expected_versions.is_empty(){
            let versions=library.call(br#"{"op":"versions"}"#)?;
            if definition.expected_versions.iter().any(|(key,value)|versions.get(key)!=Some(value)){
                return Err("native backend resolved different runtime libraries".into());
            }
        }
        if initialize{
            let input=std::fs::read(&definition.initialization.path)?;
            library.call(&input)?;
        }Ok(library)
    }
    fn call(&self,input:&[u8])->Result<Value>{
        if input.is_empty() || input.len()>backend_abi::MAX_REQUEST_BYTES{return Err("native backend request extent".into());}
        let mut reply=Bytes{data:ptr::null_mut(),len:0};
        let status=unsafe{(self.call)(input.as_ptr(),input.len(),&mut reply)};
        if reply.data.is_null() || reply.len==0 || reply.len>backend_abi::MAX_REPLY_BYTES{
            return Err("native backend returned an invalid ABI buffer; retire owner".into());
        }
        let value=serde_json::from_slice::<Value>(unsafe{std::slice::from_raw_parts(reply.data,reply.len)});
        unsafe{(self.free)(reply.data,reply.len);}
        let value=value?;
        if status!=0 || value["ok"]!=true{return Err(format!("native backend failed: {}",value["error"]).into());}
        let details=value.get("details").ok_or("native backend reply lacks details")?;
        if details["owner_pid"].as_u64()!=Some(u64::from(std::process::id())){
            return Err("native backend ran in a different owner process".into());
        }
        if details["owner_instance"].as_str()!=Some(crate::process_identity::current().as_str()){
            return Err("native backend owner lifetime differs".into());
        }
        Ok(details.clone())
    }
}
pub struct Registry {definitions:BTreeMap<String,Definition>,loaded:BTreeMap<String,Library>}

fn check_status(status:&Value,allow_weight_preload:bool,allow_host_warm:bool)->Result<()>{
    if status["failed"]==true{return Err("native backend failed; retire owner".into());}
    let host_only=status["phase"]=="host_warm" && allow_host_warm;
    let preload_safe=matches!(status["phase"].as_str(),Some("decode_tail"|"complete")) ||
        status["phase"]==crate::phase_gate::DECODE_WEIGHTS_RETIRED;
    if status["busy"]==true && !host_only && !(allow_weight_preload && preload_safe){
        return Err("another native backend still owns GPU compute".into());
    }
    Ok(())
}

impl Registry {
    pub fn from_environment()->Result<Option<Self>>{
        let path=std::env::var("NATIVE_BACKENDS_MANIFEST");let sha=std::env::var("NATIVE_BACKENDS_MANIFEST_SHA256");
        match (path,sha){
            (Err(_),Err(_))=>Ok(None),
            (Ok(path),Ok(sha))=>Self::read(Pin{path:path.into(),sha256:sha}).map(Some),
            _=>Err("native backend manifest path/SHA must be supplied together".into()),
        }
    }
    pub fn read(pin:Pin)->Result<Self>{
        pin.verify()?;let manifest:Manifest=serde_json::from_slice(&std::fs::read(pin.path)?)?;
        if manifest.schema!="native-backends-v1" || manifest.backends.is_empty() || manifest.backends.len()>16{
            return Err("native backend manifest schema or capacity differs".into());
        }
        if manifest.backends.keys().any(|name|name.is_empty() || !name.bytes().all(|v|v.is_ascii_alphanumeric() || b"-_".contains(&v))){
            return Err("invalid native backend name".into());
        }
        Ok(Self{definitions:manifest.backends,loaded:BTreeMap::new()})
    }
    pub fn call(&mut self,name:&str,input:&str)->Result<Value>{
        if !self.loaded.contains_key(name){
            let definition=self.definitions.get(name).ok_or("native backend is not admitted")?;
            let library=Library::open(definition,true)?;self.loaded.insert(name.into(),library);
        }
        self.loaded[name].call(input.as_bytes())
    }
    pub fn admits(&self,name:&str)->bool{self.definitions.contains_key(name)}
    pub fn inspect(&self,name:&str)->Result<Value>{
        let definition=self.definitions.get(name).ok_or("native backend is not admitted")?;
        Library::open(definition,false)?.call(br#"{"op":"versions"}"#)
    }
    pub fn check_other_backends(&self,allow_weight_preload:bool)->Result<()>{
        for library in self.loaded.values(){
            let status=library.call(br#"{"op":"status"}"#)?;
            check_status(&status,allow_weight_preload,crate::weight_loader::flag("NATIVE_RESIDENT_BACKGROUND_HOST_LOAD")?)?;
        }Ok(())
    }
    pub fn yield_idle_backends(&self)->Result<()>{
        self.check_other_backends(true)?;
        for library in self.loaded.values(){
            let status=library.call(br#"{"op":"status"}"#)?;
            if status["busy"]==false{library.call(br#"{"op":"evict"}"#)?;}
        }Ok(())
    }
}

#[cfg(test)]
mod tests{
    use super::*;
    #[test]
    fn terminal_retirement_allows_weight_preload_but_never_another_forward(){
        for phase in ["admitted","dit","decode_boundary","decode"]{
            let status=serde_json::json!({"busy":true,"failed":false,"phase":phase});
            assert!(check_status(&status,true,false).is_err());
        }
        for phase in [crate::phase_gate::DECODE_WEIGHTS_RETIRED,"decode_tail","complete"]{
            let status=serde_json::json!({"busy":true,"failed":false,"phase":phase});
            assert!(check_status(&status,true,false).is_ok());
            assert!(check_status(&status,false,false).is_err());
        }
        let failed=serde_json::json!({"busy":false,"failed":true,"phase":"complete"});
        assert!(check_status(&failed,true,true).is_err());
        let warming=serde_json::json!({"busy":true,"failed":false,"phase":"host_warm"});
        assert!(check_status(&warming,true,false).is_err());
        assert!(check_status(&warming,false,true).is_ok());
    }
}
