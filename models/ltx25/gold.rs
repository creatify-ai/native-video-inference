use crate::Result;
use serde::Deserialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, path::{Path, PathBuf}, rc::Rc, sync::Arc};

#[derive(Clone, Debug, Deserialize)]
pub struct Entry {
    pub file: Option<PathBuf>, pub dtype: String, pub shape: Vec<usize>,
    pub bytes: usize, pub sha256: String,
}
impl Entry {
    pub fn elements(&self) -> Result<usize> {
        self.shape.iter().try_fold(1usize, |n, &d| n.checked_mul(d).ok_or_else(|| "tensor overflow".into()))
    }
    pub fn validate_bytes(&self)->Result<()>{
        let width=match self.dtype.as_str(){"torch.bfloat16"|"torch.float16"|"torch.int16"=>2,
            "torch.float32"|"torch.int32"=>4,"torch.float64"|"torch.int64"|"torch.complex64"=>8,
            "torch.complex128"=>16,"torch.bool"|"torch.uint8"|"torch.int8"=>1,_=>return Err("unknown recorded dtype".into())};
        if self.elements()?.checked_mul(width)!=Some(self.bytes){return Err("recorded tensor is truncated: bytes differ from shape and dtype".into());}Ok(())
    }
}
pub struct Gold { pub root: PathBuf, pub tensors: Arc<BTreeMap<String, Entry>>, pub metadata: Arc<Value>, pub identity: Value,
    pub verify:bool, pub(crate) assets:Option<Rc<native_core_infra::immutable_assets::ImmutableAssets>> }
/// A parsed `Gold` for another thread, without its asset cache. Parsing a
/// profile manifest takes about a second; sharing it takes none.
pub(crate) struct Shared { root: PathBuf, tensors: Arc<BTreeMap<String, Entry>>, metadata: Arc<Value>, identity: Value, verify: bool }
impl Shared {
    pub(crate) fn root(&self) -> &Path { &self.root }
    pub(crate) fn gold(self, assets: Option<Rc<native_core_infra::immutable_assets::ImmutableAssets>>) -> Gold {
        Gold { root: self.root, tensors: self.tensors, metadata: self.metadata, identity: self.identity, verify: self.verify, assets }
    }
}
impl Gold {
    pub(crate) fn share(&self) -> Shared {
        Shared { root: self.root.clone(), tensors: self.tensors.clone(), metadata: self.metadata.clone(),
            identity: self.identity.clone(), verify: self.verify }
    }
    pub fn open(root: impl AsRef<Path>) -> Result<Self> {
        let root = root.as_ref().to_path_buf();
        let report: Value = serde_json::from_slice(&std::fs::read(root.join("report.json"))?)?;
        if report["passed"] != true { return Err("gold capture did not complete successfully".into()); }
        let accepted: Value = serde_json::from_slice(&std::fs::read(root.parent().ok_or("invalid gold root")?.join("accepted.json"))?)?;
        let runs=accepted["runs"].as_array().ok_or("gold acceptance has no runs")?;
        if accepted["passed"]!=true || runs.len()<3 || !runs.iter().any(|p|p.as_str()==root.to_str()){
            return Err("gold has not passed three-run determinism admission".into());
        }
        let raw=std::fs::read(root.join("manifest.json"))?;
        let run_name=root.file_name().and_then(|s|s.to_str()).ok_or("gold run name")?;
        if accepted["manifest_sha256"][run_name].as_str()!=Some(sha(&raw).as_str()){
            return Err("gold manifest changed after admission".into());
        }
        let manifest: Value = serde_json::from_slice(&raw)?;
        if manifest["identity"]["capture"]=="diffusion-video-components-v1"{
            return Err("superseded video capture hashed pre-write scratch; v2 admission is required".into());
        }
        if manifest["schema"] != "ltx25-gold-v1" { return Err("expected ltx25-gold-v1".into()); }
        let tensors:BTreeMap<String,Entry>=serde_json::from_value(manifest["tensors"].clone())?;
        for (name,entry) in &tensors{entry.validate_bytes().map_err(|error|format!("{name}: {error}"))?;}
        Ok(Self { root, tensors: Arc::new(tensors),
            metadata: Arc::new(manifest["metadata"].clone()), identity: manifest["identity"].clone(),verify:true,assets:None })
    }
    /// Deployment profiles contain programs, model constants and metadata.
    /// Every captured request activation has its blob removed by the exporter.
    /// A runtime profile's manifest, checked against its pinned SHA-256. Parsed once per process
    /// for each (directory, pin): later requests reuse the content that hash identifies.
    pub fn profile(root:impl AsRef<Path>,manifest_sha:&str)->Result<Self>{
        type Parsed=(Arc<BTreeMap<String,Entry>>,Arc<Value>,Value);
        static PARSED:std::sync::OnceLock<std::sync::Mutex<BTreeMap<(PathBuf,String),Parsed>>>=std::sync::OnceLock::new();
        let root=root.as_ref().to_path_buf();
        let cache=PARSED.get_or_init(Default::default);
        let key=(root.clone(),manifest_sha.to_owned());
        if let Some((tensors,metadata,identity))=cache.lock().map_err(|_|"profile cache poisoned")?.get(&key).cloned(){
            return Ok(Self{root,tensors,metadata,identity,verify:false,assets:None});
        }
        let value=Self::parse_profile(root,manifest_sha)?;
        cache.lock().map_err(|_|"profile cache poisoned")?
            .insert(key,(value.tensors.clone(),value.metadata.clone(),value.identity.clone()));
        Ok(value)
    }
    fn parse_profile(root:PathBuf,manifest_sha:&str)->Result<Self>{
        let raw=std::fs::read(root.join("manifest.json"))?;
        if sha(&raw)!=manifest_sha{return Err("native profile manifest checksum differs".into());}
        let manifest:Value=serde_json::from_slice(&raw)?;
        if manifest["schema"]!="ltx25-gold-v1" || manifest["identity"]["runtime_profile"]!=true{
            return Err("expected exported native runtime profile".into());
        }
        let tensors:BTreeMap<String,Entry>=serde_json::from_value(manifest["tensors"].clone())?;
        for entry in tensors.values(){entry.validate_bytes()?;}
        Ok(Self{root,tensors:Arc::new(tensors),metadata:Arc::new(manifest["metadata"].clone()),
            identity:manifest["identity"].clone(),verify:false,assets:None})
    }
    pub fn entry(&self, name: &str) -> Result<&Entry> {
        self.tensors.get(name).ok_or_else(|| format!("missing gold tensor {name}").into())
    }
    fn asset_path(&self,name:&str)->Result<&Path>{
        let entry = self.entry(name)?;
        let file = entry.file.as_ref().ok_or("tensor captured hash-only")?;
        let expected = self.root.parent().ok_or("invalid gold path")?.join("blobs").join(format!("{}.raw", entry.sha256));
        if file != &expected { return Err("gold blob path does not match its content address".into()); }
        Ok(file)
    }
    pub fn asset(&self,name:&str)->Result<Rc<Vec<u8>>>{
        if let Some(cache)=&self.assets{
            if self.verify{return Err("runtime asset cache cannot retain captured activations".into());}
            let entry=self.entry(name)?;
            return cache.load(self.asset_path(name)?,&entry.sha256,entry.bytes);
        }
        Ok(Rc::new(self.bytes(name)?))
    }
    pub fn bytes(&self, name: &str) -> Result<Vec<u8>> {
        if self.assets.is_some(){return Ok(self.asset(name)?.as_ref().clone());}
        let entry=self.entry(name)?;let file=self.asset_path(name)?;
        let bytes = std::fs::read(file)?;
        if bytes.len() != entry.bytes || sha(&bytes) != entry.sha256 { return Err(format!("corrupt tensor {name}").into()); }
        Ok(bytes)
    }
    pub(crate) fn warm_assets(&self)->Result<()>{
        for (name,entry) in self.tensors.iter(){if entry.file.is_some(){self.asset(name)?;}}Ok(())
    }
    pub fn scalar(&self, name: &str) -> Result<f32> {
        if let Some(value) = self.metadata.get(name).and_then(Value::as_f64) { return Ok(value as f32); }
        let data = self.bytes(name)?;
        if data.len() != 4 { return Err("expected scalar float32".into()); }
        Ok(f32::from_le_bytes(data.try_into().unwrap()))
    }
}
pub fn sha(bytes: &[u8]) -> String { format!("{:x}", Sha256::digest(bytes)) }

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rejects_four_gib_wrapped_recording(){
        let mut entry=Entry{file:None,dtype:"torch.bfloat16".into(),shape:vec![1,121,208,368,256],
            bytes:447086592,sha256:String::new()};
        assert!(entry.validate_bytes().is_err());
        entry.bytes=4742053888;assert!(entry.validate_bytes().is_ok());
    }
}
