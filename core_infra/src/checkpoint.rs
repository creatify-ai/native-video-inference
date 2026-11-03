//! In-place checkpoint manifests (schema `native-checkpoint-v1`, written by
//! core_infra/runtime/checkpoint.py). A published checkpoint is read as it is: every tensor is
//! a byte range of an unchanged safetensors or PyTorch zip file, with its own SHA-256 and the
//! trusted first/last 1 KiB guard, so a range becomes one bounded reader job.
use crate::integrity::Expected;
use crate::weight_loader::Job;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

pub type Result<T> = std::result::Result<T, String>;

pub const SCHEMA: &str = "native-checkpoint-v1";

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Dtype { F64, F32, F16, BF16, I64, I32, I16, I8, U8, Bool }
impl Dtype {
    pub fn parse(name: &str) -> Result<Self> {
        Ok(match name {
            "F64" => Self::F64, "F32" => Self::F32, "F16" => Self::F16, "BF16" => Self::BF16, "I64" => Self::I64,
            "I32" => Self::I32, "I16" => Self::I16, "I8" => Self::I8, "U8" => Self::U8, "BOOL" => Self::Bool,
            _ => return Err(format!("unsupported checkpoint dtype {name}")),
        })
    }
    pub fn size(self) -> usize {
        match self { Self::F64 | Self::I64 => 8, Self::F32 | Self::I32 => 4, Self::F16 | Self::BF16 | Self::I16 => 2,
                     Self::I8 | Self::U8 | Self::Bool => 1 }
    }
    pub fn name(self) -> &'static str {
        match self { Self::F64 => "F64", Self::F32 => "F32", Self::F16 => "F16", Self::BF16 => "BF16", Self::I64 => "I64",
                     Self::I32 => "I32", Self::I16 => "I16", Self::I8 => "I8", Self::U8 => "U8", Self::Bool => "BOOL" }
    }
}

#[derive(Clone, Debug)]
pub struct FileEntry { pub root: String, pub path: String, pub bytes: u64, pub sha256: String }

#[derive(Clone, Debug)]
pub struct TensorEntry {
    pub file: String,
    pub offset: u64,
    pub bytes: usize,
    pub dtype: Dtype,
    pub shape: Vec<usize>,
    pub sha256: String,
    pub guard: Expected,
}
impl TensorEntry {
    pub fn elements(&self) -> usize { self.shape.iter().product() }
}

#[derive(Clone)]
pub struct Manifest { pub files: BTreeMap<String, FileEntry>, pub tensors: BTreeMap<String, TensorEntry> }

/// A service's trusted offline metadata. Both identity and root declarations
/// are required; a light load must not invent guards from live weight bytes.
pub struct Declaration {pub manifest:Manifest,pub roots:HashMap<String,PathBuf>}
/// The same admission contract for a resident ABI initialization, standalone
/// model entry point, or service environment. It carries metadata, not weights.
#[derive(Clone,serde::Deserialize,serde::Serialize)]
pub struct Source {pub manifest:PathBuf,pub sha256:String,pub roots:HashMap<String,PathBuf>}
impl Source {
    pub fn load(&self)->Result<Declaration>{
        crate::startup_trace::configure_from_env();
        let _trace=crate::startup_trace::Span::new("startup_metadata","checkpoint declaration",0);
        if !self.manifest.is_absolute() || self.roots.is_empty() || self.roots.values().any(|p|!p.is_absolute()){
            return Err("checkpoint declaration requires absolute manifest and root paths".into());
        }
        let manifest=Manifest::load(&self.manifest,Some(&digest(&self.sha256)?))?;
        if manifest.files.values().any(|file|!self.roots.contains_key(&file.root)){
            return Err("checkpoint declaration lacks a source root".into());
        }
        Ok(Declaration{manifest,roots:self.roots.clone()})
    }
}
impl Declaration {
    pub fn from_env()->Result<Option<Self>> {
        let path=std::env::var_os("NATIVE_CHECKPOINT_MANIFEST");
        let sha=std::env::var("NATIVE_CHECKPOINT_SHA256").ok();
        let roots=std::env::var("NATIVE_CHECKPOINT_ROOTS").ok();
        if path.is_none() && sha.is_none() && roots.is_none(){return Ok(None);}
        let path=path.ok_or("checkpoint manifest declaration missing")?;
        let sha=digest(&sha.ok_or("checkpoint manifest SHA256 declaration missing")?)?;
        let roots:HashMap<String,PathBuf>=serde_json::from_str(&roots.ok_or("checkpoint roots declaration missing")?)
            .map_err(|e|format!("checkpoint roots: {e}"))?;
        Source{manifest:path.into(),sha256:sha,roots}.load().map(Some)
    }
}

fn digest(text: &str) -> Result<String> {
    if text.len() != 64 || !text.bytes().all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()) {
        return Err(format!("invalid SHA-256 {text:?}"));
    }
    Ok(text.to_string())
}

impl Manifest {
    /// Parses a manifest; `expected_sha256` pins its exact bytes when the caller has a trusted digest.
    pub fn parse(bytes: &[u8], expected_sha256: Option<&str>) -> Result<Self> {
        if let Some(want) = expected_sha256 {
            if format!("{:x}", Sha256::digest(bytes)) != want { return Err("checkpoint manifest digest mismatch".into()); }
        }
        let value: serde_json::Value = serde_json::from_slice(bytes).map_err(|e| format!("checkpoint manifest: {e}"))?;
        if value["schema"] != SCHEMA || value["segment_bytes"] != 1024 {
            return Err(format!("not a {SCHEMA} manifest"));
        }
        let mut files = BTreeMap::new();
        for (key, f) in value["files"].as_object().ok_or("manifest files missing")? {
            let text = |k: &str| f[k].as_str().map(str::to_string).ok_or_else(|| format!("{key}: {k} missing"));
            files.insert(key.clone(), FileEntry { root: text("root")?, path: text("path")?,
                bytes: f["bytes"].as_u64().ok_or_else(|| format!("{key}: bytes missing"))?, sha256: digest(&text("sha256")?)? });
        }
        let mut tensors = BTreeMap::new();
        for (name, t) in value["tensors"].as_object().ok_or("manifest tensors missing")? {
            let text = |k: &str| t[k].as_str().map(str::to_string).ok_or_else(|| format!("{name}: {k} missing"));
            let number = |k: &str| t[k].as_u64().ok_or_else(|| format!("{name}: {k} missing"));
            let entry = TensorEntry {
                file: text("file")?,
                offset: number("offset")?,
                bytes: usize::try_from(number("bytes")?).map_err(|_| format!("{name}: size overflow"))?,
                dtype: Dtype::parse(&text("dtype")?)?,
                shape: t["shape"].as_array().ok_or_else(|| format!("{name}: shape missing"))?.iter()
                    .map(|d| d.as_u64().map(|v| v as usize).ok_or_else(|| format!("{name}: bad shape"))).collect::<Result<_>>()?,
                sha256: digest(&text("sha256")?)?,
                guard: Expected { bytes: 0, head_sha256: digest(&text("head_sha256")?)?, tail_sha256: digest(&text("tail_sha256")?)? },
            };
            let file = files.get(&entry.file).ok_or_else(|| format!("{name}: unknown file {}", entry.file))?;
            let elements = entry.shape.iter().try_fold(1usize, |n, &d| n.checked_mul(d)).ok_or_else(|| format!("{name}: shape overflow"))?;
            if elements.checked_mul(entry.dtype.size()) != Some(entry.bytes) || entry.bytes == 0 {
                return Err(format!("{name}: {} bytes do not match {:?} {}", entry.bytes, entry.shape, entry.dtype.name()));
            }
            if entry.offset.checked_add(entry.bytes as u64).is_none_or(|end| end > file.bytes) {
                return Err(format!("{name}: range exceeds {}", entry.file));
            }
            let mut entry = entry;
            entry.guard.bytes = entry.bytes;
            tensors.insert(name.clone(), entry);
        }
        Ok(Self { files, tensors })
    }

    pub fn load(path: &Path, expected_sha256: Option<&str>) -> Result<Self> {
        let bytes = crate::startup_trace::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
        Self::parse(&bytes, expected_sha256)
    }

    /// Validate an alternate physical layout without changing one logical
    /// tensor, byte, dtype, shape, full SHA or foreground guard. File roots,
    /// container paths and offsets are storage details. No payload I/O occurs.
    pub fn validate_equivalent_tensors(&self,other:&Self)->Result<()> {
        if self.tensors.len()!=other.tensors.len(){return Err("storage layout tensor scope differs".into());}
        for (name,tensor)in &self.tensors {
            let physical=other.tensor(name)?;
            if tensor.bytes!=physical.bytes||tensor.dtype!=physical.dtype||tensor.shape!=physical.shape
                ||tensor.sha256!=physical.sha256||tensor.guard.bytes!=physical.guard.bytes
                ||tensor.guard.head_sha256!=physical.guard.head_sha256||tensor.guard.tail_sha256!=physical.guard.tail_sha256 {
                return Err(format!("storage layout changed logical tensor identity: {name}"));
            }
        }
        Ok(())
    }

    pub fn tensor(&self, name: &str) -> Result<&TensorEntry> {
        self.tensors.get(name).ok_or_else(|| format!("checkpoint has no tensor {name}"))
    }

    /// The reader job of one tensor: its range of the named file under `roots[file.root]`, with
    /// its SHA-256 (full verification) and light guard (guarded verification).
    pub fn job(&self, name: &str, roots: &HashMap<String, PathBuf>) -> Result<Job> {
        let tensor = self.tensor(name)?;
        let file = &self.files[&tensor.file];
        let root = roots.get(&file.root).ok_or_else(|| format!("no directory for checkpoint root {}", file.root))?;
        let mut job = Job::slice(root, &file.path, tensor.offset, tensor.bytes, tensor.sha256.clone(), file.bytes)?;
        job.guard = Some(tensor.guard.clone());
        Ok(job)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn sha(data: &[u8]) -> String { format!("{:x}", Sha256::digest(data)) }
    fn manifest(data: &[u8], offset: u64, bytes: usize, shape: &str) -> Vec<u8> {
        let payload = &data[offset as usize..offset as usize + bytes];
        format!(r#"{{"schema":"native-checkpoint-v1","segment_bytes":1024,
            "files":{{"checkpoint:m.bin":{{"root":"checkpoint","path":"m.bin","bytes":{},"sha256":"{}"}}}},
            "tensors":{{"w":{{"file":"checkpoint:m.bin","offset":{offset},"bytes":{bytes},"dtype":"BF16","shape":{shape},
            "sha256":"{}","head_sha256":"{}","tail_sha256":"{}"}}}}}}"#,
            data.len(), sha(data), sha(payload), sha(&payload[..payload.len().min(1024)]),
            sha(&payload[payload.len().saturating_sub(1024)..])).into_bytes()
    }
    #[test]fn service_source_requires_pinned_metadata_and_complete_roots(){
        let dir=std::env::temp_dir().join(format!("native-checkpoint-source-{}",std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let data=vec![7;64];let raw=manifest(&data,0,64,"[32]");let path=dir.join("manifest.json");
        std::fs::write(&path,&raw).unwrap();
        let mut source=Source{manifest:path,sha256:sha(&raw),roots:HashMap::from([("checkpoint".into(),dir.clone())])};
        assert_eq!(source.load().unwrap().manifest.tensors.len(),1);
        source.roots.clear();assert!(source.load().is_err());
        source.roots.insert("checkpoint".into(),dir.clone());source.sha256="0".repeat(64);
        assert!(source.load().is_err());std::fs::remove_dir_all(dir).unwrap();
    }
    #[test]fn storage_layout_preserves_complete_logical_identity() {
        let data=vec![7;5000];let canonical=Manifest::parse(&manifest(&data,13,4000,"[40,50]"),None).unwrap();
        let mut physical=canonical.clone();physical.tensors.get_mut("w").unwrap().offset=0;
        physical.files.get_mut("checkpoint:m.bin").unwrap().path="raw/w.bin".into();
        physical.files.get_mut("checkpoint:m.bin").unwrap().bytes=4000;
        physical.files.get_mut("checkpoint:m.bin").unwrap().sha256=canonical.tensors["w"].sha256.clone();
        canonical.validate_equivalent_tensors(&physical).unwrap();
        for change in 0..6 {
            let mut invalid=physical.clone();let value=invalid.tensors.get_mut("w").unwrap();
            match change{0=>value.dtype=Dtype::F16,1=>value.shape=vec![2000],2=>value.bytes=3999,
                3=>value.sha256="0".repeat(64),4=>value.guard.head_sha256="0".repeat(64),_=>value.guard.tail_sha256="0".repeat(64)};
            assert!(canonical.validate_equivalent_tensors(&invalid).is_err());
        }
        physical.tensors.clear();assert!(canonical.validate_equivalent_tensors(&physical).is_err());
    }
    #[test] fn ranges_become_verified_slice_jobs() {
        let dir = std::env::temp_dir().join(format!("native-checkpoint-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let data: Vec<u8> = (0..5000u32).map(|i| (i % 251) as u8).collect();
        std::fs::write(dir.join("m.bin"), &data).unwrap();
        let text = manifest(&data, 13, 4000, "[40,50]");
        let m = Manifest::parse(&text, Some(&sha(&text))).unwrap();
        let roots = HashMap::from([("checkpoint".to_string(), dir.clone())]);
        let job = m.job("w", &roots).unwrap();
        assert_eq!((job.file_offset, job.bytes, job.file_bytes), (13, 4000, Some(5000)));
        assert_eq!(job.guard.as_ref().unwrap().bytes, 4000);
        let config = crate::weight_loader::Config { workers: 2, chunk_bytes: 1000, pinned_slots: 2, budget_bytes: 6000 };
        let mut pool = crate::weight_loader::ReaderPool::start(vec![job], config).unwrap();
        let mut got = Vec::new();
        pool.consume_all(|_, _, chunk| { got.extend_from_slice(chunk); Ok(()) }).unwrap();
        assert_eq!(got, data[13..4013]);
        assert!(Manifest::parse(&text, Some(&sha(b"other"))).is_err());
        assert!(Manifest::parse(&manifest(&data, 13, 4000, "[40,49]"), None).is_err());
        assert!(Manifest::parse(b"{}", None).is_err());
        assert!(m.job("w", &HashMap::new()).is_err());
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
