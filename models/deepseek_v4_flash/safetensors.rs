//! Read tensors from the published checkpoint's safetensors files, in place.
use crate::Result;
use serde_json::Value;
use std::collections::BTreeMap;
use std::fs::File;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};

#[derive(Clone, Debug)]
pub struct Entry {
    pub file: usize,
    pub dtype: String,
    pub shape: Vec<usize>,
    /// Absolute byte range inside the file.
    pub begin: u64,
    pub end: u64,
}

impl Entry {
    pub fn bytes(&self) -> usize { (self.end - self.begin) as usize }
}

pub struct Checkpoint {
    pub root: PathBuf,
    pub files: Vec<(PathBuf, File)>,
    pub tensors: BTreeMap<String, Entry>,
}

impl Checkpoint {
    /// Index every tensor listed by `model.safetensors.index.json` under `root`.
    pub fn open(root: &Path) -> Result<Self> {
        let index: Value = serde_json::from_slice(&std::fs::read(root.join("model.safetensors.index.json"))
            .map_err(|e| format!("index: {e}"))?).map_err(|e| format!("index: {e}"))?;
        let map = index["weight_map"].as_object().ok_or("index lacks weight_map")?;
        let mut names: Vec<String> = map.values().filter_map(|v| v.as_str().map(String::from)).collect();
        names.sort();
        names.dedup();
        let mut files = Vec::new();
        let mut tensors = BTreeMap::new();
        for (i, name) in names.iter().enumerate() {
            let path = root.join(name);
            let file = File::open(&path).map_err(|e| format!("{}: {e}", path.display()))?;
            let mut len = [0u8; 8];
            file.read_exact_at(&mut len, 0).map_err(|e| format!("{name}: {e}"))?;
            let header_len = u64::from_le_bytes(len);
            let mut header = vec![0u8; header_len as usize];
            file.read_exact_at(&mut header, 8).map_err(|e| format!("{name}: {e}"))?;
            let header: Value = serde_json::from_slice(&header).map_err(|e| format!("{name} header: {e}"))?;
            for (tensor, info) in header.as_object().ok_or("header is not an object")? {
                if tensor == "__metadata__" { continue; }
                let offsets = info["data_offsets"].as_array().ok_or("data_offsets")?;
                let base = 8 + header_len;
                tensors.insert(tensor.clone(), Entry {
                    file: i,
                    dtype: info["dtype"].as_str().unwrap_or("").to_string(),
                    shape: info["shape"].as_array().ok_or("shape")?.iter().map(|d| d.as_u64().unwrap_or(0) as usize).collect(),
                    begin: base + offsets[0].as_u64().ok_or("offset")?,
                    end: base + offsets[1].as_u64().ok_or("offset")?,
                });
            }
            files.push((path, file));
        }
        Ok(Self { root: root.to_path_buf(), files, tensors })
    }

    pub fn entry(&self, name: &str) -> Result<&Entry> {
        self.tensors.get(name).ok_or_else(|| format!("checkpoint has no tensor {name}"))
    }

    pub fn read(&self, name: &str) -> Result<Vec<u8>> {
        let e = self.entry(name)?;
        let mut out = vec![0u8; e.bytes()];
        self.files[e.file].1.read_exact_at(&mut out, e.begin).map_err(|err| format!("{name}: {err}"))?;
        Ok(out)
    }
}
