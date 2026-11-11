//! Golds written by reference/gold/sitecustomize.py: `<name>.bin` (contiguous bytes) and
//! `<name>.json` (dtype, shape, stride, sha256) per captured tensor.
use crate::Result;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, Deserialize)]
pub struct Meta {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub stride: Vec<usize>,
    pub sha256: String,
    pub bytes: usize,
}

pub struct Gold { pub root: PathBuf }

impl Gold {
    pub fn open(root: impl Into<PathBuf>) -> Self { Self { root: root.into() } }

    fn path(&self, name: &str, ext: &str) -> PathBuf { self.root.join(format!("{}.{ext}", name.replace('/', "_"))) }

    pub fn meta(&self, name: &str) -> Result<Meta> {
        let text = std::fs::read_to_string(self.path(name, "json")).map_err(|e| format!("gold {name}: {e}"))?;
        serde_json::from_str(&text).map_err(|e| format!("gold {name}: {e}"))
    }

    pub fn bytes(&self, name: &str) -> Result<Vec<u8>> {
        let data = std::fs::read(self.path(name, "bin")).map_err(|e| format!("gold {name}: {e}"))?;
        let meta = self.meta(name)?;
        if sha(&data) != meta.sha256 { return Err(format!("gold {name}: payload differs from its sha256")); }
        Ok(data)
    }
}

pub fn sha(data: &[u8]) -> String {
    let digest = Sha256::digest(data);
    digest.iter().map(|b| format!("{b:02x}")).collect()
}

/// One rung result: native bytes against the gold's sha256, with the first differing element.
#[derive(Debug, serde::Serialize)]
pub struct Comparison { pub name: String, pub equal: bool, pub native_sha256: String, pub gold_sha256: String,
                        pub first_diff_byte: Option<usize> }

pub fn compare(gold: &Gold, name: &str, native: &[u8]) -> Result<Comparison> {
    let meta = gold.meta(name)?;
    let native_sha = sha(native);
    let equal = native_sha == meta.sha256;
    let first_diff_byte = if equal { None } else {
        gold.bytes(name).ok().and_then(|g| g.iter().zip(native).position(|(a, b)| a != b).or(Some(g.len().min(native.len()))))
    };
    Ok(Comparison { name: name.into(), equal, native_sha256: native_sha, gold_sha256: meta.sha256, first_diff_byte })
}

pub fn exists(dir: &Path, name: &str) -> bool { dir.join(format!("{}.json", name.replace('/', "_"))).exists() }
