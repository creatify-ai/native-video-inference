//! Contiguous raw tensor metadata and byte decoding. Legacy dtype labels remain valid.
use serde::Deserialize;
use serde_json::Value;
use sha2::{Digest,Sha256};
use std::path::{Component,Path};
use crate::Result;
#[derive(Clone, Debug, Deserialize, serde::Serialize)]
pub struct TensorSpec {
    pub file: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub stride: Vec<usize>,
    pub bytes: usize,
    pub contiguous: bool,
    pub storage_offset: usize,
    #[serde(default)]
    pub sha256: Option<String>,
}
impl TensorSpec {
    pub fn parse(value: &Value) -> Result<Self> {
        let result: Self = serde_json::from_value(value.clone())?;
        result.validate()?;
        Ok(result)
    }
    pub fn width(&self) -> Result<usize> {
        match self.dtype.as_str() {
            "torch.bfloat16" => Ok(2), "torch.float32" | "torch.int32" => Ok(4),
            "torch.int64" | "torch.float64" => Ok(8),
            _ => Err(format!("unsupported exported dtype {}", self.dtype).into()),
        }
    }
    pub fn elements(&self) -> Result<usize> {
        self.shape.iter().try_fold(1usize, |n, x| n.checked_mul(*x).ok_or_else(|| "tensor size overflow".into()))
    }
    pub fn validate(&self) -> Result<()> {
        if !self.contiguous || self.storage_offset != 0 || self.shape.len() != self.stride.len() {
            return Err("raw tensor must describe its stored contiguous layout".into());
        }
        let mut stride = 1usize;
        for (&dim, &actual) in self.shape.iter().zip(&self.stride).rev() {
            if dim > 1 && actual != stride { return Err("stored tensor stride is not contiguous".into()); }
            stride = stride.checked_mul(dim).ok_or("stride overflow")?;
        }
        if self.elements()?.checked_mul(self.width()?).ok_or("byte size overflow")? != self.bytes || self.bytes == 0 {
            return Err("exported tensor byte size mismatch".into());
        }
        if Path::new(&self.file).components().any(|p| !matches!(p, Component::Normal(_))) {
            return Err("raw tensor path must stay under the export root".into());
        }
        Ok(())
    }
    pub fn read(&self, root: &Path) -> Result<Vec<u8>> {
        self.validate()?;
        let data = crate::startup_trace::read(root.join(&self.file))?;
        if data.len() != self.bytes { return Err(format!("wrong file size: {}", self.file).into()); }
        if let Some(expected)=&self.sha256 {
            if format!("{:x}",crate::startup_trace::measure("sha256","",(&data).len(),||Sha256::digest(&data)))!=*expected {
                return Err(format!("raw tensor SHA256 mismatch: {}",self.file).into());
            }
        }
        Ok(data)
    }
}

pub fn f32_values(data: &[u8]) -> Result<Vec<f32>> {
    if data.len() % 4 != 0 { return Err("malformed float32 bytes".into()); }
    Ok(data.chunks_exact(4).map(|x| f32::from_le_bytes(x.try_into().unwrap())).collect())
}
pub fn i64_values(data: &[u8]) -> Result<Vec<i64>> {
    if data.len() % 8 != 0 { return Err("malformed int64 bytes".into()); }
    Ok(data.chunks_exact(8).map(|x| i64::from_le_bytes(x.try_into().unwrap())).collect())
}
pub fn bytes_i64(values: &[i64]) -> Vec<u8> { values.iter().flat_map(|x| x.to_le_bytes()).collect() }

