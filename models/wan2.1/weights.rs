//! safetensors checkpoints (converted dit/t5/clip/vae files as the reference holds them in memory).
use crate::Result;
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

#[derive(Clone, Debug)]
pub struct Entry { pub dtype: String, pub shape: Vec<usize>, pub start: u64, pub end: u64 }

/// Named tensors: a safetensors checkpoint or a gold manifest (host bytes, uploaded on use), or
/// tensors already streamed to the device by core_infra::model_load (`DeviceWeights`).
pub trait Source: Sync {
    fn bytes(&self, name: &str) -> Result<Vec<u8>>;
    /// Every tensor name (sources that cannot enumerate return an error).
    fn names(&self) -> Result<Vec<String>> { Err("tensor source cannot list its names".into()) }
    /// A tensor's dtype in safetensors spelling (F32, BF16, ...).
    fn dtype(&self, name: &str) -> Result<String> { Err(format!("{name}: tensor source has no dtype metadata").into()) }
    /// The tensor on the device.
    #[cfg(feature = "cuda")]
    fn buffer(&self, name: &str, device: &crate::cuda::Device) -> Result<crate::cuda::Buffer> {
        crate::cuda::Buffer::upload(&self.bytes(name)?, device)
    }
    /// Many tensors on the device, in the order of `names`.
    #[cfg(feature = "cuda")]
    fn buffers(&self, names: &[String], device: &crate::cuda::Device) -> Result<Vec<crate::cuda::Buffer>> {
        self.bytes_many(names, 32)?.iter().map(|b| crate::cuda::Buffer::upload(b, device)).collect()
    }
    /// Many tensors, read concurrently, in the order of `names`.
    fn bytes_many(&self, names: &[String], threads: usize) -> Result<Vec<Vec<u8>>> {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Mutex;
        let next = AtomicUsize::new(0);
        let slots: Vec<Mutex<Option<std::result::Result<Vec<u8>, String>>>> = names.iter().map(|_| Mutex::new(None)).collect();
        std::thread::scope(|scope| {
            for _ in 0..threads.max(1) {
                scope.spawn(|| loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    if i >= names.len() { break; }
                    *slots[i].lock().unwrap() = Some(self.bytes(&names[i]).map_err(|e| e.to_string()));
                });
            }
        });
        slots.into_iter().map(|slot| slot.into_inner().unwrap().unwrap().map_err(|e| e.into())).collect()
    }
}
impl Source for SafeTensors {
    fn bytes(&self, name: &str) -> Result<Vec<u8>> { SafeTensors::bytes(self, name) }
    fn names(&self) -> Result<Vec<String>> { Ok(self.entries.keys().cloned().collect()) }
    fn dtype(&self, name: &str) -> Result<String> { Ok(self.entry(name)?.dtype.clone()) }
}

/// One component's tensors of a streaming load (core_infra::model_load), named without the
/// component prefix of the checkpoint manifest (`dit.`, `t5.`, ...). Buffers are views of the
/// shared allocation, which lives as long as the `Rc`.
#[cfg(feature = "cuda")]
pub struct DeviceWeights { pub loaded: std::rc::Rc<native_core_infra::model_load::Loaded>, pub prefix: String }
#[cfg(feature = "cuda")]
unsafe impl Sync for DeviceWeights {}
#[cfg(feature = "cuda")]
impl DeviceWeights {
    fn view(&self, name: &str) -> Result<&native_core_infra::model_load::View> {
        Ok(self.loaded.get(&format!("{}{name}", self.prefix))?)
    }
}
#[cfg(feature = "cuda")]
impl Source for DeviceWeights {
    fn bytes(&self, name: &str) -> Result<Vec<u8>> { Err(format!("{name}: device-resident tensor").into()) }
    fn names(&self) -> Result<Vec<String>> {
        Ok(self.loaded.tensors.keys().filter_map(|k| k.strip_prefix(&self.prefix).map(str::to_string)).collect())
    }
    fn dtype(&self, name: &str) -> Result<String> { Ok(self.view(name)?.dtype.name().to_string()) }
    fn buffer(&self, name: &str, _device: &crate::cuda::Device) -> Result<crate::cuda::Buffer> {
        let v = self.view(name)?;
        crate::cuda::Buffer::view(v.ptr, v.bytes)
    }
    fn buffers(&self, names: &[String], device: &crate::cuda::Device) -> Result<Vec<crate::cuda::Buffer>> {
        names.iter().map(|n| self.buffer(n, device)).collect()
    }
}
impl Source for crate::gold::Gold {
    fn bytes(&self, name: &str) -> Result<Vec<u8>> { crate::gold::Gold::bytes(self, name) }
}

/// One safetensors file: its header, read lazily by tensor name.
pub struct SafeTensors { path: PathBuf, base: u64, pub entries: HashMap<String, Entry> }
impl SafeTensors {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let mut file = File::open(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        let mut len = [0u8; 8];
        file.read_exact(&mut len)?;
        let len = u64::from_le_bytes(len);
        let mut header = vec![0u8; usize::try_from(len)?];
        file.read_exact(&mut header)?;
        let json: serde_json::Value = serde_json::from_slice(&header)?;
        let mut entries = HashMap::new();
        for (name, v) in json.as_object().ok_or("safetensors header is not an object")? {
            if name == "__metadata__" { continue; }
            let offsets = v["data_offsets"].as_array().ok_or("data_offsets")?;
            entries.insert(name.clone(), Entry {
                dtype: v["dtype"].as_str().ok_or("dtype")?.to_string(),
                shape: v["shape"].as_array().ok_or("shape")?.iter().map(|d| d.as_u64().unwrap_or(0) as usize).collect(),
                start: offsets[0].as_u64().ok_or("offset")?, end: offsets[1].as_u64().ok_or("offset")?,
            });
        }
        Ok(Self { path, base: 8 + len, entries })
    }
    pub fn entry(&self, name: &str) -> Result<&Entry> {
        self.entries.get(name).ok_or_else(|| format!("{}: no tensor {name}", self.path.display()).into())
    }
    pub fn bytes(&self, name: &str) -> Result<Vec<u8>> {
        let e = self.entry(name)?;
        let mut file = File::open(&self.path)?;
        file.seek(SeekFrom::Start(self.base + e.start))?;
        let mut out = vec![0u8; usize::try_from(e.end - e.start)?];
        file.read_exact(&mut out)?;
        Ok(out)
    }
    /// The tensor on the device, checking its dtype.
    #[cfg(feature = "cuda")]
    pub fn upload(&self, name: &str, dtype: &str, device: &crate::cuda::Device) -> Result<crate::cuda::Buffer> {
        let e = self.entry(name)?;
        if e.dtype != dtype { return Err(format!("{name}: {} (expected {dtype})", e.dtype).into()); }
        crate::cuda::Buffer::upload(&self.bytes(name)?, device)
    }
}
