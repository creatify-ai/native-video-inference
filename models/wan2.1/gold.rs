//! Reader for `wan-gold-v1` tensor dumps and the bitwise comparison every parity check reports.
//!
//! A gold is a reference run recorded at the boundaries the port reproduces
//! (docs/porting-a-model.md, "Record golds"): `manifest.json` with `"schema": "wan-gold-v1"`,
//! `meta` (call arguments) and `tensors`, each `{file, dtype, shape, bytes, sha256}` where `file`
//! is a raw little-endian dump relative to the gold root, or null for hash-only entries. The
//! recorder for the official PyTorch pipeline is not part of this repository.
use crate::Result;
use serde::Deserialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::path::{Component, Path, PathBuf};

/// The original little-endian FP32 output digest, borrowing completed frames.
pub fn sha256_f32(values:&[f32])->String {
    #[cfg(target_endian="little")]
    {
        // f32 has no padding and every bit pattern is valid. The immutable
        // byte view lives no longer than the initialized owning float slice.
        let bytes=unsafe{std::slice::from_raw_parts(values.as_ptr().cast::<u8>(),std::mem::size_of_val(values))};
        sha256_hex(bytes)
    }
    #[cfg(target_endian="big")]
    {sha256_hex(&values.iter().flat_map(|value|value.to_le_bytes()).collect::<Vec<_>>())}
}

#[cfg(test)] mod host_frame_hash_tests {
    #[test] fn checksum_preserves_zero_sign_nan_payload_and_float_bits(){
        let bits=[0u32,0x80000000,0x3f800000,0x7fc12345,1,0xff800000];
        let values=bits.map(f32::from_bits);
        let bytes=[0,0,0,0, 0,0,0,128, 0,0,128,63, 69,35,193,127, 1,0,0,0, 0,0,128,255];
        assert_eq!(super::sha256_f32(&values),super::sha256_hex(&bytes));
    }
}

#[derive(Clone, Debug, Deserialize)]
pub struct Entry {
    pub file: Option<String>,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub bytes: usize,
    pub sha256: String,
    #[serde(default)]
    pub complex_view_as_real: bool,
}

pub struct Gold {
    pub root: PathBuf,
    pub tensors: BTreeMap<String, Entry>,
    pub meta: Value,
}

impl Gold {
    pub fn open(root: impl AsRef<Path>) -> Result<Self> {
        let root = root.as_ref().to_path_buf();
        let manifest: Value = serde_json::from_slice(&std::fs::read(root.join("manifest.json"))?)?;
        if manifest["schema"] != "wan-gold-v1" { return Err("not a wan-gold-v1 manifest".into()); }
        let tensors = serde_json::from_value(manifest["tensors"].clone())?;
        Ok(Self { root, tensors, meta: manifest["meta"].clone() })
    }
    pub fn entry(&self, name: &str) -> Result<&Entry> {
        self.tensors.get(name).ok_or_else(|| format!("gold has no tensor {name}").into())
    }
    /// Stored bytes, verified against the recorded SHA-256.
    pub fn bytes(&self, name: &str) -> Result<Vec<u8>> {
        let entry = self.entry(name)?;
        let file = entry.file.as_ref().ok_or_else(|| format!("{name} was recorded hash-only"))?;
        if Path::new(file).components().any(|c| !matches!(c, Component::Normal(_))) {
            return Err(format!("{name}: path escapes the gold root").into());
        }
        let data = std::fs::read(self.root.join(file))?;
        if data.len() != entry.bytes || sha256_hex(&data) != entry.sha256 {
            return Err(format!("{name}: stored bytes do not match the manifest").into());
        }
        Ok(data)
    }
    /// Reads and verifies many stored tensors concurrently (network volumes and SHA-256
    /// are both throughput-bound per thread). Returns bytes in the order of `names`.
    pub fn bytes_many(&self, names: &[String], threads: usize) -> Result<Vec<Vec<u8>>> {
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
    pub fn names<'a>(&'a self, prefix: &'a str) -> impl Iterator<Item = &'a String> + 'a {
        self.tensors.keys().filter(move |k| k.starts_with(prefix))
    }
}

pub fn sha256_hex(data: &[u8]) -> String { format!("{:x}", Sha256::digest(data)) }

pub fn width(dtype: &str) -> Result<usize> {
    Ok(match dtype {
        "torch.bfloat16" | "torch.float16" | "torch.int16" => 2,
        "torch.float32" | "torch.int32" => 4,
        "torch.float64" | "torch.int64" => 8,
        "torch.uint8" | "torch.bool" | "torch.int8" => 1,
        _ => return Err(format!("unsupported gold dtype {dtype}").into()),
    })
}

#[derive(Debug, Default, serde::Serialize)]
pub struct Diff {
    pub elements: usize,
    pub mismatched: usize,
    pub first_mismatch: Option<usize>,
    pub max_ulp: u64,
    pub max_abs: f64,
    pub sha256_equal: bool,
}

fn decode(bits: u64, dtype: &str) -> f64 {
    match dtype {
        "torch.bfloat16" => f32::from_bits((bits as u32) << 16) as f64,
        "torch.float16" => half_to_f32(bits as u16) as f64,
        "torch.float32" => f32::from_bits(bits as u32) as f64,
        "torch.float64" => f64::from_bits(bits),
        _ => bits as f64,
    }
}

fn half_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) as u32) << 31;
    let exp = ((h >> 10) & 0x1f) as u32;
    let frac = (h & 0x3ff) as u32;
    let bits = match (exp, frac) {
        (0, 0) => sign,
        (0, f) => { let mut e = 127 - 15 + 1; let mut f = f; while f & 0x400 == 0 { f <<= 1; e -= 1; } sign | (e << 23) | ((f & 0x3ff) << 13) },
        (0x1f, f) => sign | 0x7f80_0000 | (f << 13),
        (e, f) => sign | ((e + 127 - 15) << 23) | (f << 13),
    };
    f32::from_bits(bits)
}

/// Monotone integer ordering of IEEE bit patterns, so |a-b| counts ULPs.
fn ordered(bits: u64, width: usize) -> i128 {
    let sign = 1u64 << (width * 8 - 1);
    if bits & sign != 0 { -((bits & !sign) as i128) } else { bits as i128 }
}

pub fn compare(got: &[u8], want: &[u8], dtype: &str) -> Result<Diff> {
    let w = width(dtype)?;
    if got.len() != want.len() || got.len() % w != 0 { return Err("compared buffers differ in size".into()); }
    let mut diff = Diff { elements: got.len() / w, sha256_equal: got == want, ..Default::default() };
    if diff.sha256_equal { return Ok(diff); }
    let floating = dtype.contains("float");
    for (i, (a, b)) in got.chunks_exact(w).zip(want.chunks_exact(w)).enumerate() {
        if a == b { continue; }
        diff.mismatched += 1;
        diff.first_mismatch.get_or_insert(i);
        let (mut x, mut y) = ([0u8; 8], [0u8; 8]);
        x[..w].copy_from_slice(a);
        y[..w].copy_from_slice(b);
        let (x, y) = (u64::from_le_bytes(x), u64::from_le_bytes(y));
        if floating {
            diff.max_ulp = diff.max_ulp.max((ordered(x, w) - ordered(y, w)).unsigned_abs() as u64);
            diff.max_abs = diff.max_abs.max((decode(x, dtype) - decode(y, dtype)).abs());
        }
    }
    Ok(diff)
}
