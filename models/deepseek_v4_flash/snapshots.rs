//! Per-launch buffer snapshots of the reference (snapshot mode of reference/tape/cupti_tape.cpp):
//! the bytes behind every device pointer a launch was passed, before ("in") and after ("out") it
//! ran. Rungs compare native intermediates with them to find the first launch that diverges.
use crate::Result;
use serde_json::Value;
use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

#[derive(Clone, Debug)]
struct Entry { param: u64, phase: String, ptr: u64, bytes: usize, offset: u64 }

#[derive(Clone, Debug)]
pub struct LaunchInfo { pub seq: u64, pub label: String, pub name: String, snap: i64 }

pub struct Snapshots { pack: PathBuf, launches: Vec<LaunchInfo>, entries: HashMap<i64, Vec<Entry>> }

impl Snapshots {
    pub fn open(tape: &Path) -> Result<Self> {
        let read = |f: &str| std::fs::read_to_string(tape.join(f)).map_err(|e| format!("snapshots {f}: {e}"));
        let mut entries: HashMap<i64, Vec<Entry>> = HashMap::new();
        for line in read("snapshots.jsonl")?.lines() {
            let v: Value = serde_json::from_str(line).map_err(|e| e.to_string())?;
            entries.entry(v["snap"].as_i64().unwrap_or(-1)).or_default().push(Entry {
                param: v["param"].as_u64().unwrap_or(0), phase: v["phase"].as_str().unwrap_or("").into(),
                ptr: v["ptr"].as_u64().unwrap_or(0), bytes: v["bytes"].as_u64().unwrap_or(0) as usize,
                offset: v["offset"].as_u64().unwrap_or(0) });
        }
        let mut launches = Vec::new();
        for line in read("launches.jsonl")?.lines() {
            let v: Value = serde_json::from_str(line).map_err(|e| e.to_string())?;
            let Some(name) = v["name"].as_str() else { continue };
            let snap = v["snap"].as_i64().unwrap_or(-1);
            if snap < 0 { continue; }
            launches.push(LaunchInfo { seq: v["seq"].as_u64().unwrap_or(0), label: v["label"].as_str().unwrap_or("").into(),
                                       name: name.into(), snap });
        }
        Ok(Self { pack: tape.join("snapshots.bin"), launches, entries })
    }

    /// The `nth` (0-based) snapshotted launch whose label is `label` and whose name contains `fragment`.
    pub fn find(&self, label: &str, fragment: &str, nth: usize) -> Result<LaunchInfo> {
        self.launches.iter().filter(|l| l.label == label && l.name.contains(fragment)).nth(nth).cloned()
            .ok_or_else(|| format!("no snapshot of launch #{nth} '{fragment}' under '{label}'"))
    }

    /// Bytes behind parameter word (`param` * 1024 + 8-byte word index) of a launch, in or out.
    pub fn read(&self, launch: &LaunchInfo, param: usize, word: usize, phase: &str) -> Result<Vec<u8>> {
        let key = (param * 1024 + word) as u64;
        let e = self.entries.get(&launch.snap).and_then(|v| v.iter().find(|e| e.param == key && e.phase == phase))
            .ok_or_else(|| format!("seq {} has no {phase} snapshot of param {param} word {word}", launch.seq))?;
        let mut f = std::fs::File::open(&self.pack).map_err(|e| e.to_string())?;
        f.seek(SeekFrom::Start(e.offset)).map_err(|e| e.to_string())?;
        let mut data = vec![0u8; e.bytes];
        f.read_exact(&mut data).map_err(|e| e.to_string())?;
        let _ = e.ptr;
        Ok(data)
    }
}

/// Compare a native buffer with the leading bytes of a snapshot; prints one JSON line.
pub fn check_prefix(what: &str, native: &[u8], snap: &[u8]) -> bool {
    let n = native.len().min(snap.len());
    let first = native[..n].iter().zip(&snap[..n]).position(|(a, b)| a != b);
    let ok = first.is_none() && n == native.len();
    let mut diffs = Vec::new();
    if let Some(f) = first {
        // The first few differing 4-byte words, as fp32 and raw bits (native, reference).
        let mut w = f / 4;
        while diffs.len() < 6 && (w + 1) * 4 <= n {
            let a = u32::from_le_bytes(native[w * 4..w * 4 + 4].try_into().unwrap());
            let b = u32::from_le_bytes(snap[w * 4..w * 4 + 4].try_into().unwrap());
            if a != b { diffs.push(serde_json::json!([w, f32::from_bits(a), f32::from_bits(b), format!("{a:08x}"), format!("{b:08x}")])); }
            w += 1;
        }
    }
    println!("{}", serde_json::json!({"snapshot": what, "equal": ok, "bytes": native.len(), "compared": n, "first_diff_byte": first, "words": diffs}));
    ok
}
