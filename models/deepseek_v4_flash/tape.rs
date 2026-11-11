//! The CUPTI launch tape (reference/tape/cupti_tape.cpp): recorded launches, memory operations and
//! module images of a reference forward. Used to replay a recorded forward and to check that a
//! native launch equals the reference's (same binary, configuration and parameter bytes).
use crate::Result;
use serde::Deserialize;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, Deserialize)]
pub struct Param { pub offset: usize, pub size: usize, pub hex: String, #[serde(default)] pub packed: bool }

#[derive(Clone, Debug, Deserialize)]
pub struct Attr { pub id: u32, pub hex: String }

/// One line of launches.jsonl: a kernel launch, a memcpy/memset or a graph launch marker.
#[derive(Clone, Debug, Deserialize)]
pub struct Event {
    pub seq: u64,
    pub label: String,
    pub api: String,
    #[serde(default)] pub name: Option<String>,
    #[serde(default)] pub module_ids: Vec<u32>,
    #[serde(default)] pub grid: Option<[u32; 3]>,
    #[serde(default)] pub block: Option<[u32; 3]>,
    #[serde(default)] pub smem: Option<u32>,
    #[serde(default)] pub static_smem: Option<i32>,
    #[serde(default)] pub regs: Option<i32>,
    #[serde(default)] pub stream: Option<u64>,
    #[serde(default)] pub capturing: Option<i32>,
    #[serde(default)] pub attrs: Vec<Attr>,
    #[serde(default)] pub params: Vec<Param>,
    #[serde(default)] pub op: Option<String>,
    #[serde(default)] pub kind: Option<i32>,
    #[serde(default)] pub dst: Option<u64>,
    #[serde(default)] pub src: Option<u64>,
    #[serde(default)] pub bytes: Option<usize>,
    #[serde(default)] pub data: Option<String>,
    #[serde(default)] pub value: Option<u32>,
    #[serde(default)] pub count: Option<usize>,
    #[serde(default)] pub width: Option<i32>,
    #[serde(default)] pub unresolved: bool,
}

impl Event {
    pub fn is_launch(&self) -> bool { self.name.is_some() && self.op.is_none() && !self.unresolved }
}

pub fn unhex(s: &str) -> Vec<u8> {
    (0..s.len() / 2).map(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).unwrap_or(0)).collect()
}

pub struct Tape { pub root: PathBuf, pub events: Vec<Event> }

impl Tape {
    pub fn open(root: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(root.join("launches.jsonl")).map_err(|e| format!("tape: {e}"))?;
        let mut events = Vec::new();
        for (i, line) in text.lines().enumerate() {
            if line.trim().is_empty() { continue; }
            events.push(serde_json::from_str(line).map_err(|e| format!("tape line {}: {e}", i + 1))?);
        }
        Ok(Self { root: root.to_path_buf(), events })
    }

    pub fn label(&self, label: &str) -> Vec<&Event> { self.events.iter().filter(|e| e.label == label).collect() }

    pub fn module_image(&self, id: u32) -> Result<Vec<u8>> {
        std::fs::read(self.root.join("modules").join(format!("{id}.cubin"))).map_err(|e| format!("module {id}: {e}"))
    }

    /// Distinct kernels of a label in first-launch order, with launch counts.
    pub fn kernels(&self, label: &str) -> Vec<(String, usize)> {
        let mut order: Vec<String> = Vec::new();
        let mut counts: BTreeMap<String, usize> = BTreeMap::new();
        for e in self.label(label) {
            if let Some(n) = e.name.as_ref().filter(|_| e.is_launch()) {
                if !counts.contains_key(n) { order.push(n.clone()); }
                *counts.entry(n.clone()).or_default() += 1;
            }
        }
        order.into_iter().map(|n| { let c = counts[&n]; (n, c) }).collect()
    }
}
