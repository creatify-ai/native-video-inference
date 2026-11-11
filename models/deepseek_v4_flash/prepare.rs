//! Offline preparation of the serving weights (`dsv4 prepare`), for core_infra's streaming loader
//! (docs/fast-cold-start.md). Every parameter SGLang loads is rebuilt from the published
//! checkpoint (weights.rs), checked against the reference's params dump (SHA-256), put in the
//! memory layout the kernels read (weights::memory_layout) and written into 64 MiB safetensors
//! shards. Serving then only reads and uploads; no CPU transform runs at startup.
//!
//! The loader hands each tensor to one reader and places tensors 256-byte aligned in request
//! order. A tensor larger than one shard is therefore stored as consecutive 64 MiB pieces
//! `name#000`, `name#001`, ... (64 MiB is a multiple of 256): the pieces are read in parallel and
//! land contiguously, which the serving side checks before it uses the first piece's address.
//! The RoPE tables SGLang computes at load (exported from the reference process) are stored as
//! `tables.freqs_cis.swa` and `tables.freqs_cis.compressed`.
use crate::safetensors::Checkpoint;
use crate::weights::{memory_layout, Source};
use crate::Result;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// Shard size and the piece size of larger tensors.
pub const PIECE: usize = 64 << 20;
/// Tensor and header alignment inside a shard (direct reads).
const ALIGN: usize = 4096;
/// Files copied from the checkpoint next to the shards (text side and configuration).
pub const COPIED: &[&str] = &["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json"];
pub const TABLE_SWA: &str = "tables.freqs_cis.swa";
pub const TABLE_COMPRESSED: &str = "tables.freqs_cis.compressed";

pub fn piece_name(name: &str, k: usize) -> String { format!("{name}#{k:03}") }

/// Packs tensors into safetensors shards of at most PIECE payload bytes (one larger tensor = one file).
struct Shards { dir: PathBuf, next: usize, pending: Vec<(String, Vec<u8>)>, bytes: usize, files: Vec<Value> }

impl Shards {
    fn push(&mut self, name: String, data: Vec<u8>) -> Result<()> {
        let aligned = data.len().div_ceil(ALIGN) * ALIGN;
        if !self.pending.is_empty() && self.bytes + aligned > PIECE { self.flush()?; }
        self.bytes += aligned;
        self.pending.push((name, data));
        Ok(())
    }

    fn flush(&mut self) -> Result<()> {
        if self.pending.is_empty() { return Ok(()); }
        let file = format!("shard-{:05}.safetensors", self.next);
        self.next += 1;
        // Header: U8 [bytes] per tensor (the kernels read raw bytes in SGLang's memory layout);
        // payload offsets 4 KiB aligned; header padded so the payload starts 4 KiB aligned.
        let mut header = serde_json::Map::new();
        let mut offset = 0usize;
        let mut spans = Vec::new();
        for (name, data) in &self.pending {
            header.insert(name.clone(), json!({"dtype": "U8", "shape": [data.len()], "data_offsets": [offset, offset + data.len()]}));
            spans.push(offset);
            offset = (offset + data.len()).div_ceil(ALIGN) * ALIGN;
        }
        let mut text = serde_json::to_vec(&Value::Object(header)).map_err(|e| e.to_string())?;
        while (8 + text.len()) % ALIGN != 0 { text.push(b' '); }
        let mut out = Vec::with_capacity(8 + text.len() + offset);
        out.extend((text.len() as u64).to_le_bytes());
        out.extend(&text);
        let base = out.len();
        for ((_, data), span) in self.pending.iter().zip(&spans) {
            out.resize(base + span, 0);
            out.extend(data);
        }
        let path = self.dir.join(&file);
        std::fs::write(&path, &out).map_err(|e| format!("{}: {e}", path.display()))?;
        self.files.push(json!({"file": file, "bytes": out.len(), "tensors": self.pending.len()}));
        self.pending.clear();
        self.bytes = 0;
        Ok(())
    }
}

fn sha(data: &[u8]) -> String { format!("{:x}", Sha256::digest(data)) }

/// Write the prepared package to `out`: `weights/shard-*.safetensors`, the copied text/config
/// files and `prepare.json` (per parameter: reference SHA check, pieces, device-layout SHA).
pub fn prepare(ckpt_dir: &Path, params_json: &Path, tables_dir: &Path, out: &Path, workers: usize) -> Result<Value> {
    let ckpt = Checkpoint::open(ckpt_dir)?;
    let params: Vec<Value> = serde_json::from_slice(&std::fs::read(params_json).map_err(|e| format!("{}: {e}", params_json.display()))?)
        .map_err(|e| e.to_string())?;
    let weights = out.join("weights");
    std::fs::create_dir_all(&weights).map_err(|e| format!("{}: {e}", weights.display()))?;
    for f in COPIED {
        std::fs::copy(ckpt_dir.join(f), out.join(f)).map_err(|e| format!("copy {f}: {e}"))?;
    }
    // The reference's own parameters; complex buffers (RoPE) come from the exported tables.
    let rows: Vec<&Value> = params.iter().filter(|r| r["kind"] == "param" && r["dtype"] != "complex64").collect();
    let source = Source { ckpt: &ckpt };
    let mut shards = Shards { dir: weights, next: 0, pending: vec![], bytes: 0, files: vec![] };
    let mut report = Vec::new();
    let (mut equal, mut not_ported, mut stored_bytes) = (0usize, Vec::new(), 0usize);
    let started = std::time::Instant::now();
    let mut progress = std::fs::File::create(out.join("prepare.progress")).map_err(|e| e.to_string())?;
    // Batches of `workers` parameters rebuilt in parallel, then written in parameter order.
    for (b, batch) in rows.chunks(workers).enumerate() {
        let results: Vec<Result<Option<Vec<u8>>>> = std::thread::scope(|scope| {
            let handles: Vec<_> = batch.iter().map(|row| {
                let source = &source;
                scope.spawn(move || source.prepare(row["name"].as_str().unwrap_or("")))
            }).collect();
            handles.into_iter().map(|h| h.join().unwrap_or_else(|_| Err("rebuild panicked".into()))).collect()
        });
        for (row, r) in batch.iter().zip(results) {
            let name = row["name"].as_str().unwrap_or("").to_string();
            let Some(logical) = r.map_err(|e| format!("{name}: {e}"))? else { not_ported.push(name); continue; };
            let want = row["sha256"].as_str().ok_or_else(|| format!("{name}: the params dump has no SHA-256 (capture it with DSV4_GOLD_HASH_PARAMS)"))?;
            if sha(&logical) != want || logical.len() as u64 != row["bytes"].as_u64().unwrap_or(0) {
                return Err(format!("{name}: rebuilt bytes ({}) differ from the reference's parameter ({} bytes)", logical.len(), row["bytes"]));
            }
            equal += 1;
            let device = memory_layout(&name, logical);
            let pieces = device.len().div_ceil(PIECE);
            stored_bytes += device.len();
            report.push(json!({"name": name, "bytes": device.len(), "pieces": if pieces > 1 { pieces } else { 0 },
                               "reference_sha256": want, "device_sha256": sha(&device)}));
            if pieces > 1 {
                for (k, chunk) in device.chunks(PIECE).enumerate() { shards.push(piece_name(&name, k), chunk.to_vec())?; }
            } else {
                shards.push(name, device)?;
            }
        }
        use std::io::Write;
        let _ = writeln!(progress, "{}", json!({"batch": b, "done": ((b + 1) * workers).min(rows.len()), "of": rows.len(),
                                                "gb": stored_bytes as f64 / 1e9, "s": started.elapsed().as_secs()}));
    }
    // The RoPE tables, as the reference process computed them.
    let tables = crate::gold::Gold::open(tables_dir);
    for (name, gold) in [(TABLE_SWA, "model.layers.0.self_attn.freqs_cis"), (TABLE_COMPRESSED, "model.layers.2.self_attn.freqs_cis")] {
        let data = tables.bytes(gold)?;
        report.push(json!({"name": name, "bytes": data.len(), "pieces": 0, "source": gold, "device_sha256": sha(&data)}));
        stored_bytes += data.len();
        shards.push(name.to_string(), data)?;
    }
    shards.flush()?;
    let summary = json!({"schema": "dsv4-prepared-v1", "checkpoint": ckpt_dir, "params": params_json, "tables": tables_dir,
                         "reference_equal": equal, "not_ported": not_ported, "stored_bytes": stored_bytes,
                         "shards": shards.files.len(), "piece_bytes": PIECE, "seconds": started.elapsed().as_secs_f64()});
    let doc = json!({"summary": summary, "tensors": report, "files": shards.files});
    std::fs::write(out.join("prepare.json"), serde_json::to_vec_pretty(&doc).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    Ok(summary)
}
