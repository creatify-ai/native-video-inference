//! Serving weights from a prepared package (prepare.rs) through core_infra's streaming loader
//! (core_infra::model_load, docs/fast-cold-start.md): a SHA-pinned `native-checkpoint-v1`
//! manifest, bounded parallel readers started before the CUDA context exists, direct uploads
//! into device allocations, light guards (NATIVE_LOAD_* overrides apply).
//!
//! DSV4_LOAD_POOLS (default 1) splits the parameters into groups of about equal bytes, each a
//! separate loader with its own readers (the framework caps one pool at 64) and upload stream. A
//! parameter's pieces always stay in one group. Measured on one warm B300 host, two pools read no
//! faster (17.9 vs 17.4 / 20.8 GB/s) and start their readers later (1.7 vs 0.8 s).
use crate::prepare::PIECE;
use crate::Result;
use native_core_infra::checkpoint::{Manifest, Source};
use native_core_infra::model_load::{self, Loaded, Options, Pending, Request, Stats};
use std::collections::{BTreeMap, HashMap};
use std::path::Path;

/// Readers running; no CUDA call made yet.
pub struct Load { pools: Vec<Pending>, manifest: Manifest, pub requests: usize, pub bytes: usize }

/// The loaded package: its device allocations and each parameter's address.
pub struct Weights { pub loaded: Vec<Loaded>, ptrs: HashMap<String, u64>, pub stats: Stats }

fn base_of(name: &str) -> (&str, usize) {
    match name.rsplit_once('#') { Some((base, k)) => (base, k.parse().unwrap_or(0)), None => (name, 0) }
}

/// Start reading package `dir` (manifest.json + weights/), pinned by the manifest's SHA-256.
pub fn start(dir: &Path, manifest_sha256: &str) -> Result<Load> {
    let dir = std::fs::canonicalize(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let source = Source { manifest: dir.join("manifest.json"), sha256: manifest_sha256.to_string(),
                          roots: HashMap::from([("checkpoint".to_string(), dir.join("weights"))]) };
    let declaration = source.load()?;
    let tensors = &declaration.manifest.tensors;
    // Parameters (pieces grouped, in order), largest first, each to the group with the fewest bytes.
    let mut params: BTreeMap<&str, Vec<(usize, &String)>> = BTreeMap::new();
    for name in tensors.keys() { let (base, k) = base_of(name); params.entry(base).or_default().push((k, name)); }
    let mut params: Vec<(usize, Vec<&String>)> = params.into_values().map(|mut v| {
        v.sort();
        (v.iter().map(|(_, n)| tensors[*n].bytes).sum(), v.into_iter().map(|(_, n)| n).collect())
    }).collect();
    params.sort_by(|a, b| b.0.cmp(&a.0));
    let n_pools: usize = std::env::var("DSV4_LOAD_POOLS").ok().and_then(|v| v.parse().ok()).unwrap_or(1).clamp(1, 4);
    let mut groups: Vec<(usize, Vec<Request>)> = (0..n_pools).map(|_| (0, Vec::new())).collect();
    for (bytes, names) in params {
        let g = groups.iter_mut().min_by_key(|g| g.0).unwrap();
        g.0 += bytes;
        g.1.extend(names.into_iter().map(|n| Request { name: n.clone(), dtype: tensors[n].dtype }));
    }
    let options = Options::for_declaration_with_readers(&declaration, 64)?;
    let requests = groups.iter().map(|g| g.1.len()).sum();
    let bytes = groups.iter().map(|g| g.0).sum();
    let pools = groups.iter().filter(|g| !g.1.is_empty())
        .map(|g| model_load::start(&declaration.manifest, &declaration.roots, &g.1, options)).collect::<Result<Vec<_>>>()?;
    Ok(Load { pools, manifest: declaration.manifest, requests, bytes })
}

impl Load {
    /// Allocate and fill: the first pool on `stream`, the others on their own streams and threads
    /// (all in the current context). Returns when every byte is on the device.
    pub fn finish(self, stream: crate::cuda::Stream, ctx: crate::cuda::CUcontext) -> Result<Weights> {
        let mut pools = self.pools.into_iter();
        let first = pools.next().ok_or("empty package")?;
        let ctx = ctx as usize;
        let loaded: Vec<Loaded> = std::thread::scope(|scope| -> Result<Vec<Loaded>> {
            let others: Vec<_> = pools.map(|pending| scope.spawn(move || -> Result<Loaded> {
                crate::cuda::set_current(ctx as crate::cuda::CUcontext)?;
                let s = crate::cuda::Stream::new()?;
                unsafe { pending.finish(s.0 as *mut std::ffi::c_void) }
            })).collect();
            let mut out = vec![unsafe { first.finish(stream.0 as *mut std::ffi::c_void) }?];
            for h in others { out.push(h.join().map_err(|_| "loader pool panicked")??); }
            Ok(out)
        })?;
        // Reassemble split parameters: their pieces must be one contiguous range.
        let view = |name: &str| loaded.iter().find_map(|l| l.tensors.get(name)).ok_or_else(|| format!("tensor {name} was not loaded"));
        let mut pieces: BTreeMap<String, Vec<(usize, u64, usize)>> = BTreeMap::new();
        let mut ptrs = HashMap::new();
        for name in self.manifest.tensors.keys() {
            let v = view(name)?;
            match name.rsplit_once('#') {
                Some((base, k)) => pieces.entry(base.to_string()).or_default()
                    .push((k.parse().map_err(|_| format!("bad piece name {name}"))?, v.ptr, v.bytes)),
                None => { ptrs.insert(name.clone(), v.ptr); }
            }
        }
        for (base, mut parts) in pieces {
            parts.sort();
            let start = parts[0].1;
            for (i, &(k, ptr, bytes)) in parts.iter().enumerate() {
                let last = i + 1 == parts.len();
                if k != i || ptr != start + (i * PIECE) as u64 || (!last && bytes != PIECE) {
                    return Err(format!("{base}: piece {k} is not contiguous with the previous pieces"));
                }
            }
            ptrs.insert(base, start);
        }
        let stats = Stats { tensors: loaded.iter().map(|l| l.stats.tensors).sum(), read_bytes: loaded.iter().map(|l| l.stats.read_bytes).sum(),
                            device_bytes: loaded.iter().map(|l| l.stats.device_bytes).sum(), converted: loaded.iter().map(|l| l.stats.converted).sum(),
                            seconds: loaded.iter().map(|l| l.stats.seconds).fold(0.0, f64::max) };
        Ok(Weights { loaded, ptrs, stats })
    }
}

impl Weights {
    pub fn get(&self, name: &str) -> Result<u64> {
        self.ptrs.get(name).copied().ok_or_else(|| format!("parameter {name} is not in the prepared package"))
    }
}
