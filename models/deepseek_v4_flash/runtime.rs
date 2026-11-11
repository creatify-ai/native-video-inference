//! Native runtime: the reference's kernel binaries (a bundle exported from the launch tape), device
//! arenas and launch plumbing. Every kernel the port runs is one of the reference's own CUBINs; the
//! port's job is to build each launch exactly as the reference's host code does.
use crate::cuda::{self, Buffer, Function, LaunchAttribute, Module, Stream};
use crate::Result;
use serde_json::Value;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

/// The kernel bundle (reference/tape/bundle.py): manifest.json + cubin/<sha>.cubin.
pub struct Kernels {
    /// One or more bundles (`a:b:...`, merged: kernels, variants and MoE tactics; the first bundle
    /// wins a conflict); each CUBIN is read from the bundle that holds it.
    roots: Vec<PathBuf>,
    cubin_root: HashMap<String, usize>,
    manifest: Value,
    modules: Mutex<HashMap<String, std::sync::Arc<Module>>>,
    functions: Mutex<HashMap<String, Function>>,
}

impl Kernels {
    pub fn open(root: &Path) -> Result<Self> {
        let roots: Vec<PathBuf> = root.to_string_lossy().split(':').filter(|p| !p.is_empty()).map(PathBuf::from).collect();
        let mut manifest = Value::Null;
        let mut cubin_root = HashMap::new();
        for (i, r) in roots.iter().enumerate() {
            let m: Value = serde_json::from_slice(&std::fs::read(r.join("manifest.json"))
                .map_err(|e| format!("kernel bundle {}: {e}", r.display()))?).map_err(|e| format!("kernel bundle: {e}"))?;
            for entry in m["kernels"].as_object().into_iter().flat_map(|k| k.values()) {
                if let Some(v) = entry["variants"].as_object() { for sha in v.keys() { cubin_root.entry(sha.clone()).or_insert(i); } }
                if let Some(c) = entry["cubin"].as_str() { cubin_root.entry(c.to_string()).or_insert(i); }
            }
            if manifest.is_null() { manifest = m; continue; }
            for (name, entry) in m["kernels"].as_object().cloned().unwrap_or_default() {
                let mine = &mut manifest["kernels"][&name];
                if mine.is_null() { *mine = entry; continue; }
                if let (Some(dst), Some(src)) = (mine["variants"].as_object_mut(), entry["variants"].as_object()) {
                    for (sha, v) in src {
                        match dst.get_mut(sha) {
                            None => { dst.insert(sha.clone(), v.clone()); }
                            // The same image recorded by several captures: every tag set it was
                            // launched with (a capture without layer hooks labels it `meta`).
                            Some(have) => {
                                let mut sets: Vec<Value> = have["other_tags"].as_array().cloned().unwrap_or_default();
                                for t in std::iter::once(&v["tags"]).chain(v["other_tags"].as_array().into_iter().flatten()) {
                                    if *t != have["tags"] && !sets.contains(t) { sets.push(t.clone()); }
                                }
                                have["other_tags"] = Value::Array(sets);
                            }
                        }
                    }
                }
            }
            for table in ["moe_tactics", "moe_tactics_mx"] {
                for (t, v) in m[table].as_object().cloned().unwrap_or_default() {
                    if manifest[table][&t].is_null() { manifest[table][&t] = v; }
                }
            }
        }
        Ok(Self { roots, cubin_root, manifest, modules: Mutex::new(HashMap::new()), functions: Mutex::new(HashMap::new()) })
    }

    fn cubin_path(&self, sha: &str) -> PathBuf {
        self.roots[*self.cubin_root.get(sha).unwrap_or(&0)].join("cubin").join(format!("{sha}.cubin"))
    }

    /// The MoE tactic (token tile, FC1 kernel, FC2 kernel) the census run used for `t` tokens.
    pub fn moe_tactic(&self, t: usize) -> Option<(i32, String, String)> { self.tactic_of("moe_tactics", t) }
    /// The same for the MXFP4 experts (the MTP draft's layer).
    pub fn moe_tactic_mx(&self, t: usize) -> Option<(i32, String, String)> { self.tactic_of("moe_tactics_mx", t) }
    fn tactic_of(&self, table: &str, t: usize) -> Option<(i32, String, String)> {
        let entry = |e: &Value| Some((e["tile"].as_i64()? as i32, e["fc1"].as_str()?.to_string(), e["fc2"].as_str()?.to_string()));
        if let Some(x) = entry(&self.manifest[table][t.to_string()]) { return Some(x); }
        // MXFP4 (the MTP draft's experts) at a token count no capture recorded: the tactic of the
        // nearest recorded counts below and above, when they agree. Its recorded choices are a
        // step function of the count (one kernel pair for 1..612 tokens, the next for 688..1024);
        // the FP4 table's are not (several pairs interleave within one tile size), so FP4 keeps
        // `moe::tactic`'s table.
        if table != "moe_tactics_mx" { return None; }
        let m = self.manifest[table].as_object()?;
        let counts = || m.keys().filter_map(|k| k.parse::<usize>().ok());
        let (lo, hi) = (counts().filter(|&c| c < t).max()?, counts().filter(|&c| c > t).min()?);
        let (a, b) = (entry(&m[&lo.to_string()])?, entry(&m[&hi.to_string()])?);
        (a == b).then_some(a)
    }

    /// Every CUBIN the bundle names, read and checked against its SHA-256 (CPU only).
    pub fn read_all(&self) -> Result<Vec<(String, Vec<u8>)>> {
        let mut shas = std::collections::BTreeSet::new();
        for entry in self.manifest["kernels"].as_object().into_iter().flat_map(|m| m.values()) {
            if let Some(v) = entry["variants"].as_object() { shas.extend(v.keys().cloned()); }
            if let Some(c) = entry["cubin"].as_str() { shas.insert(c.to_string()); }
        }
        shas.into_iter().map(|sha| {
            let image = std::fs::read(self.cubin_path(&sha)).map_err(|e| format!("cubin {sha}: {e}"))?;
            if crate::gold::sha(&image) != sha { return Err(format!("cubin {sha} differs from its name")); }
            Ok((sha, image))
        }).collect()
    }

    /// Load images from `read_all` as modules of the current context (instead of on first launch).
    pub fn load_all(&self, images: Vec<(String, Vec<u8>)>) -> Result<usize> {
        let n = images.len();
        for (sha, image) in images {
            let m = std::sync::Arc::new(Module::load(&image)?);
            self.modules.lock().unwrap().entry(sha).or_insert(m);
        }
        Ok(n)
    }

    /// Mangled names in the bundle containing `fragment` (for selecting a template instantiation).
    pub fn names_containing(&self, fragment: &str) -> Vec<String> {
        self.manifest["kernels"].as_object().map(|m| m.keys().filter(|k| k.contains(fragment)).cloned().collect()).unwrap_or_default()
    }

    /// The function for `key`: a kernel name, optionally with a variant selector `name#tag=value`
    /// (bundle v2: one name can have several CUBINs, see reference/tape/bundle.py). Without a
    /// selector, the most launched variant.
    /// Whether the bundle has a kernel of this name.
    pub fn has(&self, name: &str) -> bool { !self.manifest["kernels"][name].is_null() }

    pub fn function(&self, key: &str) -> Result<Function> {
        if let Some(f) = self.functions.lock().unwrap().get(key) { return Ok(f.clone()); }
        let (name, selector) = match key.split_once('#') { Some((n, s)) => (n, Some(s)), None => (key, None) };
        let entry = &self.manifest["kernels"][name];
        if entry.is_null() { return Err(format!("kernel bundle has no kernel {name}")); }
        let (sha, variant) = match selector {
            None => {
                let sha = entry["cubin"].as_str().ok_or_else(|| format!("kernel bundle entry {name} has no cubin"))?.to_string();
                let v = entry["variants"].get(&sha).cloned().unwrap_or(Value::Null);
                (sha, v)
            }
            Some(sel) => {
                // `tag=value[&tag=value...]`: a variant matches if its tags, or any of its other
                // tag sets, carry every listed value.
                let conds: Vec<(&str, &str)> = sel.split('&').map(|c| c.split_once('=').ok_or_else(|| format!("bad variant selector {sel}")))
                    .collect::<Result<_>>()?;
                let variants = entry["variants"].as_object().ok_or_else(|| format!("{name}: bundle has no variants (v1 bundle?)"))?;
                // A tag no variant carries (an older bundle) selects nothing: drop that condition.
                let carried = |tag: &str| variants.values().any(|v| !v["tags"][tag].is_null()
                    || v["other_tags"].as_array().map(|a| a.iter().any(|t| !t[tag].is_null())).unwrap_or(false));
                let conds: Vec<(&str, &str)> = conds.into_iter().filter(|(tag, _)| carried(tag)).collect();
                // A variant carrying every listed tag must match them all; failing any, a variant
                // from a bundle that never recorded some tag (merged bundles) matches on the rest.
                let set_matches = |t: &Value, wild: bool| conds.iter().all(|(tag, want)| match &t[*tag] {
                    Value::String(s) => s == want, Value::Number(n) => n.to_string() == *want, Value::Null => wild, _ => false });
                let matches = |v: &Value, wild: bool| set_matches(&v["tags"], wild)
                    || v["other_tags"].as_array().map(|a| a.iter().any(|t| set_matches(t, wild))).unwrap_or(false);
                let mut found: Vec<_> = variants.iter().filter(|(_, v)| matches(v, false)).collect();
                if found.is_empty() { found = variants.iter().filter(|(_, v)| matches(v, true)).collect(); }
                // Last, a capture without layer hooks (concurrency golds of logits only) labels
                // every launch `kind=meta`: such a variant stands for any layer kind (its other
                // tags still have to match; two candidates are an error below, not a guess).
                if found.is_empty() && conds.iter().any(|(tag, _)| *tag == "kind") {
                    let unlabeled = |t: &Value| conds.iter().all(|(tag, want)| match (&t[*tag], *tag) {
                        (Value::String(s), "kind") => s == want || s == "meta",
                        (Value::String(s), _) => s == want, (Value::Number(n), _) => n.to_string() == *want, _ => false });
                    found = variants.iter().filter(|(_, v)| unlabeled(&v["tags"]) || v["other_tags"].as_array().map(|a| a.iter().any(|t| unlabeled(t))).unwrap_or(false)).collect();
                }
                // The same variant compiled by two reference processes (merged bundles) can differ in
                // bytes: take the first bundle's.
                if found.len() > 1 {
                    let first = found.iter().map(|(sha, _)| self.cubin_root.get(*sha).copied().unwrap_or(0)).min().unwrap_or(0);
                    found.retain(|(sha, _)| self.cubin_root.get(*sha).copied().unwrap_or(0) == first);
                }
                match found.as_slice() {
                    [(sha, v)] => (sha.to_string(), (*v).clone()),
                    [] => return Err(format!("{name}: no variant with {sel}; have {:?}",
                                             variants.values().map(|v| v["tags"].to_string()).collect::<Vec<_>>())),
                    _ => return Err(format!("{name}: {} variants match {sel}", found.len())),
                }
            }
        };
        let module = {
            let mut modules = self.modules.lock().unwrap();
            if let Some(m) = modules.get(&sha) { m.clone() } else {
                let image = std::fs::read(self.cubin_path(&sha)).map_err(|e| format!("cubin {sha}: {e}"))?;
                if crate::gold::sha(&image) != sha { return Err(format!("cubin {sha} differs from its name")); }
                let m = std::sync::Arc::new(Module::load(&image)?);
                modules.insert(sha.clone(), m.clone());
                m
            }
        };
        let f = module.function(name)?;
        // Kernels above the 48 KiB default opt in to their dynamic shared memory, as their hosts do.
        let example = if variant.is_null() { entry["example"].clone() } else { variant["examples"][0].clone() };
        let static_smem = example["static_smem"].as_i64().unwrap_or(0) as u32;
        f.set_max_dynamic_shared(232448u32.saturating_sub(static_smem))?;
        self.functions.lock().unwrap().insert(key.to_string(), f.clone());
        Ok(f)
    }
}

/// A Triton kernel argument. Triton specializes integer arguments (unless the kernel marks them
/// `do_not_specialize`): a value of 1 becomes a compile-time constant and leaves the signature,
/// multiples of 16 get a divisibility hint; pointers get a 16-byte alignment hint. Each
/// specialization is a separate CUBIN of the same name (bundle tag `spec`).
#[derive(Clone, Copy, Debug)]
pub enum TArg { Ptr(u64), I32(i32), I64(i64), I32NoSpec(i32), F32(f32) }

/// Launch a Triton kernel: selects the CUBIN of the arguments' specialization (plus `extra`
/// selector conditions, e.g. "kind=c4" for constexpr-only differences), packs the remaining
/// arguments plus Triton's global and profile scratch pointers.
pub fn triton(k: &Kernels, s: Stream, name: &str, grid: [u32; 3], num_warps: u32, smem: u32, args: &[TArg], extra: &str) -> Result<()> {
    triton_with(k, s, name, grid, num_warps, smem, args, extra, &[])
}

/// [`triton`] with launch attributes (programmatic dependent launch for kernels compiled with it).
#[allow(clippy::too_many_arguments)]
pub fn triton_with(k: &Kernels, s: Stream, name: &str, grid: [u32; 3], num_warps: u32, smem: u32, args: &[TArg], extra: &str,
                   attrs: &[LaunchAttribute]) -> Result<()> {
    let mut spec = String::new();
    let mut params = Vec::new();
    for a in args {
        match *a {
            TArg::Ptr(p) => { spec.push(if p % 16 == 0 { 'P' } else { 'p' }); params.push(p_u64(p)); }
            TArg::I32(v) => { if v == 1 { continue; } spec.push(if v % 16 == 0 { 'D' } else { 'i' }); params.push(p_i32(v)); }
            TArg::I64(v) => { if v == 1 { continue; } spec.push(if v % 16 == 0 { 'D' } else { 'i' }); params.push(p_i64(v)); }
            TArg::I32NoSpec(v) => { spec.push(if v == 1 { '1' } else if v % 16 == 0 { 'D' } else { 'i' }); params.push(p_i32(v)); }
            // Floats are never specialized; the bundle's spec letter reads their bits as an integer.
            TArg::F32(v) => { let b = v.to_bits() as i32; spec.push(if b == 1 { '1' } else if b % 16 == 0 { 'D' } else { 'i' }); params.push(p_f32(v)); }
        }
    }
    params.push(p_u64(0));
    params.push(p_u64(0));
    let key = if extra.is_empty() { format!("{name}#spec={spec}") } else { format!("{name}#spec={spec}&{extra}") };
    if trace(k, &key, grid, [num_warps * 32, 1, 1], smem, &params) { return Ok(()); }
    k.function(&key)?.launch(grid, [num_warps * 32, 1, 1], smem, s, &params, attrs)
}

/// Launch trace (DSV4_TRACE=<file>): one JSON line per launch this process issues (kernel key,
/// grid, block, dynamic shared memory, parameter bytes), to diff against the reference's tape.
/// With DSV4_DRY=1 the bundle's kernels are not launched at all: a forward then only records
/// its launch sequence, and a kernel or variant the bundle lacks is recorded (`missing`) instead
/// of failing, so one run lists everything a new model or shape still needs.
pub fn dry() -> bool {
    static DRY: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *DRY.get_or_init(|| std::env::var("DSV4_DRY").as_deref() == Ok("1"))
}

/// Records a launch; returns true when the launch must be skipped (dry mode).
pub fn trace(k: &Kernels, key: &str, grid: [u32; 3], block: [u32; 3], smem: u32, params: &[Vec<u8>]) -> bool {
    static OUT: std::sync::OnceLock<Option<Mutex<std::fs::File>>> = std::sync::OnceLock::new();
    let out = OUT.get_or_init(|| std::env::var("DSV4_TRACE").ok().and_then(|p| std::fs::File::create(p).ok()).map(Mutex::new));
    if let Some(f) = out {
        use std::io::Write;
        let missing = if dry() { k.function(key).err() } else { None };
        let hex: Vec<String> = params.iter().map(|p| p.iter().map(|b| format!("{b:02x}")).collect()).collect();
        let _ = writeln!(f.lock().unwrap(), "{}", serde_json::json!({"key": key, "grid": grid, "block": block, "smem": smem, "params": hex, "missing": missing}));
    }
    dry()
}

/// One kernel launch, as the reference's host code issues it.
pub struct Launch<'a> {
    pub name: &'a str,
    pub grid: [u32; 3],
    pub block: [u32; 3],
    pub smem: u32,
    pub attrs: Vec<LaunchAttribute>,
    pub params: Vec<Vec<u8>>,
}

impl Launch<'_> {
    pub fn run(&self, kernels: &Kernels, stream: Stream) -> Result<()> {
        if trace(kernels, self.name, self.grid, self.block, self.smem, &self.params) { return Ok(()); }
        kernels.function(self.name)?.launch(self.grid, self.block, self.smem, stream, &self.params, &self.attrs)
    }
}

/// Launch attribute helpers (CUlaunchAttributeID values).
pub fn attr_pdl() -> LaunchAttribute {
    let mut value = [0u8; 64];
    value[0] = 1;
    LaunchAttribute { id: 6, pad: [0; 4], value } // PROGRAMMATIC_STREAM_SERIALIZATION
}

pub fn attr_cluster(x: u32, y: u32, z: u32) -> LaunchAttribute {
    let mut value = [0u8; 64];
    value[0..4].copy_from_slice(&x.to_le_bytes());
    value[4..8].copy_from_slice(&y.to_le_bytes());
    value[8..12].copy_from_slice(&z.to_le_bytes());
    LaunchAttribute { id: 4, pad: [0; 4], value } // CLUSTER_DIMENSION
}

/// CLUSTER_SCHEDULING_POLICY_PREFERENCE = CU_CLUSTER_SCHEDULING_POLICY_SPREAD.
pub fn attr_cluster_spread() -> LaunchAttribute {
    let mut value = [0u8; 64];
    value[0] = 1;
    LaunchAttribute { id: 5, pad: [0; 4], value }
}

/// Parameter byte builders.
pub fn p_u64(v: u64) -> Vec<u8> { v.to_le_bytes().to_vec() }
pub fn p_i64(v: i64) -> Vec<u8> { v.to_le_bytes().to_vec() }
pub fn p_i32(v: i32) -> Vec<u8> { v.to_le_bytes().to_vec() }
pub fn p_u32(v: u32) -> Vec<u8> { v.to_le_bytes().to_vec() }
pub fn p_f32(v: f32) -> Vec<u8> { v.to_le_bytes().to_vec() }
pub fn p_map(m: &cuda::TensorMap) -> Vec<u8> { m.0.to_vec() }

/// Little-endian struct writer for by-value parameter structs.
#[derive(Default)]
pub struct Pod(pub Vec<u8>);

impl Pod {
    pub fn align(&mut self, a: usize) -> &mut Self { while self.0.len() % a != 0 { self.0.push(0); } self }
    pub fn u64(&mut self, v: u64) -> &mut Self { self.align(8); self.0.extend(v.to_le_bytes()); self }
    pub fn i64(&mut self, v: i64) -> &mut Self { self.align(8); self.0.extend(v.to_le_bytes()); self }
    pub fn u32(&mut self, v: u32) -> &mut Self { self.align(4); self.0.extend(v.to_le_bytes()); self }
    pub fn i32(&mut self, v: i32) -> &mut Self { self.align(4); self.0.extend(v.to_le_bytes()); self }
    pub fn f32(&mut self, v: f32) -> &mut Self { self.align(4); self.0.extend(v.to_le_bytes()); self }
    pub fn u8(&mut self, v: u8) -> &mut Self { self.0.push(v); self }
    pub fn bytes(&mut self, b: &[u8]) -> &mut Self { self.0.extend_from_slice(b); self }
    pub fn pad_to(&mut self, n: usize) -> &mut Self { while self.0.len() < n { self.0.push(0); } self }
    pub fn take(&mut self) -> Vec<u8> { std::mem::take(&mut self.0) }
}

/// A device bump allocator: activations of one forward, released together.
pub struct Arena { pub buffer: Buffer, offset: usize, pub high_water: usize }

impl Arena {
    pub fn new(bytes: usize) -> Result<Self> { Ok(Self { buffer: Buffer::alloc(bytes)?, offset: 0, high_water: 0 }) }
    /// 512-byte aligned like PyTorch's caching allocator blocks.
    pub fn alloc(&mut self, bytes: usize) -> Result<u64> {
        let start = self.offset.div_ceil(512) * 512;
        let end = start + bytes.max(1);
        if end > self.buffer.bytes { return Err(format!("arena exhausted: {end} > {}", self.buffer.bytes)); }
        self.offset = end;
        self.high_water = self.high_water.max(end);
        Ok(self.buffer.ptr + start as u64)
    }
    pub fn reset(&mut self) { self.offset = 0; }
    /// Bytes an allocation can still take (after 512-byte alignment).
    pub fn remaining(&self) -> usize { self.buffer.bytes.saturating_sub(self.offset.div_ceil(512) * 512) }
}

/// The CUPTI launch tape, when the process runs with it injected (CUDA_INJECTION64_PATH): label
/// native launches the way the reference's are labelled so the two tapes can be diffed.
pub struct NativeTape { start: Option<unsafe extern "C" fn(*const std::ffi::c_char)>, stop: Option<unsafe extern "C" fn()> }

impl NativeTape {
    pub fn open() -> Self {
        let none = Self { start: None, stop: None };
        let Ok(path) = std::env::var("CUDA_INJECTION64_PATH") else { return none };
        if !path.contains("cupti_tape") { return none; }
        extern "C" {
            fn dlopen(file: *const std::ffi::c_char, mode: i32) -> *mut std::ffi::c_void;
            fn dlsym(handle: *mut std::ffi::c_void, name: *const std::ffi::c_char) -> *mut std::ffi::c_void;
        }
        let c = std::ffi::CString::new(path).unwrap();
        let h = unsafe { dlopen(c.as_ptr(), 2 /* RTLD_NOW */) };
        if h.is_null() { return none; }
        let start = unsafe { dlsym(h, c"tape_start".as_ptr()) };
        let stop = unsafe { dlsym(h, c"tape_stop".as_ptr()) };
        if start.is_null() || stop.is_null() { return none; }
        Self { start: Some(unsafe { std::mem::transmute(start) }), stop: Some(unsafe { std::mem::transmute(stop) }) }
    }
    pub fn label(&self, label: &str) {
        if let Some(f) = self.start { let c = std::ffi::CString::new(label).unwrap(); unsafe { f(c.as_ptr()) } }
    }
    pub fn stop(&self) { if let Some(f) = self.stop { unsafe { f() } } }
}

// ---------------------------------------------------------------------------------------------
// Launch redirection (kernels/redirect.cpp): reference host code compiled into this binary
// launches through the runtime; those launches are re-issued from the kernel bundle.
// ---------------------------------------------------------------------------------------------

static REDIRECT_KERNELS: std::sync::OnceLock<&'static Kernels> = std::sync::OnceLock::new();

type Handler = unsafe extern "C" fn(*const std::ffi::c_char, *const u32, *const u32, u32, *mut std::ffi::c_void,
                                    *mut *mut std::ffi::c_void, *const std::ffi::c_void, u32, *mut i32) -> i32;

extern "C" {
    fn dsv4_set_launch_handler(h: Option<Handler>);
}

unsafe extern "C" fn redirect_handler(name: *const std::ffi::c_char, grid: *const u32, block: *const u32, smem: u32,
                                      stream: *mut std::ffi::c_void, args: *mut *mut std::ffi::c_void,
                                      attrs: *const std::ffi::c_void, num_attrs: u32, err: *mut i32) -> i32 {
    let Some(kernels) = REDIRECT_KERNELS.get() else { return 0 };
    let name = std::ffi::CStr::from_ptr(name).to_string_lossy().into_owned();
    if std::env::var_os("DSV4_REDIRECT_LOG").is_some() { eprintln!("redirect: {}", &name[..name.len().min(140)]); }
    let f = match kernels.function(&name) {
        Ok(f) => f,
        Err(e) => {
            if dry() { trace(kernels, &name, [*grid, *grid.add(1), *grid.add(2)], [*block, *block.add(1), *block.add(2)], smem, &[]); *err = 0; return 1; }
            eprintln!("redirect: {e}"); *err = 1; return 1;
        }
    };
    let layout = f.params();
    let params: Vec<Vec<u8>> = layout.iter().enumerate()
        .map(|(i, &(_, size))| std::slice::from_raw_parts(*args.add(i) as *const u8, size).to_vec()).collect();
    let attrs: Vec<LaunchAttribute> = (0..num_attrs as usize)
        .map(|i| *(attrs as *const LaunchAttribute).add(i)).collect();
    let g = [*grid, *grid.add(1), *grid.add(2)];
    let b = [*block, *block.add(1), *block.add(2)];
    if trace(kernels, &name, g, b, smem, &params) { *err = 0; return 1; }
    match f.launch(g, b, smem, Stream(stream), &params, &attrs) {
        Ok(()) => { *err = 0; 1 }
        Err(e) => { eprintln!("redirect {name}: {e}"); *err = 1; 1 }
    }
}

/// Route compiled reference host code's runtime launches to `kernels` for the rest of the process.
pub fn install_redirect(kernels: &'static Kernels) {
    if REDIRECT_KERNELS.set(kernels).is_ok() {
        unsafe { dsv4_set_launch_handler(Some(redirect_handler)) };
    }
}
