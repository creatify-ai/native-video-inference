//! Bounded, content-addressed host assets. Only complete, SHA-verified files
//! enter the cache; callers retain immutable bytes without another copy/read.
use crate::Result;
use sha2::{Digest, Sha256};
use std::{cell::{Cell, RefCell}, collections::BTreeMap, fs::File, io::Read,
          path::Path, rc::Rc};
use std::{path::PathBuf, sync::{Arc, Mutex, OnceLock}};

#[derive(Default)]
struct Snapshots {
    bytes: usize,
    paths: BTreeMap<PathBuf, String>,
    contents: BTreeMap<String, Arc<Vec<u8>>>,
}

/// Explicitly registered CPU assets, shared by a backend's GPU worker threads.
/// Readers use the verified snapshot, never bytes reread after verification.
/// Request inputs and checkpoint payloads must not be registered here.
pub struct SharedAssets { limit: usize, max_paths: usize, snapshots: Mutex<Snapshots> }
impl SharedAssets {
    pub fn new(limit: usize, max_paths: usize) -> Self {
        Self { limit, max_paths, snapshots: Mutex::new(Snapshots::default()) }
    }
    pub fn register(&self, path: &Path, expected: &str) -> Result<Arc<Vec<u8>>> {
        if !path.is_absolute() || expected.len() != 64 ||
            !expected.bytes().all(|v| v.is_ascii_hexdigit() && !v.is_ascii_uppercase()) {
            return Err("immutable snapshot requires absolute path and full lowercase SHA256".into());
        }
        let mut snapshots = self.snapshots.lock().map_err(|_| "immutable snapshot lock poisoned")?;
        if let Some(previous) = snapshots.paths.get(path) {
            if previous != expected { return Err("immutable snapshot path identity changed".into()); }
            return Ok(snapshots.contents[expected].clone());
        }
        if snapshots.paths.len() >= self.max_paths { return Err("immutable snapshot path budget exceeded".into()); }
        let value = if let Some(value) = snapshots.contents.get(expected) { value.clone() } else {
            let mut file = File::open(path)?;
            let bytes = usize::try_from(file.metadata()?.len())?;
            let used = snapshots.bytes.checked_add(bytes).ok_or("immutable snapshot extent overflow")?;
            if used > self.limit { return Err("immutable snapshot byte budget exceeded".into()); }
            let mut value = Vec::new(); value.try_reserve_exact(bytes)?; value.resize(bytes, 0);
            file.read_exact(&mut value)?;
            if file.read(&mut [0u8; 1])? != 0 || format!("{:x}", Sha256::digest(&value)) != expected {
                return Err("immutable snapshot SHA256/extent mismatch".into());
            }
            let value = Arc::new(value);
            snapshots.contents.insert(expected.into(), value.clone()); snapshots.bytes = used;
            value
        };
        snapshots.paths.insert(path.to_owned(), expected.into());
        Ok(value)
    }
    pub fn read(&self, path: &Path) -> Result<Arc<Vec<u8>>> {
        {
            let snapshots = self.snapshots.lock().map_err(|_| "immutable snapshot lock poisoned")?;
            if let Some(sha) = snapshots.paths.get(path) { return Ok(snapshots.contents[sha].clone()); }
        }
        Ok(Arc::new(std::fs::read(path)?))
    }
}
fn shared() -> &'static SharedAssets {
    static ASSETS: OnceLock<SharedAssets> = OnceLock::new();
    ASSETS.get_or_init(|| SharedAssets::new(2usize << 30, 16384))
}
pub fn register(path: &Path, expected: &str) -> Result<Arc<Vec<u8>>> { shared().register(path, expected) }
pub fn read_registered_or_file(path: &Path) -> Result<Arc<Vec<u8>>> { shared().read(path) }

pub struct ImmutableAssets {
    limit: usize,
    max_entries: usize,
    used: Cell<usize>,
    entries: RefCell<BTreeMap<String, Rc<Vec<u8>>>>,
}

impl ImmutableAssets {
    pub fn new(limit: usize, max_entries: usize) -> Self {
        Self { limit, max_entries, used: Cell::new(0), entries: RefCell::new(BTreeMap::new()) }
    }
    pub fn bytes(&self) -> usize { self.used.get() }
    pub fn entries(&self) -> usize { self.entries.borrow().len() }

    pub fn load(&self, path: &Path, expected: &str, bytes: usize) -> Result<Rc<Vec<u8>>> {
        if expected.len() != 64 || !expected.bytes().all(|v| v.is_ascii_hexdigit() && !v.is_ascii_uppercase()) {
            return Err("full lowercase asset SHA256 required".into());
        }
        if let Some(value) = self.entries.borrow().get(expected) {
            if value.len() != bytes { return Err("cached asset extent differs".into()); }
            return Ok(value.clone());
        }
        let _trace=crate::startup_trace::Span::new("model_asset_load",path.to_string_lossy(),bytes);
        let used = self.bytes().checked_add(bytes).ok_or("host asset cache extent overflow")?;
        if used > self.limit || self.entries() >= self.max_entries {
            return Err("immutable host asset cache budget exceeded".into());
        }
        let mut file = File::open(path)?;
        if file.metadata()?.len() != u64::try_from(bytes)? {
            return Err("immutable asset file extent differs".into());
        }
        let mut value = Vec::new();
        value.try_reserve_exact(bytes)?;
        value.resize(bytes, 0);
        file.read_exact(&mut value)?;
        if file.read(&mut [0u8; 1])? != 0 { return Err("immutable asset grew during load".into()); }
        if format!("{:x}", Sha256::digest(&value)) != expected {
            return Err("immutable asset SHA256 mismatch".into());
        }
        let value = Rc::new(value);
        self.entries.borrow_mut().insert(expected.into(), value.clone());
        self.used.set(used);
        Ok(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture(name: &str, data: &[u8]) -> (std::path::PathBuf, String) {
        let path = std::env::temp_dir().join(format!("immutable-assets-{}-{name}", std::process::id()));
        std::fs::write(&path, data).unwrap();
        (path, format!("{:x}", Sha256::digest(data)))
    }
    #[test]
    fn retains_one_verified_snapshot_by_content_identity() {
        let (path, sha) = fixture("snapshot", b"immutable");
        let cache = ImmutableAssets::new(9, 1);
        let first = cache.load(&path, &sha, 9).unwrap();
        std::fs::remove_file(&path).unwrap();
        let second = cache.load(&path, &sha, 9).unwrap();
        assert!(Rc::ptr_eq(&first, &second));
        assert_eq!(cache.bytes(), 9);
        assert_eq!(cache.entries(), 1);
        assert!(cache.load(&path, &sha, 8).is_err());
    }
    #[test]
    fn corruption_and_truncation_never_enter_the_cache() {
        let (path, sha) = fixture("corrupt", b"correct");
        let cache = ImmutableAssets::new(32, 4);
        std::fs::write(&path, b"wrong!!").unwrap();
        assert!(cache.load(&path, &sha, 7).is_err());
        std::fs::write(&path, b"short").unwrap();
        assert!(cache.load(&path, &sha, 7).is_err());
        assert_eq!((cache.bytes(), cache.entries()), (0, 0));
        std::fs::remove_file(path).unwrap();
    }
    #[test]
    fn bounds_are_checked_before_opening_or_allocating() {
        let cache = ImmutableAssets::new(8, 1);
        let path = Path::new("/nonexistent-immutable-asset");
        let sha = "0".repeat(64);
        assert!(cache.load(path, &sha, 9).unwrap_err().to_string().contains("budget"));
        let (path, sha) = fixture("count", b"a");
        cache.load(&path, &sha, 1).unwrap();
        assert!(cache.load(&path, &"1".repeat(64), 1).unwrap_err().to_string().contains("budget"));
        std::fs::remove_file(path).unwrap();
    }
    #[test]
    fn shared_snapshots_retain_verified_bytes_and_reject_rebinding() {
        let (path, sha) = fixture("shared", b"verified");
        let cache = SharedAssets::new(8, 2);
        let first = cache.register(&path, &sha).unwrap();
        std::fs::write(&path, b"modified").unwrap();
        assert!(Arc::ptr_eq(&first, &cache.read(&path).unwrap()));
        assert!(Arc::ptr_eq(&first, &cache.register(&path, &sha).unwrap()));
        assert!(cache.register(&path, &format!("{:x}", Sha256::digest(b"modified"))).is_err());
        std::fs::remove_file(path).unwrap();
    }
    #[test]
    fn shared_snapshot_failures_leave_capacity_and_dynamic_reads_intact() {
        let (path, sha) = fixture("shared-failure", b"verified");
        let cache = SharedAssets::new(8, 1);
        assert!(cache.register(&path, &"0".repeat(64)).is_err());
        std::fs::write(&path, b"unbound!").unwrap();
        assert_eq!(cache.read(&path).unwrap().as_slice(), b"unbound!");
        std::fs::write(&path, b"verified").unwrap();
        cache.register(&path, &sha).unwrap();
        assert!(cache.register(Path::new("/missing-shared-snapshot"), &sha).unwrap_err().to_string().contains("budget"));
        assert!(SharedAssets::new(7, 1).register(&path, &sha).unwrap_err().to_string().contains("budget"));
        std::fs::remove_file(path).unwrap();
    }
}
