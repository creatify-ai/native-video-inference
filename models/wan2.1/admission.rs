//! Trusted checkpoint admission for the maintained Wan CLI and resident owner.
//!
//! Only metadata is read here. Payload reads and preparation remain owned by
//! the framework, with Wan's existing encoder-then-DiT dependency schedule.
use native_core_infra::checkpoint::{Declaration, Source};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// The maintained cold path must prepare its backend before computation.
/// The old diagnostic switch disabled this work by presence, even when set to 0.
pub fn require_backend_preparation() -> Result<(), String> {
    backend_preparation(std::env::var_os("WAN_NO_PREWARM").is_some())
}

fn backend_preparation(disabled: bool) -> Result<(), String> {
    if disabled {
        return Err("Wan declared startup requires backend preparation; remove WAN_NO_PREWARM".into());
    }
    Ok(())
}

pub fn checkpoint(prepared: &Path, published: &Path) -> Result<Declaration, String> {
    let declared = Declaration::from_env()?;
    let pin = std::env::var("WAN21_MANIFEST_SHA256").ok();
    admit(prepared, published, declared, pin.as_deref())
}

fn absolute(path: &Path) -> Result<PathBuf, String> {
    path.canonicalize().map_err(|e| format!("{}: {e}", path.display()))
}

fn admit(prepared: &Path, published: &Path, declared: Option<Declaration>,
         legacy_pin: Option<&str>) -> Result<Declaration, String> {
    if declared.is_none() && legacy_pin.is_none() {
        return Err("Wan checkpoint requires NATIVE_CHECKPOINT_MANIFEST/SHA256/ROOTS or the offline WAN21_MANIFEST_SHA256 pin".into());
    }
    let prepared = absolute(prepared)?;
    let published = absolute(published)?;
    let expected = HashMap::from([("prepared".to_string(), prepared.clone()),
                                  ("checkpoint".to_string(), published)]);
    let declaration = match declared {
        Some(declaration) => declaration,
        None => Source { manifest: prepared.join("manifest.json"),
            sha256: legacy_pin.unwrap().to_string(), roots: expected.clone() }.load()?,
    };
    // Tokenizer/input assets still use the model's published directory. Do not
    // admit weights from one checkpoint while reading those assets from another.
    for (name, wanted) in expected {
        let actual = declaration.roots.get(&name)
            .ok_or_else(|| format!("Wan checkpoint declaration lacks the {name} root"))?;
        if absolute(actual)? != wanted {
            return Err(format!("Wan checkpoint {name} root differs from the configured model directory"));
        }
    }
    Ok(declaration)
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};
    use std::sync::atomic::{AtomicUsize, Ordering};
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    struct Fixture { root: PathBuf, pin: String }
    impl Fixture {
        fn new() -> Self {
            let root = std::env::temp_dir().join(format!("wan-admission-{}-{}", std::process::id(), NEXT.fetch_add(1, Ordering::Relaxed)));
            std::fs::create_dir_all(root.join("prepared")).unwrap();
            std::fs::create_dir_all(root.join("published")).unwrap();
            // Admission must not read payload files. They intentionally do not exist.
            let bytes = br#"{"schema":"native-checkpoint-v1","segment_bytes":1024,"files":{},"tensors":{}}"#;
            std::fs::write(root.join("prepared/manifest.json"), bytes).unwrap();
            Self { root, pin: format!("{:x}", Sha256::digest(bytes)) }
        }
        fn source(&self) -> Source {
            Source { manifest: self.root.join("prepared/manifest.json"), sha256: self.pin.clone(),
                roots: HashMap::from([("prepared".into(), self.root.join("prepared")),
                                      ("checkpoint".into(), self.root.join("published"))]) }
        }
        fn load(&self, declared: Option<Declaration>, pin: Option<&str>) -> Result<Declaration, String> {
            admit(&self.root.join("prepared"), &self.root.join("published"), declared, pin)
        }
    }
    impl Drop for Fixture { fn drop(&mut self) { let _ = std::fs::remove_dir_all(&self.root); } }

    #[test]
    fn declared_startup_cannot_omit_required_backend_preparation() {
        assert!(backend_preparation(false).is_ok());
        assert!(backend_preparation(true).err().unwrap().contains("remove WAN_NO_PREWARM"));
    }

    #[test]
    fn missing_or_wrong_pin_fails_before_payload_reads() {
        let f = Fixture::new();
        assert!(f.load(None, None).err().unwrap().contains("offline WAN21_MANIFEST_SHA256"));
        assert!(f.load(None, Some(&"0".repeat(64))).err().unwrap().contains("digest mismatch"));
        assert!(f.load(None, Some(&f.pin)).is_ok());
    }

    #[test]
    fn standard_declaration_and_pinned_compatibility_route_have_the_same_roots() {
        let f = Fixture::new();
        let standard = f.load(Some(f.source().load().unwrap()), None).unwrap();
        let compatible = f.load(None, Some(&f.pin)).unwrap();
        for name in ["prepared", "checkpoint"] {
            assert_eq!(absolute(&standard.roots[name]).unwrap(), absolute(&compatible.roots[name]).unwrap());
        }
    }

    #[test]
    fn declared_weight_roots_must_match_frontend_assets() {
        let f = Fixture::new();
        let mut source = f.source();
        source.roots.insert("checkpoint".into(), f.root.join("prepared"));
        assert!(f.load(Some(source.load().unwrap()), None).err().unwrap().contains("differs"));
        source.roots.remove("checkpoint");
        assert!(f.load(Some(source.load().unwrap()), None).err().unwrap().contains("lacks the checkpoint root"));
    }
}
