//! Serving always uses trusted checkpoint metadata. Only `Session::run` with a
//! non-exported (reference) configuration keeps the full-file verification path.
use native_core_infra::checkpoint::{Declaration, Source};

pub(crate) const RELOAD_HEADROOM_BYTES: usize = 3usize << 30;

/// A fresh load keeps its dependency schedule. Only a consumed/evicted load
/// needs the existing complete-copy reload operation before inference.
pub(crate) fn needs_prefetch(resident: bool, has_pending_components: bool) -> bool {
    resident && !has_pending_components
}

pub(crate) fn checkpoint(source: Option<&Source>) -> Result<Declaration, String> {
    if let Some(source) = source {
        return source.load();
    }
    require(native_core_infra::checkpoint::Declaration::from_env()?)
}

pub(crate) fn require(declaration: Option<Declaration>) -> Result<Declaration, String> {
    declaration.ok_or_else(|| {
        "LTX serving requires a trusted checkpoint declaration; pass init.checkpoint or NATIVE_CHECKPOINT_{MANIFEST,SHA256,ROOTS}".into()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serving_cannot_silently_select_full_file_fallback() {
        assert!(require(None).err().unwrap().contains("trusted checkpoint declaration"));
    }

    #[test]
    fn explicit_untrusted_checkpoint_does_not_use_an_environment_fallback() {
        let source = Source {
            manifest: "relative/manifest.json".into(),
            sha256: "0".repeat(64),
            roots: [("weights".into(), "/original/weights".into())].into(),
        };
        assert!(checkpoint(Some(&source)).err().unwrap().contains("absolute"));
    }

    #[test]
    fn fresh_components_do_not_get_an_all_copy_barrier() {
        assert!(!needs_prefetch(true, true));
        assert!(needs_prefetch(true, false));
        assert!(!needs_prefetch(false, false));
    }
}
