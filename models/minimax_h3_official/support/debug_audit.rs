//! Request-scoped diagnostic policy. Model integrity checks do not use this switch.
use std::cell::Cell;
thread_local! { static ENABLED: Cell<bool> = const { Cell::new(false) }; }

pub fn begin() -> Result<bool, String> {
    let value = match std::env::var_os("H3_DEBUG_AUDIT_MODE_FILE") {
        Some(path) => std::fs::read_to_string(&path)
            .map_err(|e| format!("cannot read H3_DEBUG_AUDIT_MODE_FILE: {e}"))?,
        None => match std::env::var("H3_DEBUG_AUDIT") {
            Ok(value) => value,
            Err(std::env::VarError::NotPresent) => "0".into(),
            Err(error) => return Err(format!("invalid H3_DEBUG_AUDIT: {error}")),
        },
    };
    let enabled = match value.trim() {
        "0" => false,
        "1" => true,
        _ => return Err("H3_DEBUG_AUDIT must be exactly 0 or 1".into()),
    };
    ENABLED.with(|state| state.set(enabled));
    Ok(enabled)
}
pub fn enabled() -> bool { ENABLED.with(Cell::get) }
pub fn sha(f: impl FnOnce() -> String) -> serde_json::Value {
    if enabled() { serde_json::Value::String(f()) } else { serde_json::Value::Null }
}
pub fn record() -> serde_json::Value {
    serde_json::json!({"enabled":enabled(),"repeated_sha":enabled(),
        "diagnostic_files":enabled(),"independent_output_acceptance_required":true})
}

/// Structural tensors define resident indexing/RoPE. Their contents remain a
/// semantic invariant even when optional cryptographic audits are disabled.
pub fn check_fixed_bytes(cache: &mut std::collections::HashMap<String, Vec<u8>>,
    name: &str, bytes: &[u8]) -> Result<(), String> {
    match cache.get(name) {
        Some(expected) if expected.as_slice() != bytes =>
            Err(format!("resident structural input content changed: {name}")),
        Some(_) => Ok(()),
        None => { cache.insert(name.to_owned(), bytes.to_vec()); Ok(()) },
    }
}
