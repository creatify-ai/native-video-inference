//! Request-scoped optional audit. CLI children are one request per process.
use std::sync::OnceLock;
static ENABLED: OnceLock<bool> = OnceLock::new();
pub fn begin_request() -> crate::Result<bool> {
    let raw = match std::env::var_os("H3_DEBUG_AUDIT_MODE_FILE") {
        Some(path) => std::fs::read_to_string(path)?,
        None => match std::env::var("H3_DEBUG_AUDIT") {
            Ok(value) => value,
            Err(std::env::VarError::NotPresent) => "0".to_string(),
            Err(error) => return Err(error.into()),
        },
    };
    let value = match raw.trim() {
        "0" => false, "1" => true,
        _ => return Err("H3_DEBUG_AUDIT must be 0 or 1 (mode file overrides environment)".into()),
    };
    ENABLED.set(value).map_err(|_| "audit mode already initialized for this process")?;
    Ok(value)
}
pub fn enabled() -> bool { *ENABLED.get().expect("audit mode must be initialized before request") }
pub fn hash(bytes: &[u8]) -> serde_json::Value {
    if enabled() { serde_json::json!(crate::sha256(bytes)) } else { serde_json::Value::Null }
}
