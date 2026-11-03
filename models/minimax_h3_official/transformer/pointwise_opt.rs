//! Read once per resident request. Independent of optional output auditing.
use std::cell::Cell;
thread_local! { static ENABLED: Cell<bool> = const { Cell::new(true) }; }
#[cfg(feature="cuda")]
extern "C" {
    fn h3_pointwise_opt_set_mode(mode: i32);
    fn h3_pointwise_opt_calls(mode: i32) -> u64;
}
pub fn begin() -> Result<(), String> {
    let raw = match std::env::var_os("H3_POINTWISE_OPT_MODE_FILE") {
        Some(path) => std::fs::read_to_string(path).map_err(|e| e.to_string())?,
        None => match std::env::var("H3_POINTWISE_OPT") {
            Ok(value) => value,
            Err(std::env::VarError::NotPresent) => "1".into(),
            Err(e) => return Err(e.to_string()),
        },
    };
    let enabled = match raw.trim() {
        "0" => false, "1" => true,
        _ => return Err("H3_POINTWISE_OPT must be 0 or 1".into()),
    };
    ENABLED.with(|state| state.set(enabled));
    #[cfg(feature="cuda")]
    unsafe { h3_pointwise_opt_set_mode(enabled as i32); }
    Ok(())
}
pub fn enabled()->bool { ENABLED.with(Cell::get) }
pub fn record() -> serde_json::Value {
    #[cfg(feature="cuda")]
    let calls = unsafe { [h3_pointwise_opt_calls(0), h3_pointwise_opt_calls(1)] };
    #[cfg(not(feature="cuda"))]
    let calls = [0u64, 0u64];
    serde_json::json!({"enabled":ENABLED.with(Cell::get),
        "baseline_dispatch_calls":calls[0],"optimized_dispatch_calls":calls[1]})
}
