//! Distinguish an owner lifetime from a recycled PID or another Modal container.
use std::sync::OnceLock;
pub fn current()->String{
    static ID:OnceLock<String>=OnceLock::new();
    ID.get_or_init(||{
        let start=std::fs::read_to_string("/proc/self/stat").ok().and_then(|stat|
            stat.rsplit_once(") ").and_then(|(_,fields)|fields.split_whitespace().nth(19)).map(str::to_owned))
            .unwrap_or_else(||"unavailable".into());
        format!("{}:{}:{}",std::env::var("MODAL_TASK_ID").unwrap_or_else(|_|"local".into()),std::process::id(),start)
    }).clone()
}
