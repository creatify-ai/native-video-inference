//! Read exactly once at a request boundary, never while launching a kernel.
pub fn read() -> Result<bool, String> {
    let value = if let Some(path) = std::env::var_os("H3_AUDIO_POOL_MODE_FILE") {
        if path.is_empty() { return Err("H3_AUDIO_POOL_MODE_FILE is empty".into()); }
        let path = std::path::Path::new(&path);
        if std::fs::metadata(path).map_err(|e| format!("audio pool mode file: {e}"))?.len() > 32 {
            return Err("audio pool mode file exceeds 32 bytes".into());
        }
        std::fs::read_to_string(path).map_err(|e| format!("audio pool mode file: {e}"))?
    } else {
        match std::env::var("H3_AUDIO_POOL") {
            Ok(value) => value,
            Err(std::env::VarError::NotPresent) => "1".into(),
            Err(_) => return Err("H3_AUDIO_POOL must be UTF-8 0 or 1".into()),
        }
    };
    match value.trim() {
        "0" => Ok(false),
        "1" => Ok(true),
        _ => Err("H3_AUDIO_POOL mode must be exactly 0 or 1".into()),
    }
}
