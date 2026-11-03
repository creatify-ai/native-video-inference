//! Public checkpoint admission and model-independent SHA helpers.
use crate::Result;
use serde_json::Value;
use sha2::{Digest, Sha256};

pub fn digest(bytes: &[u8]) -> String {
    format!("{:x}", crate::startup_trace::measure("sha256", "", bytes.len(), || Sha256::digest(bytes)))
}
pub fn valid_sha(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
}
pub fn base_identity(model: &Value, bytes: &[u8]) -> Result<String> {
    let value = if let Some(shared) = model.get("shared_weights") {
        shared["manifest_sha256"].as_str().ok_or("shared weight SHA missing")?.to_owned()
    } else { digest(bytes) };
    if !valid_sha(&value) { return Err("invalid base weight SHA256".into()); }
    Ok(value)
}
/// Fields of the single official checkpoint identity; anything else is rejected.
const SOURCE_FIELDS: [&str; 4] = ["repo_id", "variant", "revision", "partition"];

pub fn admit_base(model: &Value) -> Result<()> {
    let source = model["model_source"].as_object().ok_or("public H3 requires an official base checkpoint identity")?;
    if source.keys().any(|key| !SOURCE_FIELDS.contains(&key.as_str())) {
        return Err("public H3 accepts only the official checkpoint identity fields".into());
    }
    let revision = source.get("revision").and_then(Value::as_str).ok_or("public model requires a pinned checkpoint revision")?;
    if source.get("repo_id").and_then(Value::as_str) != Some("MiniMaxAI/MiniMax-H3")
        || source.get("variant").and_then(Value::as_str) != Some("base")
        || revision.len() != 40 || !revision.bytes().all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c)) {
        return Err("public H3 requires an official base checkpoint identity".into());
    }
    if let Some(partition) = source.get("partition") {
        if partition != "FL2VA" && partition != "Ref2VA" { return Err("public H3 requires an official partition".into()); }
    }
    // One package, one checkpoint: no nested identity can add another weight source.
    for (key, item) in model.as_object().ok_or("model manifest must be an object")? {
        if key != "model_source" && declares_identity(item) {
            return Err("public H3 model packages declare exactly one checkpoint source".into());
        }
    }
    Ok(())
}
fn declares_identity(value: &Value) -> bool {
    match value {
        Value::Object(fields) => fields.contains_key("repo_id") || fields.contains_key("model_source")
            || fields.values().any(declares_identity),
        Value::Array(items) => items.iter().any(declares_identity),
        _ => false,
    }
}

#[cfg(test)] mod tests {
    use super::*;
    use serde_json::json;
    #[test] fn requires_a_pinned_official_base_source() {
        let model = json!({"model_source":{"repo_id":"MiniMaxAI/MiniMax-H3", "variant":"base", "revision":"a".repeat(40)}});
        assert!(admit_base(&model).is_ok());
        assert!(admit_base(&json!({})).is_err());
        let mut changed=model.clone();changed["model_source"]["revision"]=json!("main");
        assert!(admit_base(&changed).is_err());
        let mut changed=model.clone();changed["model_source"]["partition"]=json!("Ref2VA");
        assert!(admit_base(&changed).is_ok());
        changed["model_source"]["partition"]=json!("other");
        assert!(admit_base(&changed).is_err());
    }
    #[test] fn rejects_extra_identity_fields_and_second_sources() {
        let model = json!({"model_source":{"repo_id":"MiniMaxAI/MiniMax-H3", "variant":"base", "revision":"a".repeat(40)}});
        let mut changed=model.clone();changed["model_source"]["weights"]=json!("/other");
        assert!(admit_base(&changed).is_err());
        for nested in [json!({"repo_id":"other/model"}), json!([{"model_source":{}}]), json!({"a":{"b":{"repo_id":"x"}}})] {
            let mut changed=model.clone();changed["extension"]=nested;
            assert!(admit_base(&changed).is_err());
        }
        let mut shared=model.clone();shared["shared_weights"]=json!({"root":"/models/base","manifest_sha256":"b".repeat(64)});
        assert!(admit_base(&shared).is_ok());
    }
}
