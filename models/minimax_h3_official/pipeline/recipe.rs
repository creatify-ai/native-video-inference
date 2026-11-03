//! Official partitions and model evaluations. Sigma endpoints are not evaluations.
use crate::Result;
use serde_json::Value;

pub const DEFAULT_NFE: usize = 49;
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Recipe { pub workflow: &'static str, pub nfe: usize }
impl Recipe {
    pub fn from_request(request: &Value) -> Result<Self> {
        let workflow=match request.get("workflow") {
            None => "fl2va",
            Some(Value::String(value)) => value,
            _ => return Err("workflow must be an official H3 partition".into()),
        };
        let workflow=Self::partition(workflow).ok_or("unsupported public H3 partition")?;
        let nfe=match request.get("nfe") {
            None => DEFAULT_NFE,
            Some(value) => usize::try_from(value.as_u64().ok_or("NFE must be a positive integer")?)?,
        };
        if !(1..=999).contains(&nfe) { return Err("public H3 supports 1..999 model evaluations".into()); }
        Ok(Self {workflow,nfe})
    }
    pub fn from_model(model: &Value) -> Result<Self> {
        let value=model.get("recipe").ok_or("model requires an explicit public schedule")?;
        let recipe=Self::from_request(value)?;
        if model["steps"].as_array().map(Vec::len)!=Some(recipe.nfe) {
            return Err("model step tables disagree with its evaluation count".into());
        }
        Ok(recipe)
    }
    pub fn partition(name:&str)->Option<&'static str>{ match name {
        "fl2va"=>Some("fl2va"), "ref2va"=>Some("ref2va"), _=>None,
    }}
}
#[cfg(test)] mod tests {
    use super::*;
    use serde_json::json;
    #[test] fn base_schedule_has_forty_nine_evaluations_for_fifty_points() {
        assert_eq!(Recipe::from_request(&json!({})).unwrap().nfe,49);
        for workflow in ["fl2va","ref2va"] {
            let model=json!({"recipe":{"workflow":workflow,"nfe":49},"steps":vec![0;49]});
            assert_eq!(Recipe::from_model(&model).unwrap().nfe,49);
            assert!(Recipe::from_model(&json!({"recipe":{"workflow":workflow,"nfe":49},"steps":vec![0;50]})).is_err());
        }
        for value in [json!(0),json!(-1),json!(true),json!(1.5),json!(1000)] {
            assert!(Recipe::from_request(&json!({"nfe":value})).is_err());
        }
        assert!(Recipe::from_request(&json!({"workflow":"unknown"})).is_err());
        assert!(Recipe::from_model(&json!({"steps":vec![0;49]})).is_err());
    }
}
