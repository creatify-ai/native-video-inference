//! Numerical identity is part of the immutable model manifest, never caller JSON.
//! A numerical profile does not establish equivalence to the reference engine.
use serde_json::Value;
use crate::Result;
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Numerics { Eager, SglangBf16, ProductionFused }
impl Numerics {
    pub fn from_model(model:&Value)->Result<Self>{
        match model.get("numerics") {
            None=>Ok(Self::SglangBf16),
            Some(Value::String(s)) if s=="eager-bf16-v1"=>Ok(Self::Eager),
            Some(Value::String(s)) if s=="sglang-bf16-cudnn-v1"=>Ok(Self::SglangBf16),
            Some(Value::String(s)) if s=="native-fused-bf16-v1"=>Ok(Self::ProductionFused),
            _=>Err("unsupported model numerical identity".into()),
        }
    }
    pub fn fused(self)->bool{self==Self::ProductionFused}
    pub fn sglang(self)->bool{self==Self::SglangBf16}
    pub fn rope_bytes(self)->usize{if self.fused(){4}else{2}}
    pub fn name(self)->&'static str{match self {
        Self::ProductionFused=>"native-fused-bf16-v1",Self::SglangBf16=>"sglang-bf16-cudnn-v1",Self::Eager=>"eager-bf16-v1"}}
}
#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    #[test]
    fn identity_is_explicit_and_fail_closed(){
        assert_eq!(Numerics::from_model(&json!({})).unwrap(),Numerics::SglangBf16);
        assert!(Numerics::from_model(&json!({"numerics":"sglang-bf16-cudnn-v1"})).unwrap().sglang());
        assert_eq!(Numerics::from_model(&json!({"numerics":"eager-bf16-v1"})).unwrap(),Numerics::Eager);
        assert_eq!(Numerics::from_model(&json!({"numerics":"native-fused-bf16-v1"})).unwrap().rope_bytes(),4);
        for value in [json!(null),json!(true),json!("fused"),json!({})]{
            assert!(Numerics::from_model(&json!({"numerics":value})).is_err());
        }
    }
}
