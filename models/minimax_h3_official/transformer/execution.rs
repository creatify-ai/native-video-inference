//! Official base schedules with the pinned SGLang BF16 SubBlock defaults.
use crate::{recipe::Recipe, Result};
use serde_json::{json, Value};

pub const MAIN_LAYERS:usize=50;
pub const DENSE_PREFIX:usize=10;
const SPARSE_MIN_SEQUENCE:usize=4096;
#[derive(Clone,Copy,Debug,PartialEq,Eq)]
pub struct Execution {pub start:usize,pub end:usize,pub dense_only:bool}
impl Execution {
    pub fn full(nfe:usize)->Self{Self{start:0,end:nfe,dense_only:false}}
    pub fn from_request(request:&Value)->Result<Self>{
        if request.get("denoise_phase").is_some() {
            return Err("public H3 executes the complete official base schedule".into());
        }
        let mut plan=Self::full(Recipe::from_request(request)?.nfe);
        plan.dense_only=match request.get("attention_backend") {
            None=>false,
            Some(Value::String(name)) if name=="subblock_sparse_attn"=>false,
            Some(Value::String(name)) if name=="torch_cudnn_sdpa"=>true,
            _=>return Err("public H3 attention_backend must be subblock_sparse_attn or torch_cudnn_sdpa".into()),
        };
        Ok(plan)
    }
    pub fn evaluations(self)->usize{self.end-self.start}
    pub fn attention_index(self,index:usize)->Result<usize>{
        if !(self.start..self.end).contains(&index){return Err("schedule index outside request".into());}
        Ok(index-self.start)
    }
    pub fn sparse(self,index:usize,sequence:usize)->Result<bool>{
        Ok(self.attention_index(index)? >= DENSE_PREFIX && !self.dense_only && sequence >= SPARSE_MIN_SEQUENCE)
    }
    pub fn sparse_layer(self,index:usize,sequence:usize,layer:usize)->Result<bool>{
        if layer>=MAIN_LAYERS{return Err("layer outside H3 main blocks".into());}
        self.sparse(index,sequence)
    }
    pub fn sparse_layers(self,index:usize,sequence:usize)->Result<usize>{
        Ok(if self.sparse(index,sequence)?{MAIN_LAYERS}else{0})
    }
    pub fn sparse_calls_expected(self,sequence:usize,head_groups:usize)->Result<u64>{
        if head_groups==0{return Err("head groups must be nonzero".into());}
        let layers=(self.start..self.end).try_fold(0u64,|sum,index|->Result<u64>{Ok(sum+self.sparse_layers(index,sequence)? as u64)})?;
        layers.checked_mul(head_groups as u64).ok_or_else(||"sparse dispatch count overflow".into())
    }
    pub fn json(self)->Value{json!({"start":self.start,"end":self.end,"evaluations":self.evaluations(),
        "attention_backend":if self.dense_only{"torch_cudnn_sdpa"}else{"subblock_sparse_attn"},
        "dense_prefix_evaluations":if self.dense_only{self.evaluations()}else{DENSE_PREFIX},"main_layers":MAIN_LAYERS,
        "sparse_min_seq_len":SPARSE_MIN_SEQUENCE,"dense_layers":[]})}
}
#[cfg(test)] mod tests {
    use super::*;
    #[test] fn cudnn_executes_the_full_schedule_without_sparse_calls(){
        let plan=Execution::from_request(&json!({"nfe":49,"attention_backend":"torch_cudnn_sdpa"})).unwrap();
        assert_eq!(plan.evaluations(),49);
        for step in 0..49 { for layer in 0..50 {
            assert!(!plan.sparse_layer(step,21743,layer).unwrap());
        }}
        assert_eq!(plan.sparse_calls_expected(21743,1).unwrap(),0);
        assert!(plan.sparse_layer(49,21743,0).is_err());
        for value in [json!("fa"),json!(null),json!(true)] {
            assert!(Execution::from_request(&json!({"attention_backend":value})).is_err());
        }
    }
    #[test] fn matches_the_pinned_reference_schedule_boundaries(){
        for workflow in ["fl2va","ref2va"] {
            let plan=Execution::from_request(&json!({"workflow":workflow,"nfe":49})).unwrap();
            for index in 0..49 {for layer in 0..50 {
                assert_eq!(plan.sparse_layer(index,4096,layer).unwrap(),index>=10);
                assert!(!plan.sparse_layer(index,4095,layer).unwrap());
            }}
            assert_eq!(plan.sparse_calls_expected(4096,1).unwrap(),39*50);
            assert_eq!(plan.sparse_calls_expected(4096,8).unwrap(),39*50*8);
            assert!(plan.sparse_layer(49,4096,0).is_err());
            assert!(plan.sparse_layer(10,4096,50).is_err());
            assert!(plan.sparse_calls_expected(4096,0).is_err());
        }
    }
}
