//! The checkpoint's `config.json`, as SGLang reads it for `DeepseekV4ForCausalLM`.
use serde::Deserialize;

#[derive(Clone, Debug, Deserialize)]
pub struct RopeScaling {
    pub beta_fast: f64,
    pub beta_slow: f64,
    pub factor: f64,
    pub original_max_position_embeddings: u64,
    #[serde(rename = "type")]
    pub kind: String,
}

#[derive(Clone, Debug, Deserialize)]
pub struct Config {
    pub architectures: Vec<String>,
    pub bos_token_id: u32,
    pub eos_token_id: u32,
    pub compress_ratios: Vec<u32>,
    pub compress_rope_theta: f64,
    pub expert_dtype: String,
    pub hc_eps: f64,
    pub hc_mult: usize,
    pub hc_sinkhorn_iters: usize,
    pub head_dim: usize,
    pub hidden_size: usize,
    pub index_head_dim: usize,
    pub index_n_heads: usize,
    pub index_topk: usize,
    pub max_position_embeddings: u64,
    pub moe_intermediate_size: usize,
    pub n_routed_experts: usize,
    pub n_shared_experts: usize,
    pub norm_topk_prob: bool,
    pub num_attention_heads: usize,
    pub num_experts_per_tok: usize,
    pub num_hash_layers: usize,
    pub num_hidden_layers: usize,
    pub num_key_value_heads: usize,
    pub num_nextn_predict_layers: usize,
    pub o_groups: usize,
    pub o_lora_rank: usize,
    pub q_lora_rank: usize,
    pub qk_rope_head_dim: usize,
    pub rms_norm_eps: f64,
    pub rope_scaling: RopeScaling,
    pub rope_theta: f64,
    pub routed_scaling_factor: f64,
    pub scoring_func: String,
    pub sliding_window: usize,
    pub swiglu_limit: f64,
    pub topk_method: String,
    pub vocab_size: usize,
}

impl Config {
    pub fn parse(text: &str) -> crate::Result<Self> {
        let config: Config = serde_json::from_str(text).map_err(|e| format!("config.json: {e}"))?;
        config.validate()?;
        Ok(config)
    }

    /// The configuration this port implements; anything else is rejected rather than run.
    fn validate(&self) -> crate::Result<()> {
        let expect = |ok: bool, what: &str| if ok { Ok(()) } else { Err(format!("unsupported config: {what}")) };
        expect(self.architectures == ["DeepseekV4ForCausalLM"], "architecture")?;
        expect(self.compress_ratios.len() == self.num_hidden_layers + self.num_nextn_predict_layers, "compress_ratios length")?;
        expect(self.compress_ratios.iter().all(|r| matches!(r, 0 | 4 | 128)), "compress ratio other than 0/4/128")?;
        expect(self.num_key_value_heads == 1, "num_key_value_heads != 1")?;
        expect(self.n_shared_experts == 1, "n_shared_experts != 1")?;
        expect(self.scoring_func == "sqrtsoftplus", "scoring_func")?;
        expect(self.expert_dtype == "fp4", "expert_dtype")?;
        expect(self.rope_scaling.kind == "yarn", "rope_scaling type")?;
        expect(self.head_dim > self.qk_rope_head_dim, "head_dim")?;
        Ok(())
    }

    pub fn compress_ratio(&self, layer: usize) -> u32 { self.compress_ratios[layer] }
    pub fn is_hash_layer(&self, layer: usize) -> bool { layer < self.num_hash_layers }
    pub fn nope_head_dim(&self) -> usize { self.head_dim - self.qk_rope_head_dim }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CONFIG: &str = r#"{"architectures":["DeepseekV4ForCausalLM"],"attention_bias":false,"bos_token_id":0,
      "compress_ratios":[0,0,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,0],
      "compress_rope_theta":160000,"eos_token_id":1,"expert_dtype":"fp4","hc_eps":1e-06,"hc_mult":4,"hc_sinkhorn_iters":20,
      "head_dim":512,"hidden_act":"silu","hidden_size":4096,"index_head_dim":128,"index_n_heads":64,"index_topk":512,
      "max_position_embeddings":1048576,"model_type":"deepseek_v4","moe_intermediate_size":2048,"n_routed_experts":256,
      "n_shared_experts":1,"norm_topk_prob":true,"num_attention_heads":64,"num_experts_per_tok":6,"num_hash_layers":3,
      "num_hidden_layers":43,"num_key_value_heads":1,"num_nextn_predict_layers":1,"o_groups":8,"o_lora_rank":1024,
      "q_lora_rank":1024,"qk_rope_head_dim":64,"rms_norm_eps":1e-06,
      "rope_scaling":{"beta_fast":32,"beta_slow":1,"factor":16,"original_max_position_embeddings":65536,"type":"yarn"},
      "rope_theta":10000,"routed_scaling_factor":1.5,"scoring_func":"sqrtsoftplus","sliding_window":128,"swiglu_limit":10.0,
      "tie_word_embeddings":false,"topk_method":"noaux_tc","torch_dtype":"bfloat16","vocab_size":129280}"#;

    #[test]
    fn parses_the_pinned_checkpoint_config() {
        let c = Config::parse(CONFIG).unwrap();
        assert_eq!(c.num_hidden_layers, 43);
        assert_eq!(c.compress_ratio(2), 4);
        assert_eq!(c.compress_ratio(3), 128);
        assert!(c.is_hash_layer(2) && !c.is_hash_layer(3));
        assert_eq!(c.nope_head_dim(), 448);
    }
}
