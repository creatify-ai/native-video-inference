//! DeepSeek-V4-Flash (NVFP4 experts) on one B300, ported from SGLang byte for byte.
//!
//! The reference is SGLang v0.5.23 serving `nvidia/DeepSeek-V4-Flash-NVFP4` with TP=1; see
//! `reference/` for the pinned environment, the gold capture and the CUPTI launch tape.
pub mod config;
#[cfg(feature = "cuda")]
pub mod compress;
#[cfg(feature = "cuda")]
pub mod cublaslt;
pub mod dense_prepare;
#[cfg(feature = "cuda")]
pub mod cuda;
#[cfg(feature = "cuda")]
pub mod deepgemm;
#[cfg(feature = "cuda")]
pub mod exact;
#[cfg(feature = "cuda")]
pub mod flashmla;
pub mod gold;
pub mod logprobs;
#[cfg(feature = "cuda")]
pub mod mhc;
#[cfg(feature = "cuda")]
pub mod model;
#[cfg(feature = "cuda")]
pub mod moe;
pub mod moe_prepare;
#[cfg(feature = "cuda")]
pub mod parity;
pub mod prepare;
#[cfg(feature = "cuda")]
pub mod prepared;
#[cfg(feature = "cuda")]
pub mod runtime;
pub mod safetensors;
#[cfg(feature = "cuda")]
pub mod snapshots;
#[cfg(feature = "cuda")]
pub mod sgl;
pub mod tape;
#[cfg(feature = "cuda")]
pub mod topk;
pub mod tokenize;
#[cfg(feature = "cuda")]
pub mod torch_ops;
pub mod weights;

pub type Result<T> = std::result::Result<T, String>;
