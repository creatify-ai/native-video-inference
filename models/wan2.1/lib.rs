//! Official Wan 2.1 I2V-14B inference in native Rust/CUDA, bitwise identical to the PyTorch
//! reference (Wan-Video/Wan2.1 9737cba run on B200, FlashAttention swapped for cuDNN SDPA; see
//! README.md). Built from the core_infra crate (cuBLAS/cuDNN bridges, loading, serving) plus the
//! operators below, each checked against recorded fixtures before it is composed further.
pub mod gold;
pub mod weights;
pub mod schedules;
pub mod frontend;
pub mod admission;
pub mod preload_contract;
mod html5_entities;
#[cfg(feature = "cuda")]
pub mod cuda;
#[cfg(feature = "cuda")]
pub mod linear;
#[cfg(feature = "cuda")]
pub mod conv;
#[cfg(feature = "cuda")]
pub mod sdpa;
#[cfg(feature = "cuda")]
pub mod eager;
#[cfg(feature = "cuda")]
pub mod rng;
#[cfg(feature = "cuda")]
pub mod dit;
#[cfg(feature = "cuda")]
pub mod sampler;
#[cfg(feature = "cuda")]
pub mod encoders;
#[cfg(feature = "cuda")]
pub mod t5;
#[cfg(feature = "cuda")]
pub mod clip;
#[cfg(feature = "cuda")]
pub mod vae;
#[cfg(feature = "cuda")]
pub mod pipeline;
#[cfg(feature = "cuda")]
mod startup;
pub mod serving;
#[cfg(feature = "cuda")]
mod cfg_parallel;
#[cfg(feature = "cuda")]
mod backend;

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
