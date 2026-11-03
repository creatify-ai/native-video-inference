//! LTX 2.5 distilled BF16 native inference from prepared profiles on B200.
//! Model code and CUDA calls live here; the shared core remains model independent.
pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
pub mod gold;
pub mod weights;
pub mod frontend;
mod admission;
#[cfg(feature = "cuda")]
pub mod cuda;
#[cfg(feature = "cuda")]
pub mod linear;
#[cfg(feature = "cuda")]
pub mod rng;
#[cfg(feature = "cuda")]
pub mod ops;
#[cfg(feature = "cuda")]
pub mod sdpa;
#[cfg(feature = "cuda")]
pub mod tensor;
#[cfg(feature = "cuda")]
pub mod graph;
#[cfg(feature = "cuda")]
pub mod backbone;
#[cfg(feature = "cuda")]
pub mod parallel;
#[cfg(feature = "cuda")]
pub mod forward;
#[cfg(feature = "cuda")]
pub mod sampling;
#[cfg(feature = "cuda")]
pub mod convolution;
#[cfg(feature = "cuda")]
pub mod text;
#[cfg(feature = "cuda")]
pub mod triton;
#[cfg(feature = "cuda")]
pub mod natten;
#[cfg(feature = "cuda")]
pub mod request;
#[cfg(feature = "cuda")]
mod startup;
#[cfg(feature = "cuda")]
mod declared;
#[cfg(feature = "cuda")]
pub mod media;
#[cfg(feature = "cuda")]
pub mod decode;
#[cfg(feature="backend")]
pub mod backend;
