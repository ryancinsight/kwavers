//! Adaptive sampling and mini-batching for efficient PINN training.
//!
//! Concentrates collocation points where residuals are high:
//! `p_i ∝ r_i^α / Σ r_j^α`.

mod batch;
mod sampler;
#[cfg(all(test, feature = "pinn"))]
mod tests;

pub use batch::{extract_batch, BatchIterator};
pub use sampler::{AdaptiveSampler, ElasticAdaptiveSamplingStrategy};
