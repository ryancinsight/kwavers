//! Model Order Selection for Source Localization
//!
//! Implements information-theoretic criteria (AIC, MDL) for automatic estimation
//! of the number of signal sources from sensor array covariance matrices.
//!
//! # References
//!
//! - Wax, M., & Kailath, T. (1985). "Detection of signals by information theoretic criteria"
//!   IEEE Trans. Acoust., Speech, Signal Process., 33(2), 387-392.

mod estimator;
mod types;

#[cfg(test)]
mod tests;

pub use estimator::ModelOrderEstimator;
pub use types::{ModelOrderConfig, ModelOrderCriterion, ModelOrderResult};
