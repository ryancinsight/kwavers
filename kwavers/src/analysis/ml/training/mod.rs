//! ML Training Loop with Burn Autodiff
//!
//! Provides a comprehensive training pipeline for neural beamforming models
//! using the Burn deep learning framework.
//!
//! ## References
//!
//! - Kingma & Ba (2015) "Adam: A Method for Stochastic Optimization"
//! - Raissi et al. (2019) "Physics-informed neural networks"
//! - Goodfellow et al. (2016) "Deep Learning" (Chapters 8-9)

pub mod config;
pub mod dataset;
pub mod history;
pub mod loss;
#[cfg(test)]
mod tests;

pub use config::TrainingConfig;
pub use dataset::{TrainingDataset, TrainingMetrics};
pub use history::TrainingHistory;
pub use loss::{Optimizer, PhysicsLoss};
