//! Job management system for PINN training operations.
//!
//! Provides asynchronous job queuing, execution, and monitoring for PINN training tasks.

pub mod manager;
#[cfg(test)]
mod tests;
pub mod types;

pub use manager::JobManager;
pub use types::{TrainingExecutor, TrainingFuture, TrainingJob, TrainingOutput, TrainingResult};
