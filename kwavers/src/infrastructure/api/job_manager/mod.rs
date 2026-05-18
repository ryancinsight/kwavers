//! Job management system for PINN training operations.
//!
//! Provides asynchronous job queuing, execution, and monitoring for PINN training tasks.

pub mod manager;
#[cfg(test)]
mod tests;
pub mod types;

pub use manager::JobManager;
pub use types::{
    JobManagerTrainingResult, TrainingExecutor, TrainingFuture, TrainingJob, TrainingOutput,
};
