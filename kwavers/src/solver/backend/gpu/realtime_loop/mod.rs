//! GPU Real-Time Simulation Loop Orchestrator
//!
//! Coordinates GPU-accelerated multiphysics timesteps with real-time budget
//! enforcement, performance monitoring, and async I/O for checkpoints.

mod orchestrator;
#[cfg(test)]
mod tests;
mod types;

pub use orchestrator::RealtimeSimulationOrchestrator;
pub use types::{GpuRealtimeSimulationStatistics, RealtimeConfig, StepResult};
