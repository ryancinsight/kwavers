//! GPU Real-Time Simulation Loop Orchestrator
//!
//! Coordinates GPU-accelerated multiphysics timesteps with real-time budget
//! enforcement, performance monitoring, and async I/O for checkpoints.

mod orchestrator;
mod types;
#[cfg(test)]
mod tests;

pub use orchestrator::RealtimeSimulationOrchestrator;
pub use types::{RealtimeConfig, SimulationStatistics, StepResult};
