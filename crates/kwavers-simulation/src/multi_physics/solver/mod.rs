//! Multi-physics simulation orchestrator.

pub mod core;
#[cfg(test)]
mod tests;

pub use core::SimulationMultiPhysicsSolver;
