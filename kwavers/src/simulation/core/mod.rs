//! Simulation control module
//!
//! This module provides the core simulation control functionality that
//! orchestrates solvers, sources, sensor probe sets, and medium to run complete simulations.
//! This module supports building a simulation orchestration layer.
//!
//! # Sensor model
//! Kwavers supports multi-physics (acoustics + optics). The canonical high-level
//! sensor representation is therefore a grid probe set (`GridSensorSet`) rather than
//! a domain-specific array sensor.

mod builder;
mod controller;
mod types;

#[cfg(test)]
mod tests;

pub use builder::SimulationBuilder;
pub use controller::CoreSimulation;
pub use types::{CoreSimulationStatistics, SimulationResult};
