//! Generalized Spectral Solver Implementation
//!
//! Main solver implementation following GRASP principles.

pub mod config;
pub mod data;
pub mod dg;
pub mod implementation;
pub mod numerics;
pub mod physics;
pub mod plugin;
pub mod propagator;
pub mod utils;

pub use config::PSTDConfig;
pub use implementation::core::orchestrator::PSTDSolver;
pub use plugin::PSTDPlugin;
