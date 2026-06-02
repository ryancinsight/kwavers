//! GPU-Accelerated Electromagnetic Physics Solver
//!
//! This module provides high-performance GPU implementations for electromagnetic
//! field simulations using Maxwell's equations. It integrates with the PINN
//! framework to provide physics-informed neural network training with GPU acceleration.
//!
//! ## Features
//!
//! - **FDTD Solver**: Finite Difference Time Domain implementation of Maxwell's equations
//! - **GPU Acceleration**: WGSL compute shaders for parallel field updates
//! - **PINN Integration**: Physics-informed loss functions with GPU-accelerated residuals
//! - **Boundary Conditions**: PEC, PMC, Absorbing boundary conditions
//! - **Multi-GPU Support**: Distributed electromagnetic simulations
//!
//! ## Usage

mod accessors;
pub mod compute;
pub mod config;
mod construction;
pub mod fields;
pub mod solver;
mod stepping;
#[cfg(test)]
mod tests;

pub use config::{EMConfig, ElectromagneticBc};
pub use fields::EMFieldData;
pub use solver::GPUEMSolver;
