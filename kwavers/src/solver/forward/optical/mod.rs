//! Optical Wave and Transport Solvers
//!
//! This module contains solvers for optical wave propagation and photon
//! transport in biological tissues.

pub mod diffusion;

pub use diffusion::{DiffusionSolver, DiffusionSolverConfig};
