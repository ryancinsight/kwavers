//! Discontinuous Galerkin solver components
//!
//! This module provides a modular DG solver implementation with separated concerns.

pub mod config;
pub mod operations;
pub mod solver;

pub use config::DGConfig;
pub use operations::DGOperations;
pub use solver::DGSolver;