//! Westervelt equation solver for nonlinear acoustic wave propagation
//!
//! This module implements the Westervelt equation with proper second-order time derivatives
//! and full nonlinear term implementation.

mod metrics;
mod nonlinear;
mod solver;
mod spectral;

pub use metrics::PerformanceMetrics;
pub use solver::WesterveltWave;

// Re-export key functions for compatibility
pub use nonlinear::compute_nonlinear_term;
pub use spectral::{apply_kspace_correction, compute_laplacian_spectral};
