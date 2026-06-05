//! Dispersion Analysis and Correction for Numerical Methods
//!
//! Provides tools for analyzing and correcting numerical dispersion
//! in FDTD and PSTD wave propagation solvers.
//!
//! ## Mathematical Foundation
//!
//! Relative dispersion error:
//! ```text
//! ε = (ω_numerical - ω_exact) / ω_exact
//! ```

pub mod correction;
pub mod fdtd;
pub mod pstd;

#[cfg(test)]
mod tests;

// Re-export the main analysis struct and method enum
pub use correction::DispersionMethod;

/// Dispersion analysis for numerical methods
#[derive(Debug)]
pub struct DispersionAnalysis;
