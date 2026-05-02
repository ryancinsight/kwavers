//! Modified Born Series for Viscoacoustic Media
//!
//! Extends the standard Born approximation to include absorption and dispersion
//! effects from viscosity and thermal conduction.
//!
//! ## Mathematical Foundation
//!
//! The viscoacoustic wave equation includes absorption:
//! ```text
//! ∇²p - (1/c₀²)∂²p/∂t² + (δ/c₀⁴)∂³p/∂t³ = -β ∂²(p²)/∂t² + S
//! ```
//!
//! ## References
//!
//! Sun, Y., et al. (2025). "A viscoacoustic wave equation solver using modified Born series"

use crate::domain::grid::Grid;
use ndarray::Array3;
use num_complex::Complex64;

mod construction;
mod differential;
mod green;
mod solve;
#[cfg(test)]
mod tests;

/// Modified Born series solver for viscoacoustic media
#[derive(Debug)]
pub struct ModifiedBornSolver {
    config: super::BornConfig,
    grid: Grid,
    workspace: super::BornWorkspace,
    absorption_field: Array3<Complex64>,
    diffusivity_field: Array3<f64>,
}

/// Statistics from modified Born series solution
#[derive(Debug, Clone, Default)]
pub struct ModifiedBornStats {
    pub orders_computed: usize,
    pub final_residual: f64,
    pub converged: bool,
}
