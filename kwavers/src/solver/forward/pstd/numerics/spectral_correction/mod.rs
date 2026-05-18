//! Unified spectral correction module for spectral methods
//!
//! This module provides a theoretically correct spectral correction that accounts
//! for both spatial and temporal discretization errors in pseudospectral methods.
//!
//! # Theory
//!
//! The spectral pseudospectral method introduces numerical dispersion due to:
//! 1. Spatial discretization: The finite difference approximation of spatial derivatives
//! 2. Temporal discretization: The finite difference approximation of time derivatives
//!
//! The correct approach is to modify the wavenumber k such that the numerical
//! dispersion relation matches the true physical relation ω = ck.
//!
//! For the PSTD method with leapfrog time stepping:
//! - Numerical dispersion: sin(ωΔt/2) = (cΔt/2) * |`k_mod`|
//! - Physical dispersion: ω = c|k|
//!
//! The correction factor κ is derived to ensure the numerical scheme propagates
//! waves at the correct phase velocity.
//!
//! # References
//!
//! - Liu, Q. H. (1997). "The PSTD algorithm: A time-domain method requiring only
//!   two cells per wavelength." Microwave and Optical Technology Letters, 15(3), 158-165.
//! - Treeby, B. E., & Cox, B. T. (2010). "MATLAB toolbox for the simulation and
//!   reconstruction of photoacoustic wave fields." Journal of Biomedical Optics, 15(2).

use crate::domain::grid::Grid;
use ndarray::Array3;
use serde::{Deserialize, Serialize};

mod corrections;
mod diagnostics;

#[cfg(test)]
mod tests;

pub use corrections::apply_correction;
pub use diagnostics::{compute_dispersion_error, compute_numerical_phase_velocity};

/// Spectral correction configuration
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct SpectralCorrectionConfig {
    /// Enable spectral correction
    pub enabled: bool,
    /// Correction method
    pub method: SpectralCorrectionMethod,
    /// CFL number for stability
    pub cfl_number: f64,
    /// Maximum correction factor (for stability)
    pub max_correction: f64,
}

/// Correction method selection
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum SpectralCorrectionMethod {
    /// Exact dispersion correction (most accurate)
    ExactDispersion,
    /// Treeby & Cox (2010) methodology
    Treeby2010,
    /// Liu's PSTD correction (Liu 1997)
    LiuPSTD,
    /// Low-dispersion correction (for PSTD)
    LowDispersionPSTD,
    /// Sinc correction (spatial only)
    SincSpatial,
}

impl Default for SpectralCorrectionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            // Treeby & Cox 2010 canonical kappa = sinc(c_ref·dt·|k|/2),
            // matching k-wave-python's `kspace_solver.py` default.
            method: SpectralCorrectionMethod::Treeby2010,
            cfl_number: 0.3,
            max_correction: 2.0,
        }
    }
}

/// Compute spectral correction factors with proper dispersion correction
///
/// This implements the theoretically correct spectral correction that accounts
/// for both spatial and temporal discretization errors.
///
/// # Arguments
/// * `grid` - Computational grid
/// * `config` - Spectral correction configuration
/// * `dt` - Time step
/// * `c_ref` - Reference sound speed
///
/// # Returns
/// Array of spectral correction factors (κ)
pub fn compute_spectral_correction(
    grid: &Grid,
    config: &SpectralCorrectionConfig,
    dt: f64,
    c_ref: f64,
) -> Array3<f64> {
    if !config.enabled {
        return Array3::from_elem((grid.nx, grid.ny, grid.nz), 1.0);
    }

    corrections::compute_spectral_correction_dispatch(
        grid,
        config.method,
        dt,
        c_ref,
        config.cfl_number,
        config.max_correction,
    )
}
