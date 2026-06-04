//! Spectral operators for pseudospectral methods
//!
//! Implements k-space differentiation and filtering operators for the
//! generalized spectral solver.
//!
//! References:
//! - Treeby & Cox (2010) J. Biomed. Opt. 15(2)
//! - Liu, Q. H. (1997) "The PSTD algorithm" MOTL 15(3)

use kwavers_core::error::KwaversResult;
use kwavers_grid::Grid;
use kwavers_domain::medium::Medium;
use kwavers_math::fft::KSpaceCalculator;
use ndarray::Array3;

use crate::forward::pstd::config::PSTDConfig;
use crate::forward::pstd::numerics::spectral_correction::{
    compute_spectral_correction, SpectralCorrectionConfig,
};

/// Spectral operator collection for k-space methods
#[derive(Debug, Clone)]
pub struct SpectralOperators {
    pub kx: Array3<f64>,
    pub ky: Array3<f64>,
    pub kz: Array3<f64>,
    pub filter: Option<Array3<f64>>,
    pub k_max: f64,
}

use crate::pstd::utils::{compute_anti_aliasing_filter, compute_wavenumbers};

/// Initialize spectral operators and correction factors.
///
/// Kappa is computed via [`compute_spectral_correction`] — the single canonical
/// dispatch path in `numerics::spectral_correction`. This ensures every
/// [`SpectralCorrectionMethod`] variant reaches its distinct mathematical
/// formula and the dispatch is non-lossy.
///
/// # Errors
/// - Returns [`Err`] if an internal constraint is violated.
///
pub fn initialize_spectral_operators(
    config: &PSTDConfig,
    grid: &Grid,
    medium: &dyn Medium,
) -> KwaversResult<(SpectralOperators, Array3<f64>, f64, f64)> {
    let (mut k_ops, k_max) = compute_k_operators(grid);
    let c_ref = medium.max_sound_speed();

    // Initialize anti-aliasing filter if enabled
    if config.anti_aliasing.enabled {
        k_ops.filter = Some(compute_anti_aliasing_filter(
            grid,
            config.anti_aliasing.cutoff,
            config.anti_aliasing.order,
        ));
    }

    // Route through the unified spectral correction dispatch so every
    // SpectralCorrectionMethod variant reaches its distinct formula.
    let correction_config = SpectralCorrectionConfig {
        enabled: config.spectral_correction.enabled,
        method: config.spectral_correction.method,
        cfl_number: config.spectral_correction.cfl_number,
        max_correction: config.spectral_correction.max_correction,
    };
    let kappa = compute_spectral_correction(grid, &correction_config, config.dt, c_ref);

    Ok((k_ops, kappa, k_max, c_ref))
}

/// Compute k-space operators for spectral differentiation
pub fn compute_k_operators(grid: &Grid) -> (SpectralOperators, f64) {
    let (kx, ky, kz) = compute_wavenumbers(grid);

    // Maximum k-vector magnitude
    let k_max = KSpaceCalculator::max_k_stable(grid.dx, grid.dy, grid.dz);

    let operators = SpectralOperators {
        kx,
        ky,
        kz,
        filter: None,
        k_max,
    };
    (operators, k_max)
}
