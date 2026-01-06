//! Spectral operators for pseudospectral methods
//!
//! Implements k-space differentiation and filtering operators for the
//! generalized spectral solver.
//!
//! References:
//! - Treeby & Cox (2010) "k-Wave: MATLAB toolbox" J. Biomed. Opt. 15(2)
//! - Liu, Q. H. (1997) "The PSTD algorithm" MOTL 15(3)

use crate::error::KwaversResult;
use crate::fft::KSpaceCalculator;
use crate::grid::Grid;
use crate::medium::Medium;
use ndarray::Array3;

use super::super::config::SpectralConfig;
use crate::solver::spectral_correction::CorrectionMethod;

/// Spectral operator collection for k-space methods
#[derive(Debug, Clone)]
pub struct SpectralOperators {
    pub kx: Array3<f64>,
    pub ky: Array3<f64>,
    pub kz: Array3<f64>,
    pub filter: Option<Array3<f64>>,
    pub k_max: f64,
}

use crate::solver::spectral::utils::{
    compute_anti_aliasing_filter, compute_kspace_correction_factors, compute_wavenumbers,
    CorrectionType,
};

/// Initialize spectral operators and correction factors
pub fn initialize_spectral_operators(
    config: &SpectralConfig,
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

    let kappa = if config.spectral_correction.enabled {
        let correction_type = match config.spectral_correction.method {
            CorrectionMethod::ExactDispersion
            | CorrectionMethod::KWave
            | CorrectionMethod::LowDispersionPSTD => CorrectionType::Treeby2010,
            CorrectionMethod::LiuPSTD | CorrectionMethod::SincSpatial => CorrectionType::Liu1997,
        };

        compute_kspace_correction_factors(
            &k_ops.kx,
            &k_ops.ky,
            &k_ops.kz,
            grid,
            correction_type,
            config.dt,
            c_ref,
        )
    } else {
        Array3::from_elem((grid.nx, grid.ny, grid.nz), 1.0)
    };

    Ok((k_ops, kappa, k_max, c_ref))
}

/// Compute k-space operators for spectral differentiation
pub fn compute_k_operators(grid: &Grid) -> (SpectralOperators, f64) {
    let (kx, ky, kz) = compute_wavenumbers(grid);

    // Maximum k-vector magnitude
    let k_max = KSpaceCalculator::max_k_stable(grid);

    let operators = SpectralOperators {
        kx,
        ky,
        kz,
        filter: None,
        k_max,
    };
    (operators, k_max)
}
