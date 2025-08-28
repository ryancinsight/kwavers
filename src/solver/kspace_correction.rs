//! Unified k-space correction module for spectral methods
//!
//! This module provides a theoretically correct k-space correction that accounts
//! for both spatial and temporal discretization errors in pseudospectral methods.
//!
//! # Theory
//!
//! The k-space pseudospectral method introduces numerical dispersion due to:
//! 1. Spatial discretization: The finite difference approximation of spatial derivatives
//! 2. Temporal discretization: The finite difference approximation of time derivatives
//!
//! The correct approach is to modify the wavenumber k such that the numerical
//! dispersion relation matches the true physical relation ω = ck.
//!
//! For the PSTD method with leapfrog time stepping:
//! - Numerical dispersion: sin(ωΔt/2) = (cΔt/2) * |k_mod|
//! - Physical dispersion: ω = c|k|
//!
//! The correction factor κ is derived to ensure the numerical scheme propagates
//! waves at the correct phase velocity.
//!
//! # References
//!
//! - Liu, Q. H. (1997). "The PSTD algorithm: A time-domain method requiring only
//!   two cells per wavelength." Microwave and Optical Technology Letters, 15(3), 158-165.
//! - Treeby, B. E., & Cox, B. T. (2010). "k-Wave: MATLAB toolbox for the simulation
//!   and reconstruction of photoacoustic wave fields." Journal of Biomedical Optics, 15(2).

use crate::grid::Grid;
use ndarray::{Array3, Zip};
use std::f64::consts::PI;

/// K-space correction configuration
#[derive(Debug, Clone, Copy)]
pub struct KSpaceCorrectionConfig {
    /// Enable k-space correction
    pub enabled: bool,
    /// Correction method
    pub method: CorrectionMethod,
    /// CFL number for stability
    pub cfl_number: f64,
    /// Maximum correction factor (for stability)
    pub max_correction: f64,
}

/// Correction method selection
#[derive(Debug, Clone, Copy)]
pub enum CorrectionMethod {
    /// Exact dispersion correction (most accurate)
    ExactDispersion,
    /// k-Wave methodology (Treeby & Cox 2010)
    KWave,
    /// Liu's PSTD correction (Liu 1997)
    LiuPSTD,
    /// Simple sinc correction (spatial only)
    SincSpatial,
}

impl Default for KSpaceCorrectionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            method: CorrectionMethod::ExactDispersion,
            cfl_number: 0.3,
            max_correction: 2.0,
        }
    }
}

/// Compute k-space correction factors with proper dispersion correction
///
/// This implements the theoretically correct k-space correction that accounts
/// for both spatial and temporal discretization errors.
///
/// # Arguments
/// * `grid` - Computational grid
/// * `config` - K-space correction configuration
/// * `dt` - Time step
/// * `c_ref` - Reference sound speed
///
/// # Returns
/// Array of k-space correction factors (κ)
pub fn compute_kspace_correction(
    grid: &Grid,
    config: &KSpaceCorrectionConfig,
    dt: f64,
    c_ref: f64,
) -> Array3<f64> {
    if !config.enabled {
        return Array3::from_elem((grid.nx, grid.ny, grid.nz), 1.0);
    }

    match config.method {
        CorrectionMethod::ExactDispersion => {
            compute_exact_dispersion_correction(grid, dt, c_ref, config.max_correction)
        }
        CorrectionMethod::KWave => {
            compute_kwave_correction(grid, dt, c_ref, config.cfl_number, config.max_correction)
        }
        CorrectionMethod::LiuPSTD => {
            compute_liu_pstd_correction(grid, dt, c_ref, config.max_correction)
        }
        CorrectionMethod::SincSpatial => compute_sinc_spatial_correction(grid),
    }
}

/// Compute exact dispersion correction
///
/// This derives the correction factor by matching the numerical dispersion
/// relation to the physical dispersion relation ω = c|k|.
fn compute_exact_dispersion_correction(
    grid: &Grid,
    dt: f64,
    c_ref: f64,
    max_correction: f64,
) -> Array3<f64> {
    let mut kappa = Array3::from_elem((grid.nx, grid.ny, grid.nz), 1.0);

    for i in 0..grid.nx {
        for j in 0..grid.ny {
            for k in 0..grid.nz {
                // Compute physical wavenumber components
                let kx = compute_wavenumber_component(i, grid.nx, grid.dx);
                let ky = compute_wavenumber_component(j, grid.ny, grid.dy);
                let kz = compute_wavenumber_component(k, grid.nz, grid.dz);

                // Physical wavenumber magnitude
                let k_phys = (kx * kx + ky * ky + kz * kz).sqrt();

                if k_phys > 0.0 {
                    // Modified wavenumber components (accounting for finite differences)
                    let kx_mod = 2.0 * (kx * grid.dx / 2.0).sin() / grid.dx;
                    let ky_mod = 2.0 * (ky * grid.dy / 2.0).sin() / grid.dy;
                    let kz_mod = 2.0 * (kz * grid.dz / 2.0).sin() / grid.dz;

                    // Modified wavenumber magnitude
                    let k_mod = (kx_mod * kx_mod + ky_mod * ky_mod + kz_mod * kz_mod).sqrt();

                    if k_mod > 0.0 {
                        // Physical angular frequency
                        let omega_phys = c_ref * k_phys;

                        // Numerical angular frequency (from leapfrog scheme)
                        // sin(ω_num * dt/2) = (c * dt/2) * k_mod
                        let arg = c_ref * dt * k_mod / 2.0;

                        if arg < 1.0 {
                            // Ensure stability
                            let omega_num = 2.0 * arg.asin() / dt;

                            // Correction factor to match physical dispersion
                            let correction = omega_phys / omega_num;

                            // Apply correction with limiting for stability
                            kappa[[i, j, k]] =
                                correction.min(max_correction).max(1.0 / max_correction);
                        } else {
                            // Near Nyquist frequency - apply maximum damping
                            kappa[[i, j, k]] = 1.0 / max_correction;
                        }
                    }
                }
            }
        }
    }

    kappa
}

/// Compute k-Wave correction (Treeby & Cox 2010)
///
/// This implements the k-Wave methodology which combines spatial sinc
/// correction with temporal correction for the k-space method.
fn compute_kwave_correction(
    grid: &Grid,
    dt: f64,
    c_ref: f64,
    cfl: f64,
    max_correction: f64,
) -> Array3<f64> {
    let mut kappa = Array3::from_elem((grid.nx, grid.ny, grid.nz), 1.0);

    for i in 0..grid.nx {
        for j in 0..grid.ny {
            for k in 0..grid.nz {
                // Compute wavenumber components
                let kx = compute_wavenumber_component(i, grid.nx, grid.dx);
                let ky = compute_wavenumber_component(j, grid.ny, grid.dy);
                let kz = compute_wavenumber_component(k, grid.nz, grid.dz);

                // Modified wavenumbers (finite difference operators)
                let kx_mod = 2.0 * (kx * grid.dx / 2.0).sin() / grid.dx;
                let ky_mod = 2.0 * (ky * grid.dy / 2.0).sin() / grid.dy;
                let kz_mod = 2.0 * (kz * grid.dz / 2.0).sin() / grid.dz;

                let k_mod_sq = kx_mod * kx_mod + ky_mod * ky_mod + kz_mod * kz_mod;
                let k_phys_sq = kx * kx + ky * ky + kz * kz;

                if k_mod_sq > 0.0 && k_phys_sq > 0.0 {
                    // Spatial correction (sinc function)
                    let spatial_correction = (k_mod_sq / k_phys_sq).sqrt();

                    // Temporal correction for k-space method
                    let omega_dt = c_ref * dt * k_mod_sq.sqrt();
                    let temporal_correction = if omega_dt > 0.0 {
                        omega_dt / (2.0 * (omega_dt / 2.0).sin())
                    } else {
                        1.0
                    };

                    // Combined correction
                    let correction = spatial_correction * temporal_correction;

                    // Apply with stability limiting
                    kappa[[i, j, k]] = correction.min(max_correction).max(1.0 / max_correction);
                }
            }
        }
    }

    kappa
}

/// Compute Liu's PSTD correction (Liu 1997)
///
/// This implements the correction from Liu's original PSTD paper,
/// which focuses on maintaining accuracy with only 2 cells per wavelength.
fn compute_liu_pstd_correction(
    grid: &Grid,
    dt: f64,
    c_ref: f64,
    max_correction: f64,
) -> Array3<f64> {
    let mut kappa = Array3::from_elem((grid.nx, grid.ny, grid.nz), 1.0);

    // Liu's correction parameter
    let dx_min = grid.dx.min(grid.dy).min(grid.dz);
    let stability_factor = c_ref * dt / dx_min;

    for i in 0..grid.nx {
        for j in 0..grid.ny {
            for k in 0..grid.nz {
                // Compute wavenumber components
                let kx = compute_wavenumber_component(i, grid.nx, grid.dx);
                let ky = compute_wavenumber_component(j, grid.ny, grid.dy);
                let kz = compute_wavenumber_component(k, grid.nz, grid.dz);

                let k_mag = (kx * kx + ky * ky + kz * kz).sqrt();

                if k_mag > 0.0 {
                    // Liu's dispersion correction formula
                    let k_dx = k_mag * dx_min;

                    // Correction based on maintaining 2 points per wavelength accuracy
                    let correction = if k_dx < PI {
                        // Low frequency - minimal correction needed
                        1.0 + stability_factor * stability_factor * k_dx * k_dx / 24.0
                    } else {
                        // High frequency - stronger correction
                        let sinc = (k_dx / 2.0).sin() / (k_dx / 2.0);
                        1.0 / sinc
                    };

                    kappa[[i, j, k]] = correction.min(max_correction).max(1.0 / max_correction);
                }
            }
        }
    }

    kappa
}

/// Compute simple spatial sinc correction
///
/// This is the simplest correction that only accounts for spatial discretization.
/// It does not account for temporal errors and is included for comparison.
fn compute_sinc_spatial_correction(grid: &Grid) -> Array3<f64> {
    let mut kappa = Array3::from_elem((grid.nx, grid.ny, grid.nz), 1.0);

    for i in 0..grid.nx {
        for j in 0..grid.ny {
            for k in 0..grid.nz {
                // Compute wavenumber components
                let kx = compute_wavenumber_component(i, grid.nx, grid.dx);
                let ky = compute_wavenumber_component(j, grid.ny, grid.dy);
                let kz = compute_wavenumber_component(k, grid.nz, grid.dz);

                // Sinc correction for each dimension
                let sinc_x = if kx.abs() > 1e-12 {
                    let arg = kx * grid.dx / 2.0;
                    arg.sin() / arg
                } else {
                    1.0
                };

                let sinc_y = if ky.abs() > 1e-12 {
                    let arg = ky * grid.dy / 2.0;
                    arg.sin() / arg
                } else {
                    1.0
                };

                let sinc_z = if kz.abs() > 1e-12 {
                    let arg = kz * grid.dz / 2.0;
                    arg.sin() / arg
                } else {
                    1.0
                };

                // Combined correction (inverse of sinc to compensate)
                kappa[[i, j, k]] = 1.0 / (sinc_x * sinc_y * sinc_z);
            }
        }
    }

    kappa
}

/// Compute wavenumber component for a given index
#[inline]
fn compute_wavenumber_component(index: usize, n: usize, dx: f64) -> f64 {
    if index <= n / 2 {
        2.0 * PI * index as f64 / (n as f64 * dx)
    } else {
        2.0 * PI * (index as f64 - n as f64) / (n as f64 * dx)
    }
}

/// Apply k-space correction to spectral field
pub fn apply_correction(field_k: &mut Array3<num_complex::Complex<f64>>, kappa: &Array3<f64>) {
    Zip::from(field_k).and(kappa).for_each(|f, &k| {
        *f *= num_complex::Complex::new(k, 0.0);
    });
}

/// Compute the numerical phase velocity for validation
pub fn compute_numerical_phase_velocity(k: f64, dx: f64, dt: f64, c_ref: f64) -> f64 {
    // Modified wavenumber
    let k_mod = 2.0 * (k * dx / 2.0).sin() / dx;

    // Numerical angular frequency (from leapfrog)
    let arg = c_ref * dt * k_mod / 2.0;

    if arg < 1.0 {
        let omega_num = 2.0 * arg.asin() / dt;
        omega_num / k // Phase velocity
    } else {
        0.0 // Beyond stability limit
    }
}

/// Compute the dispersion error for a given wavenumber
pub fn compute_dispersion_error(k: f64, dx: f64, dt: f64, c_ref: f64) -> f64 {
    let c_num = compute_numerical_phase_velocity(k, dx, dt, c_ref);
    (c_num - c_ref).abs() / c_ref // Relative error
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::Grid;

    #[test]
    fn test_exact_dispersion_correction() {
        let grid = Grid::new(64, 64, 64, 1e-3, 1e-3, 1e-3);
        let dt = 1e-6;
        let c_ref = 1500.0;

        let config = KSpaceCorrectionConfig {
            enabled: true,
            method: CorrectionMethod::ExactDispersion,
            cfl_number: 0.3,
            max_correction: 2.0,
        };

        let kappa = compute_kspace_correction(&grid, &config, dt, c_ref);

        // Check DC component (should be 1.0)
        assert!((kappa[[0, 0, 0]] - 1.0).abs() < 1e-10);

        // Check that correction factors are within bounds
        for val in kappa.iter() {
            assert!(*val >= 0.5 && *val <= 2.0);
        }
    }

    #[test]
    fn test_dispersion_error() {
        let dx = 1e-3;
        let dt = 1e-6;
        let c_ref = 1500.0;

        // Low frequency - should have minimal error
        let k_low = PI / (10.0 * dx); // 10 points per wavelength
        let error_low = compute_dispersion_error(k_low, dx, dt, c_ref);
        assert!(error_low < 0.01); // Less than 1% error

        // High frequency - will have more error
        let k_high = PI / (2.0 * dx); // 2 points per wavelength (Nyquist)
        let error_high = compute_dispersion_error(k_high, dx, dt, c_ref);
        assert!(error_high > error_low); // Error increases with frequency
    }

    #[test]
    fn test_phase_velocity_computation() {
        let dx = 1e-3;
        let dt = 1e-6;
        let c_ref = 1500.0;

        // DC component should propagate at exactly c_ref
        let c_dc = compute_numerical_phase_velocity(1e-12, dx, dt, c_ref);
        assert!((c_dc - c_ref).abs() / c_ref < 1e-6);
    }

    #[test]
    fn test_correction_methods_consistency() {
        let grid = Grid::new(32, 32, 32, 1e-3, 1e-3, 1e-3);
        let dt = 1e-6;
        let c_ref = 1500.0;

        // Test all methods produce reasonable corrections
        let methods = vec![
            CorrectionMethod::ExactDispersion,
            CorrectionMethod::KWave,
            CorrectionMethod::LiuPSTD,
            CorrectionMethod::SincSpatial,
        ];

        for method in methods {
            let config = KSpaceCorrectionConfig {
                enabled: true,
                method,
                cfl_number: 0.3,
                max_correction: 2.0,
            };

            let kappa = compute_kspace_correction(&grid, &config, dt, c_ref);

            // All methods should give unity at DC
            assert!((kappa[[0, 0, 0]] - 1.0).abs() < 0.01);

            // All corrections should be positive
            for val in kappa.iter() {
                assert!(*val > 0.0);
            }
        }
    }
}
