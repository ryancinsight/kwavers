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
use ndarray::{Array3, Zip};
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

#[cfg(test)]
mod tests;

/// Spectral correction configuration
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct SpectralCorrectionConfig {
    /// Enable spectral correction
    pub enabled: bool,
    /// Correction method
    pub method: CorrectionMethod,
    /// CFL number for stability
    pub cfl_number: f64,
    /// Maximum correction factor (for stability)
    pub max_correction: f64,
}

/// Correction method selection
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum CorrectionMethod {
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
            // matching k-wave-python's `kspace_solver.py` default. Was
            // previously ExactDispersion which uses an `omega_phys/omega_num`
            // ratio that produces high-frequency amplification (correction
            // > 1 near Nyquist) rather than the Treeby/sinc attenuation
            // (correction < 1). The mismatch surfaced as ~30% peak inflation
            // in pykwavers parity scripts vs k-wave-python; switching the
            // default to Treeby2010 (combined with the kappa-inversion fix
            // landed in the same commit) closes the gap.
            method: CorrectionMethod::Treeby2010,
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

    match config.method {
        CorrectionMethod::ExactDispersion => {
            compute_exact_dispersion_correction(grid, dt, c_ref, config.max_correction)
        }
        CorrectionMethod::Treeby2010 => {
            compute_treeby2010_correction(grid, dt, c_ref, config.cfl_number, config.max_correction)
        }
        CorrectionMethod::LiuPSTD => {
            compute_liu_pstd_correction(grid, dt, c_ref, config.max_correction)
        }
        CorrectionMethod::LowDispersionPSTD => {
            compute_low_dispersion_pstd_correction(grid, dt, c_ref, config.max_correction)
        }
        CorrectionMethod::SincSpatial => compute_sinc_spatial_correction(grid),
    }
}

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
                let kx = compute_wavenumber_component(i, grid.nx, grid.dx);
                let ky = compute_wavenumber_component(j, grid.ny, grid.dy);
                let kz = compute_wavenumber_component(k, grid.nz, grid.dz);

                let k_phys = (kx * kx + ky * ky + kz * kz).sqrt();

                if k_phys > 0.0 {
                    let kx_mod = 2.0 * (kx * grid.dx / 2.0).sin() / grid.dx;
                    let ky_mod = 2.0 * (ky * grid.dy / 2.0).sin() / grid.dy;
                    let kz_mod = 2.0 * (kz * grid.dz / 2.0).sin() / grid.dz;

                    let k_mod = (kx_mod * kx_mod + ky_mod * ky_mod + kz_mod * kz_mod).sqrt();

                    if k_mod > 0.0 {
                        let omega_phys = c_ref * k_phys;

                        let arg = c_ref * dt * k_mod / 2.0;

                        if arg < 1.0 {
                            let omega_num = 2.0 * arg.asin() / dt;
                            let correction = omega_phys / omega_num;

                            kappa[[i, j, k]] =
                                correction.min(max_correction).max(1.0 / max_correction);
                        } else {
                            kappa[[i, j, k]] = 1.0 / max_correction;
                        }
                    }
                }
            }
        }
    }

    kappa
}

/// Treeby & Cox 2010 k-space correction κ = sinc(c_ref·dt·|k|/2).
///
/// This is the canonical kappa derived in Treeby B. E. & Cox B. T. (2010),
/// "Modeling power law absorption and dispersion for acoustic propagation
/// using the fractional Laplacian," J. Acoust. Soc. Am. 127(5) 2741-2748,
/// Eq. 18. With this kappa applied to the spatial derivative `i·k`, the
/// leapfrog k-space scheme reproduces the EXACT analytical wave-equation
/// update step for plane waves at `c = c_ref`:
///
///     p^{n+1} − 2·p^n + p^{n−1} = −4·sin²(c·dt·|k|/2) · p^n   (in k-space)
///
/// which matches the exact solution `p(t±dt) = exp(±i·c·|k|·dt) · p(t)`.
///
/// **Reference implementation**: k-wave-python's `kspace_solver.py` line 389:
///     `self.kappa = xp.sinc((self.c_ref * k_mag * self.dt / 2) / np.pi)`
///
/// where numpy's `sinc(x) = sin(πx)/(πx)`, so passing `(arg)/π` recovers the
/// unnormalised `sin(arg)/arg`. Equivalent to `sin(c·k·dt/2)/(c·k·dt/2)`.
///
/// **Prior pykwavers bug**: this function previously computed
/// `1/sinc(c·dt·|k_mod|/2)` (the reciprocal) combined with a `(|k_mod|/|k_phys|)`
/// spatial correction. Both were wrong for the canonical Treeby & Cox
/// formulation — the bug surfaced as ~30% peak amplitude inflation in
/// 1-D `na_modelling_absorption_compare.py` and ~28% RMS drift at d=40
/// grid points in 2-D `ivp_loading_external_image_compare.py`. Fixed in
/// the commit that introduced this comment.
fn compute_treeby2010_correction(
    grid: &Grid,
    dt: f64,
    c_ref: f64,
    _cfl: f64,
    max_correction: f64,
) -> Array3<f64> {
    let mut kappa = Array3::from_elem((grid.nx, grid.ny, grid.nz), 1.0);

    for i in 0..grid.nx {
        for j in 0..grid.ny {
            for k in 0..grid.nz {
                let kx = compute_wavenumber_component(i, grid.nx, grid.dx);
                let ky = compute_wavenumber_component(j, grid.ny, grid.dy);
                let kz = compute_wavenumber_component(k, grid.nz, grid.dz);

                let k_phys = (kx * kx + ky * ky + kz * kz).sqrt();
                if k_phys <= 0.0 {
                    continue; // DC: kappa = 1 (already initialised)
                }

                // arg = c_ref · dt · |k_phys| / 2
                // kappa = sinc(arg) = sin(arg) / arg  (unnormalised)
                let arg = 0.5 * c_ref * dt * k_phys;
                let correction = if arg.abs() < 1e-12 {
                    1.0 // L'Hôpital: lim sin(x)/x = 1
                } else {
                    arg.sin() / arg
                };
                kappa[[i, j, k]] = correction
                    .min(max_correction)
                    .max(1.0 / max_correction);
            }
        }
    }

    kappa
}

fn compute_liu_pstd_correction(
    grid: &Grid,
    dt: f64,
    c_ref: f64,
    max_correction: f64,
) -> Array3<f64> {
    let mut kappa = Array3::from_elem((grid.nx, grid.ny, grid.nz), 1.0);

    let dx_min = grid.dx.min(grid.dy).min(grid.dz);
    let stability_factor = c_ref * dt / dx_min;

    for i in 0..grid.nx {
        for j in 0..grid.ny {
            for k in 0..grid.nz {
                let kx = compute_wavenumber_component(i, grid.nx, grid.dx);
                let ky = compute_wavenumber_component(j, grid.ny, grid.dy);
                let kz = compute_wavenumber_component(k, grid.nz, grid.dz);

                let k_mag = (kx * kx + ky * ky + kz * kz).sqrt();

                if k_mag > 0.0 {
                    let k_dx = k_mag * dx_min;

                    let correction = if k_dx < PI {
                        1.0 + stability_factor * stability_factor * k_dx * k_dx / 24.0
                    } else {
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

fn compute_low_dispersion_pstd_correction(
    grid: &Grid,
    dt: f64,
    c_ref: f64,
    max_correction: f64,
) -> Array3<f64> {
    let mut kappa = Array3::from_elem((grid.nx, grid.ny, grid.nz), 1.0);

    for i in 0..grid.nx {
        for j in 0..grid.ny {
            for k in 0..grid.nz {
                let kx = compute_wavenumber_component(i, grid.nx, grid.dx);
                let ky = compute_wavenumber_component(j, grid.ny, grid.dy);
                let kz = compute_wavenumber_component(k, grid.nz, grid.dz);

                let k_mag = (kx * kx + ky * ky + kz * kz).sqrt();

                if k_mag > 0.0 {
                    let omega_dt = c_ref * dt * k_mag;
                    let correction = if omega_dt > 0.0 {
                        omega_dt / (2.0 * (omega_dt / 2.0).sin())
                    } else {
                        1.0
                    };

                    kappa[[i, j, k]] = correction.min(max_correction).max(1.0 / max_correction);
                }
            }
        }
    }

    kappa
}

fn compute_sinc_spatial_correction(grid: &Grid) -> Array3<f64> {
    let mut kappa = Array3::from_elem((grid.nx, grid.ny, grid.nz), 1.0);

    for i in 0..grid.nx {
        for j in 0..grid.ny {
            for k in 0..grid.nz {
                let kx = compute_wavenumber_component(i, grid.nx, grid.dx);
                let ky = compute_wavenumber_component(j, grid.ny, grid.dy);
                let kz = compute_wavenumber_component(k, grid.nz, grid.dz);

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

                kappa[[i, j, k]] = 1.0 / (sinc_x * sinc_y * sinc_z);
            }
        }
    }

    kappa
}

#[inline]
fn compute_wavenumber_component(index: usize, n: usize, dx: f64) -> f64 {
    if index <= n / 2 {
        2.0 * PI * index as f64 / (n as f64 * dx)
    } else {
        2.0 * PI * (index as f64 - n as f64) / (n as f64 * dx)
    }
}

pub fn apply_correction(field_k: &mut Array3<num_complex::Complex<f64>>, kappa: &Array3<f64>) {
    Zip::from(field_k).and(kappa).for_each(|f, &k| {
        *f *= num_complex::Complex::new(k, 0.0);
    });
}

#[must_use]
pub fn compute_numerical_phase_velocity(k: f64, dx: f64, dt: f64, c_ref: f64) -> f64 {
    let k_mod = 2.0 * (k * dx / 2.0).sin() / dx;
    let arg = c_ref * dt * k_mod / 2.0;

    if arg < 1.0 {
        let omega_num = 2.0 * arg.asin() / dt;
        omega_num / k
    } else {
        0.0
    }
}

#[must_use]
pub fn compute_dispersion_error(k: f64, dx: f64, dt: f64, c_ref: f64) -> f64 {
    let c_num = compute_numerical_phase_velocity(k, dx, dt, c_ref);
    (c_num - c_ref).abs() / c_ref
}
