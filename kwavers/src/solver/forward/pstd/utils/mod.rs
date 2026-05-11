//! Centralized spectral utilities for k-space operations
//!
//! This module provides common functionality for computing wavenumbers,
//! k-space corrections, and spectral derivatives used by multiple solvers.

use crate::core::error::KwaversResult;
use crate::domain::grid::Grid;
use crate::math::fft::{fft_3d_array_into, ifft_3d_complex_inplace, KSpaceCalculator};
use ndarray::{s, Array3, Axis, Zip};
use num_complex::Complex64;

#[cfg(test)]
mod tests;

/// Compute wavenumber arrays for spectral operations
/// Returns (kx, ky, kz) arrays with proper Nyquist handling
pub fn compute_wavenumbers(grid: &Grid) -> (Array3<f64>, Array3<f64>, Array3<f64>) {
    let kx_vec = KSpaceCalculator::generate_k_vector(grid.nx, grid.dx);
    let ky_vec = KSpaceCalculator::generate_k_vector(grid.ny, grid.dy);
    let kz_vec = KSpaceCalculator::generate_k_vector(grid.nz, grid.dz);

    let mut kx = Array3::zeros((grid.nx, grid.ny, grid.nz));
    let mut ky = Array3::zeros((grid.nx, grid.ny, grid.nz));
    let mut kz = Array3::zeros((grid.nx, grid.ny, grid.nz));

    // Fill 3D arrays using broadcasting logic
    for i in 0..grid.nx {
        kx.slice_mut(s![i, .., ..]).fill(kx_vec[i]);
    }
    for j in 0..grid.ny {
        ky.slice_mut(s![.., j, ..]).fill(ky_vec[j]);
    }
    for k in 0..grid.nz {
        kz.slice_mut(s![.., .., k]).fill(kz_vec[k]);
    }

    (kx, ky, kz)
}

/// Compute anti-aliasing filter (low-pass filter in k-space)
/// Uses a Butterworth-style filter to smoothly roll off high frequencies
pub fn compute_anti_aliasing_filter(grid: &Grid, cutoff: f64, order: u32) -> Array3<f64> {
    let mut filter = Array3::zeros((grid.nx, grid.ny, grid.nz));
    let (nx, ny, nz) = (grid.nx, grid.ny, grid.nz);

    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                let ik = if i <= nx / 2 {
                    i as f64
                } else {
                    (nx as i64 - i as i64).abs() as f64
                };
                let jk = if j <= ny / 2 {
                    j as f64
                } else {
                    (ny as i64 - j as i64).abs() as f64
                };
                let kk = if k <= nz / 2 {
                    k as f64
                } else {
                    (nz as i64 - k as i64).abs() as f64
                };

                let r_x = ik / (nx as f64 / 2.0);
                let r_y = jk / (ny as f64 / 2.0);
                let r_z = kk / (nz as f64 / 2.0);

                let r = (r_x * r_x + r_y * r_y + r_z * r_z).sqrt();
                filter[[i, j, k]] = 1.0 / (1.0 + (r / cutoff).powi(2 * order as i32));
            }
        }
    }
    filter
}

/// Compute k² magnitude array for Laplacian operations
#[must_use]
pub fn compute_k_squared(kx: &Array3<f64>, ky: &Array3<f64>, kz: &Array3<f64>) -> Array3<f64> {
    let mut k_squared = Array3::zeros(kx.dim());

    Zip::from(&mut k_squared)
        .and(kx)
        .and(ky)
        .and(kz)
        .par_for_each(|k2, &kx_val, &ky_val, &kz_val| {
            *k2 = kz_val.mul_add(kz_val, kx_val.mul_add(kx_val, ky_val * ky_val));
        });

    k_squared
}

/// Compute k magnitude array
#[must_use]
pub fn compute_k_magnitude(kx: &Array3<f64>, ky: &Array3<f64>, kz: &Array3<f64>) -> Array3<f64> {
    let mut k_mag = Array3::zeros(kx.dim());

    Zip::from(&mut k_mag)
        .and(kx)
        .and(ky)
        .and(kz)
        .par_for_each(|km, &kx_val, &ky_val, &kz_val| {
            *km = kz_val.mul_add(kz_val, kx_val.mul_add(kx_val, ky_val * ky_val)).sqrt();
        });

    k_mag
}

/// Compute k-space correction factors for PSTD
/// These factors compensate for numerical dispersion in the finite difference stencil
pub fn compute_kspace_correction_factors(
    kx: &Array3<f64>,
    ky: &Array3<f64>,
    kz: &Array3<f64>,
    grid: &Grid,
    correction_type: CorrectionType,
    dt: f64,
    c_ref: f64,
) -> Array3<f64> {
    let mut correction = Array3::from_elem(kx.dim(), 1.0);

    match correction_type {
        CorrectionType::Liu1997 => {
            // Liu (1997) correction: sinc function
            Zip::from(&mut correction).and(kx).and(ky).and(kz).par_for_each(
                |c, &kx_val, &ky_val, &kz_val| {
                    let sinc_x = sinc(kx_val * grid.dx / 2.0);
                    let sinc_y = sinc(ky_val * grid.dy / 2.0);
                    let sinc_z = sinc(kz_val * grid.dz / 2.0);
                    *c = sinc_x * sinc_y * sinc_z;
                },
            );
        }
        CorrectionType::Treeby2010 => {
            // Treeby & Cox (2010) k-space correction: sinc(c_ref * dt * |k| / 2)
            // Uses UNNORMALIZED sinc sin(x)/x with x = c_ref·dt·|k|/2.
            //
            // Empirically verified against the k-Wave C++ binary (kspaceFirstOrder-OMP):
            // for N=16, dx=1mm, c=1500 m/s, dt=2e-7s, single-pulse injection:
            //   step2 p[8,8,8] = 0.5344 Pa (injection)
            //   step3 p[8,8,8] = 0.1128 Pa (free propagation, ratio ~0.211)
            // This ratio ~0.211 matches unnormalized sinc, NOT normalized sinc
            // (normalized sinc gives ratio ~0.66, mismatching k-Wave).
            //
            // Although np.sinc (Python kspaceFirstOrder3D.py line 298) uses normalised
            // sinc, the Python-precomputed kappa is NOT saved to disk — the C++ binary
            // recomputes kappa internally using sin(x)/x.
            Zip::from(&mut correction).and(kx).and(ky).and(kz).par_for_each(
                |c, &kx_val, &ky_val, &kz_val| {
                    let k_sq = kz_val.mul_add(kz_val, kx_val.mul_add(kx_val, ky_val * ky_val));
                    let k_mag = k_sq.sqrt();
                    let arg = c_ref * dt * k_mag / 2.0;
                    *c = sinc(arg);
                },
            );
        }
        CorrectionType::None => {
            // No correction (array initialized to 1.0)
        }
    }

    correction
}

/// Type of k-space correction to apply
#[derive(Debug, Clone, Copy)]
pub enum CorrectionType {
    /// No correction
    None,
    /// Liu (1997) sinc correction
    Liu1997,
    /// Treeby & Cox (2010) exact staggered grid correction
    Treeby2010,
}

/// Unnormalized sinc function: sinc(x) = sin(x)/x
/// Used for spatial corrections (Liu 1997) where the argument already
/// includes the correct scaling (e.g., kx·dx/2).
#[inline]
fn sinc(x: f64) -> f64 {
    if x.abs() < 1e-10 {
        1.0
    } else {
        x.sin() / x
    }
}

/// Normalized sinc function: sinc(x) = sin(π·x)/(π·x)
/// Matches numpy's np.sinc(x) convention. NOT used for kappa (C++ binary uses unnormalized).
#[cfg(test)]
#[inline]
fn sinc_normalized(x: f64) -> f64 {
    if x.abs() < 1e-10 {
        1.0
    } else {
        let pi_x = std::f64::consts::PI * x;
        pi_x.sin() / pi_x
    }
}

/// Apply anti-aliasing filter in k-space
#[must_use]
pub fn apply_antialiasing_filter(
    k_squared: &Array3<f64>,
    k_max_squared: f64,
    filter_type: FilterType,
) -> Array3<f64> {
    let mut filter = Array3::from_elem(k_squared.dim(), 1.0);

    match filter_type {
        FilterType::Smooth => {
            // Smooth roll-off filter
            let transition_width = 0.1 * k_max_squared;
            let cutoff = 0.8 * k_max_squared;

            Zip::from(&mut filter).and(k_squared).par_for_each(|f, &k2| {
                if k2 > cutoff {
                    let x = (k2 - cutoff) / transition_width;
                    *f = 0.5 * (1.0 - x.tanh());
                }
            });
        }
        FilterType::Sharp => {
            // Sharp cutoff at Nyquist
            Zip::from(&mut filter).and(k_squared).par_for_each(|f, &k2| {
                if k2 > k_max_squared {
                    *f = 0.0;
                }
            });
        }
        FilterType::None => {
            // No filtering
        }
    }

    filter
}

/// Type of anti-aliasing filter
#[derive(Debug, Clone, Copy)]
pub enum FilterType {
    /// No filtering
    None,
    /// Smooth roll-off filter
    Smooth,
    /// Sharp cutoff at Nyquist
    Sharp,
}

/// Spectral derivative of `field` along `axis` (0=x, 1=y, 2=z).
///
/// Implements ∂f/∂x_axis = IFFT(i · k_axis · FFT(f)).
///
/// The axis dispatch is hoisted outside the inner loop: after the 3D FFT,
/// each 2D slice along `axis` is multiplied by the scalar `i·k_vec[slice_idx]`.
/// This avoids a branch per element and lets LLVM vectorize the inner multiply.
fn spectral_deriv_axis(field: &Array3<f64>, grid: &Grid, axis: usize) -> Array3<f64> {
    let (nx, ny, nz) = field.dim();
    let mut fhat = Array3::<Complex64>::zeros((nx, ny, nz));
    fft_3d_array_into(field, &mut fhat);

    let k_vec = match axis {
        0 => KSpaceCalculator::generate_k_vector(grid.nx, grid.dx),
        1 => KSpaceCalculator::generate_k_vector(grid.ny, grid.dy),
        _ => KSpaceCalculator::generate_k_vector(grid.nz, grid.dz),
    };

    // Multiply each axis-slice by i·k[slice_idx].  Branch on `axis` is
    // outside the per-element loop; inner loop is branch-free and vectorisable.
    for (idx, &ki) in k_vec.iter().enumerate() {
        let scale = Complex64::new(0.0, ki);
        fhat.index_axis_mut(Axis(axis), idx)
            .par_mapv_inplace(|c| c * scale);
    }

    ifft_3d_complex_inplace(&mut fhat);
    fhat.mapv(|c| c.re)
}
/// Gradient x.
/// # Errors
/// - Returns [`Err`] if an internal constraint is violated.
///
pub fn gradient_x(field: &Array3<f64>, grid: &Grid) -> KwaversResult<Array3<f64>> {
    Ok(spectral_deriv_axis(field, grid, 0))
}
/// Gradient y.
/// # Errors
/// - Returns [`Err`] if an internal constraint is violated.
///
pub fn gradient_y(field: &Array3<f64>, grid: &Grid) -> KwaversResult<Array3<f64>> {
    Ok(spectral_deriv_axis(field, grid, 1))
}
/// Gradient z.
/// # Errors
/// - Returns [`Err`] if an internal constraint is violated.
///
pub fn gradient_z(field: &Array3<f64>, grid: &Grid) -> KwaversResult<Array3<f64>> {
    Ok(spectral_deriv_axis(field, grid, 2))
}
