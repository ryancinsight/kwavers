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
            *km = kz_val
                .mul_add(kz_val, kx_val.mul_add(kx_val, ky_val * ky_val))
                .sqrt();
        });

    k_mag
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
