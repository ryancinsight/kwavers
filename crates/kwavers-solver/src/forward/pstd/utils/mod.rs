//! Centralized spectral utilities for k-space operations
//!
//! This module provides common functionality for computing wavenumbers,
//! k-space corrections, and spectral derivatives used by multiple solvers.

use kwavers_core::error::KwaversResult;
use kwavers_grid::Grid;
use kwavers_math::fft::{fft_3d_array_into, ifft_3d_complex_inplace, KSpaceCalculator};
use leto::{Array3, ArrayViewMut2};
use moirai_parallel::{enumerate_mut_with, Adaptive};
use kwavers_math::fft::Complex64;

#[cfg(test)]
mod tests;

#[derive(Clone, Copy)]
enum KNormOutput {
    Squared,
    Magnitude,
}

impl KNormOutput {
    fn finish(self, squared: f64) -> f64 {
        match self {
            Self::Squared => squared,
            Self::Magnitude => squared.sqrt(),
        }
    }
}

fn fill_k_norm(
    output: &mut Array3<f64>,
    kx: &Array3<f64>,
    ky: &Array3<f64>,
    kz: &Array3<f64>,
    output_kind: KNormOutput,
) {
    assert_eq!(
        output.shape(),
        kx.shape(),
        "invariant: PSTD k-norm output shape matches kx"
    );
    assert_eq!(
        output.shape(),
        ky.shape(),
        "invariant: PSTD k-norm output shape matches ky"
    );
    assert_eq!(
        output.shape(),
        kz.shape(),
        "invariant: PSTD k-norm output shape matches kz"
    );

    if let (Some(output_values), Some(kx_values), Some(ky_values), Some(kz_values)) = (
        output.as_slice_mut(),
        kx.as_slice(),
        ky.as_slice(),
        kz.as_slice(),
    ) {
        enumerate_mut_with::<Adaptive, _, _>(output_values, |index, value| {
            let squared = kz_values[index].mul_add(
                kz_values[index],
                kx_values[index].mul_add(kx_values[index], ky_values[index] * ky_values[index]),
            );
            *value = output_kind.finish(squared);
        });
    } else {
        let [nx, ny, nz] = output.shape();
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let squared = kz[[i, j, k]].mul_add(
                        kz[[i, j, k]],
                        kx[[i, j, k]].mul_add(kx[[i, j, k]], ky[[i, j, k]] * ky[[i, j, k]]),
                    );
                    *output.get_mut([i, j, k]).unwrap() = output_kind.finish(squared);
                }
            }
        }
    }
}

fn scale_complex_view_in_place(mut values: ArrayViewMut2<'_, Complex64>, scale: Complex64) {
    if let Some(slice) = values.as_mut_slice() {
        enumerate_mut_with::<Adaptive, _, _>(slice, |_index, value| {
            *value *= scale;
        });
    } else {
        let [rows, cols] = values.shape();
        for row in 0..rows {
            for col in 0..cols {
                values[[row, col]] *= scale;
            }
        }
    }
}

/// Compute wavenumber arrays for spectral operations
/// Returns (kx, ky, kz) arrays with proper Nyquist handling
pub fn compute_wavenumbers(grid: &Grid) -> (Array3<f64>, Array3<f64>, Array3<f64>) {
    let kx_vec = KSpaceCalculator::generate_k_vector(grid.nx, grid.dx);
    let ky_vec = KSpaceCalculator::generate_k_vector(grid.ny, grid.dy);
    let kz_vec = KSpaceCalculator::generate_k_vector(grid.nz, grid.dz);

    let mut kx = Array3::<f64>::zeros([grid.nx, grid.ny, grid.nz]);
    let mut ky = Array3::<f64>::zeros([grid.nx, grid.ny, grid.nz]);
    let mut kz = Array3::<f64>::zeros([grid.nx, grid.ny, grid.nz]);

    // Fill 3D arrays broadcasting each wavenumber axis
    for i in 0..grid.nx {
        for j in 0..grid.ny {
            for k in 0..grid.nz {
                *kx.get_mut([i, j, k]).unwrap() = kx_vec[i];
                *ky.get_mut([i, j, k]).unwrap() = ky_vec[j];
                *kz.get_mut([i, j, k]).unwrap() = kz_vec[k];
            }
        }
    }

    (kx, ky, kz)
}

/// Compute anti-aliasing filter (low-pass filter in k-space)
/// Uses a Butterworth-style filter to smoothly roll off high frequencies
pub fn compute_anti_aliasing_filter(grid: &Grid, cutoff: f64, order: u32) -> Array3<f64> {
    let mut filter = Array3::<f64>::zeros([grid.nx, grid.ny, grid.nz]);
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
    let shape = kx.shape();
    let mut k_squared = Array3::<f64>::zeros([shape[0], shape[1], shape[2]]);
    fill_k_norm(&mut k_squared, kx, ky, kz, KNormOutput::Squared);
    k_squared
}

/// Compute k magnitude array
#[must_use]
pub fn compute_k_magnitude(kx: &Array3<f64>, ky: &Array3<f64>, kz: &Array3<f64>) -> Array3<f64> {
    let shape = kx.shape();
    let mut k_mag = Array3::<f64>::zeros([shape[0], shape[1], shape[2]]);
    fill_k_norm(&mut k_mag, kx, ky, kz, KNormOutput::Magnitude);
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
    let [nx, ny, nz] = field.shape();
    let mut fhat = Array3::<Complex64>::from_elem([nx, ny, nz], Complex64::default());
    fft_3d_array_into(field, &mut fhat);

    let k_vec = match axis {
        0 => KSpaceCalculator::generate_k_vector(grid.nx, grid.dx),
        1 => KSpaceCalculator::generate_k_vector(grid.ny, grid.dy),
        _ => KSpaceCalculator::generate_k_vector(grid.nz, grid.dz),
    };

    // Multiply each axis-slice by i·k[slice_idx].
    for (idx, &ki) in k_vec.iter().enumerate() {
        let scale = Complex64::new(0.0, ki);
        let slice = fhat
            .index_axis_mut::<2>(axis, idx)
            .expect("valid axis index");
        scale_complex_view_in_place(slice, scale);
    }

    ifft_3d_complex_inplace(&mut fhat);
    Array3::<f64>::from_shape_vec([nx, ny, nz], fhat.iter().map(|c| c.re).collect())
        .expect("spectral_deriv_axis: shape matches element count")
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
