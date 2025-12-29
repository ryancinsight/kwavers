//! Centralized spectral utilities for k-space operations
//!
//! This module provides common functionality for computing wavenumbers,
//! k-space corrections, and spectral derivatives used by multiple solvers.

use crate::error::KwaversResult;
use crate::grid::Grid;
use ndarray::{s, Array3, Array4, Axis, Zip};
use num_complex::Complex;
use std::f64::consts::PI;

/// Compute wavenumber arrays for spectral operations
/// Returns (kx, ky, kz) arrays with proper Nyquist handling
pub fn compute_wavenumbers(grid: &Grid) -> (Array3<f64>, Array3<f64>, Array3<f64>) {
    let (nx, ny, nz) = (grid.nx, grid.ny, grid.nz);
    let (dx, dy, dz) = (grid.dx, grid.dy, grid.dz);

    // Create 1D wavenumber arrays
    let kx_1d = compute_1d_wavenumbers(nx, dx);
    let ky_1d = compute_1d_wavenumbers(ny, dy);
    let kz_1d = compute_1d_wavenumbers(nz, dz);

    // Broadcast to 3D arrays
    let mut kx = Array3::zeros((nx, ny, nz));
    let mut ky = Array3::zeros((nx, ny, nz));
    let mut kz = Array3::zeros((nx, ny, nz));

    // Use broadcasting for efficient filling
    for (i, &kx_val) in kx_1d.iter().enumerate().take(nx) {
        kx.slice_mut(s![i, .., ..]).fill(kx_val);
    }
    for (j, &ky_val) in ky_1d.iter().enumerate().take(ny) {
        ky.slice_mut(s![.., j, ..]).fill(ky_val);
    }
    for (k_idx, &kz_val) in kz_1d.iter().enumerate().take(nz) {
        kz.slice_mut(s![.., .., k_idx]).fill(kz_val);
    }

    (kx, ky, kz)
}

/// Compute 1D wavenumber array with proper Nyquist handling
fn compute_1d_wavenumbers(n: usize, dx: f64) -> Vec<f64> {
    let mut k = vec![0.0; n];
    let dk = 2.0 * PI / (n as f64 * dx);

    for (i, k_val) in k.iter_mut().enumerate().take(n) {
        if i <= n / 2 {
            *k_val = i as f64 * dk;
        } else {
            *k_val = -((n - i) as f64) * dk;
        }
    }

    // Note: Nyquist frequency should NOT be set to zero
    // It represents the highest resolvable frequency
    // Setting it to zero would lose information

    k
}

/// Compute k² magnitude array for Laplacian operations
#[must_use]
pub fn compute_k_squared(kx: &Array3<f64>, ky: &Array3<f64>, kz: &Array3<f64>) -> Array3<f64> {
    let mut k_squared = Array3::zeros(kx.dim());

    Zip::from(&mut k_squared)
        .and(kx)
        .and(ky)
        .and(kz)
        .for_each(|k2, &kx_val, &ky_val, &kz_val| {
            *k2 = kx_val * kx_val + ky_val * ky_val + kz_val * kz_val;
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
        .for_each(|km, &kx_val, &ky_val, &kz_val| {
            *km = (kx_val * kx_val + ky_val * ky_val + kz_val * kz_val).sqrt();
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
            Zip::from(&mut correction).and(kx).and(ky).and(kz).for_each(
                |c, &kx_val, &ky_val, &kz_val| {
                    let sinc_x = sinc(kx_val * grid.dx / 2.0);
                    let sinc_y = sinc(ky_val * grid.dy / 2.0);
                    let sinc_z = sinc(kz_val * grid.dz / 2.0);
                    *c = sinc_x * sinc_y * sinc_z;
                },
            );
        }
        CorrectionType::Treeby2010 => {
            // Treeby & Cox (2010) correction: exact for staggered grid + temporal dispersion
            Zip::from(&mut correction).and(kx).and(ky).and(kz).for_each(
                |c, &kx_val, &ky_val, &kz_val| {
                    // Physical wavenumber magnitude
                    let k_sq = kx_val * kx_val + ky_val * ky_val + kz_val * kz_val;
                    let k_mag = k_sq.sqrt();

                    // Temporal correction (k-Wave standard)
                    // kappa = sinc(c0 * dt * k / 2)
                    let temporal_correction = sinc(c_ref * dt * k_mag / 2.0);

                    // Note: k-Wave also includes spatial sinc if using staggered grid.
                    // Assuming collocated for now, so only temporal correction is applied.
                    // If we want to match k-Wave's "exact" behavior on a staggered grid,
                    // we would also multiply by spatial sincs.

                    *c = temporal_correction;
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

/// Sinc function for k-space corrections
#[inline]
fn sinc(x: f64) -> f64 {
    if x.abs() < 1e-10 {
        1.0
    } else {
        x.sin() / x
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

            Zip::from(&mut filter).and(k_squared).for_each(|f, &k2| {
                if k2 > cutoff {
                    let x = (k2 - cutoff) / transition_width;
                    *f = 0.5 * (1.0 - x.tanh());
                }
            });
        }
        FilterType::Sharp => {
            // Sharp cutoff at Nyquist
            Zip::from(&mut filter).and(k_squared).for_each(|f, &k2| {
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

/// Compute spectral gradient in x direction
///
/// Uses FFT to compute the gradient with spectral accuracy
pub fn gradient_x(field: &Array3<f64>, grid: &Grid) -> KwaversResult<Array3<f64>> {
    let (nx, ny, nz) = field.dim();

    // Convert to Array4 for FFT
    let mut fields = Array4::zeros((1, nx, ny, nz));
    fields.index_axis_mut(Axis(0), 0).assign(field);

    // Perform FFT using utils function
    let slice = fields.index_axis(Axis(0), 0);
    let field_fft = crate::utils::fft_3d_array(&slice.to_owned());

    // Get wavenumbers
    let (kx, _, _) = compute_wavenumbers(grid);

    // Multiply by i*kx in Fourier space
    let mut grad_fft = Array3::<Complex<f64>>::zeros((nx, ny, nz));
    Zip::from(&mut grad_fft)
        .and(&field_fft)
        .and(&kx)
        .for_each(|g, &f, &k| {
            *g = Complex::new(0.0, k) * f;
        });

    // Perform inverse FFT
    Ok(crate::utils::fft_operations::ifft_3d_array(&grad_fft))
}

/// Compute spectral gradient in y direction
///
/// Uses FFT to compute the gradient with spectral accuracy
pub fn gradient_y(field: &Array3<f64>, grid: &Grid) -> KwaversResult<Array3<f64>> {
    let (nx, ny, nz) = field.dim();

    // Convert to Array4 for FFT
    let mut fields = Array4::zeros((1, nx, ny, nz));
    fields.index_axis_mut(Axis(0), 0).assign(field);

    // Perform FFT using utils function
    let slice = fields.index_axis(Axis(0), 0);
    let field_fft = crate::utils::fft_3d_array(&slice.to_owned());

    // Get wavenumbers
    let (_, ky, _) = compute_wavenumbers(grid);

    // Multiply by i*ky in Fourier space
    let mut grad_fft = Array3::<Complex<f64>>::zeros((nx, ny, nz));
    Zip::from(&mut grad_fft)
        .and(&field_fft)
        .and(&ky)
        .for_each(|g, &f, &k| {
            *g = Complex::new(0.0, k) * f;
        });

    // Perform inverse FFT
    Ok(crate::utils::fft_operations::ifft_3d_array(&grad_fft))
}

/// Compute spectral gradient in z direction
///
/// Uses FFT to compute the gradient with spectral accuracy
pub fn gradient_z(field: &Array3<f64>, grid: &Grid) -> KwaversResult<Array3<f64>> {
    let (nx, ny, nz) = field.dim();

    // Convert to Array4 for FFT
    let mut fields = Array4::zeros((1, nx, ny, nz));
    fields.index_axis_mut(Axis(0), 0).assign(field);

    // Perform FFT using utils function
    let slice = fields.index_axis(Axis(0), 0);
    let field_fft = crate::utils::fft_3d_array(&slice.to_owned());

    // Get wavenumbers
    let (_, _, kz) = compute_wavenumbers(grid);

    // Multiply by i*kz in Fourier space
    let mut grad_fft = Array3::<Complex<f64>>::zeros((nx, ny, nz));
    Zip::from(&mut grad_fft)
        .and(&field_fft)
        .and(&kz)
        .for_each(|g, &f, &k| {
            *g = Complex::new(0.0, k) * f;
        });

    // Perform inverse FFT
    Ok(crate::utils::fft_operations::ifft_3d_array(&grad_fft))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wavenumber_computation() {
        let grid = Grid::new(8, 8, 8, 0.001, 0.001, 0.001).expect("Failed to create test grid");
        let (kx, ky, kz) = compute_wavenumbers(&grid);

        // Check dimensions
        assert_eq!(kx.dim(), (8, 8, 8));
        assert_eq!(ky.dim(), (8, 8, 8));
        assert_eq!(kz.dim(), (8, 8, 8));

        // Check DC component
        assert_eq!(kx[[0, 0, 0]], 0.0);
        assert_eq!(ky[[0, 0, 0]], 0.0);
        assert_eq!(kz[[0, 0, 0]], 0.0);

        // Check symmetry
        assert!((kx[[1, 0, 0]] + kx[[7, 0, 0]]).abs() < 1e-10);
    }

    #[test]
    fn test_sinc_function() {
        assert_eq!(sinc(0.0), 1.0);
        assert!((sinc(PI) - 0.0).abs() < 1e-10);
        assert!((sinc(PI / 2.0) - 2.0 / PI).abs() < 1e-10);
    }
}
