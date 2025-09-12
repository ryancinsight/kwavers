//! k-Wave Utility Functions
//!
//! This module contains helper functions for k-Wave implementation,
//! separated according to Single Responsibility Principle (GRASP).

use crate::grid::Grid;
use ndarray::{Array3, Zip};
use rustfft::{num_complex::Complex64, FftPlanner};

/// Compute k-space correction following k-Wave methodology
/// Note: k-Wave compatibility function for heterogeneous media
pub(super) fn compute_kspace_correction(
    grid: &Grid,
    k_vec: &(Array3<f64>, Array3<f64>, Array3<f64>),
) -> Array3<f64> {
    let mut kappa = Array3::ones((grid.nx, grid.ny, grid.nz));

    Zip::from(&mut kappa)
        .and(&k_vec.0)
        .and(&k_vec.1)
        .and(&k_vec.2)
        .for_each(|kap, &kx, &ky, &kz| {
            let k_mag = (kx * kx + ky * ky + kz * kz).sqrt();

            if k_mag > 0.0 {
                // Exact k-space correction for PSTD
                let kx_exact = if kx.abs() > 0.0 {
                    (kx * grid.dx / 2.0).sin() * 2.0 / grid.dx
                } else {
                    kx
                };

                let ky_exact = if ky.abs() > 0.0 {
                    (ky * grid.dy / 2.0).sin() * 2.0 / grid.dy
                } else {
                    ky
                };

                let kz_exact = if kz.abs() > 0.0 {
                    (kz * grid.dz / 2.0).sin() * 2.0 / grid.dz
                } else {
                    kz
                };

                let k_exact =
                    (kx_exact * kx_exact + ky_exact * ky_exact + kz_exact * kz_exact).sqrt();

                *kap = k_exact / k_mag;
            }
        });

    kappa
}

/// Compute PML absorption operators
pub(super) fn compute_pml_operators(
    grid: &Grid,
    pml_size: usize,
    pml_alpha: f64,
) -> Array3<f64> {
    let mut pml = Array3::ones((grid.nx, grid.ny, grid.nz));

    // Apply PML in each direction
    for ((i, j, k), pml_val) in pml.indexed_iter_mut() {
        let mut absorption = 0.0;

        // X boundaries
        if i < pml_size {
            let dist = (pml_size - i) as f64 / pml_size as f64;
            absorption += pml_alpha * dist * dist;
        } else if i >= grid.nx - pml_size {
            let dist = (i - (grid.nx - pml_size - 1)) as f64 / pml_size as f64;
            absorption += pml_alpha * dist * dist;
        }

        // Y boundaries (similar for other directions)
        if j < pml_size {
            let dist = (pml_size - j) as f64 / pml_size as f64;
            absorption += pml_alpha * dist * dist;
        } else if j >= grid.ny - pml_size {
            let dist = (j - (grid.ny - pml_size - 1)) as f64 / pml_size as f64;
            absorption += pml_alpha * dist * dist;
        }

        // Z boundaries
        if k < pml_size {
            let dist = (pml_size - k) as f64 / pml_size as f64;
            absorption += pml_alpha * dist * dist;
        } else if k >= grid.nz - pml_size {
            let dist = (k - (grid.nz - pml_size - 1)) as f64 / pml_size as f64;
            absorption += pml_alpha * dist * dist;
        }

        *pml_val = (-absorption).exp();
    }

    pml
}

/// FFT wrapper - k-Wave compatibility function
pub(super) fn fft_3d(input: &Array3<f64>, _planner: &mut FftPlanner<f64>) -> Array3<Complex64> {
    // Use modern FFT from utils instead of placeholder
    crate::utils::fft_3d_array(input)
}

/// IFFT wrapper - k-Wave compatibility function
pub(super) fn ifft_3d(input: &Array3<Complex64>, _planner: &mut FftPlanner<f64>) -> Array3<f64> {
    // Use modern IFFT from utils instead of placeholder
    crate::utils::ifft_3d_array(input)
}

/// Smooth source for stability - k-Wave compatibility function
pub(super) fn smooth_source(source: &Array3<f64>, _grid: &Grid) -> Array3<f64> {
    // Apply spatial smoothing filter
    // For now, return original - proper smoothing would use convolution
    source.clone()
}