//! Spectral operations for Westervelt solver

use crate::domain::grid::Grid;
use crate::math::fft::{fft_3d_array, ifft_3d_array};
use ndarray::Array3;
use num_complex::Complex;
use std::f64::consts::PI;

/// Initialize k-space grids for spectral operations
pub fn initialize_kspace_grids(
    grid: &Grid,
) -> (Array3<f64>, Array3<f64>, Array3<f64>, Array3<f64>) {
    let (nx, ny, nz) = (grid.nx, grid.ny, grid.nz);

    let mut kx = Array3::<f64>::zeros((nx, ny, nz));
    let mut ky = Array3::<f64>::zeros((nx, ny, nz));
    let mut kz = Array3::<f64>::zeros((nx, ny, nz));
    let mut k_squared = Array3::<f64>::zeros((nx, ny, nz));

    // Create k-space grids
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let kx_val = if i <= nx / 2 {
                    2.0 * PI * i as f64 / (nx as f64 * grid.dx)
                } else {
                    2.0 * PI * (i as f64 - nx as f64) / (nx as f64 * grid.dx)
                };

                let ky_val = if j <= ny / 2 {
                    2.0 * PI * j as f64 / (ny as f64 * grid.dy)
                } else {
                    2.0 * PI * (j as f64 - ny as f64) / (ny as f64 * grid.dy)
                };

                let kz_val = if k <= nz / 2 {
                    2.0 * PI * k as f64 / (nz as f64 * grid.dz)
                } else {
                    2.0 * PI * (k as f64 - nz as f64) / (nz as f64 * grid.dz)
                };

                kx[[i, j, k]] = kx_val;
                ky[[i, j, k]] = ky_val;
                kz[[i, j, k]] = kz_val;
                k_squared[[i, j, k]] = kx_val * kx_val + ky_val * ky_val + kz_val * kz_val;
            }
        }
    }

    (k_squared, kx, ky, kz)
}

/// Compute Laplacian using spectral method
#[must_use]
pub fn compute_laplacian_spectral(field: &Array3<f64>, k_squared: &Array3<f64>) -> Array3<f64> {
    // Transform to k-space
    let field_k = fft_3d_array(field);

    // Apply Laplacian in k-space: ∇²f = -k²f
    let laplacian_k = &field_k * &k_squared.mapv(|k2| Complex::new(-k2, 0.0));

    // Transform back to real space

    ifft_3d_array(&laplacian_k)
}

/// Apply k-space correction for heterogeneous media
///
/// This implements the k-space correction term from Tabei et al. (2002)
/// to improve accuracy in heterogeneous media.
pub fn apply_kspace_correction(
    pressure_k: &mut Array3<Complex<f64>>,
    kx: &Array3<f64>,
    ky: &Array3<f64>,
    kz: &Array3<f64>,
    rho_grad_x: &Array3<f64>,
    rho_grad_y: &Array3<f64>,
    rho_grad_z: &Array3<f64>,
) {
    let (nx, ny, nz) = pressure_k.dim();

    // Transform density gradients to k-space
    let rho_grad_x_k = fft_3d_array(rho_grad_x);
    let rho_grad_y_k = fft_3d_array(rho_grad_y);
    let rho_grad_z_k = fft_3d_array(rho_grad_z);

    // Apply correction term
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let kx_val = kx[[i, j, k]];
                let ky_val = ky[[i, j, k]];
                let kz_val = kz[[i, j, k]];

                let k_squared = kx_val * kx_val + ky_val * ky_val + kz_val * kz_val;

                if k_squared > 1e-10 {
                    // Compute k · ∇ρ
                    let k_dot_grad_rho = Complex::new(kx_val, 0.0) * rho_grad_x_k[[i, j, k]]
                        + Complex::new(ky_val, 0.0) * rho_grad_y_k[[i, j, k]]
                        + Complex::new(kz_val, 0.0) * rho_grad_z_k[[i, j, k]];

                    // Apply correction: p_k = p_k * (1 - i * k·∇ρ / k²)
                    let correction = Complex::new(1.0, 0.0)
                        - Complex::new(0.0, 1.0) * k_dot_grad_rho / k_squared;
                    pressure_k[[i, j, k]] *= correction;
                }
            }
        }
    }
}

/// Compute density gradients for k-space correction
/// Note: Reserved for future heterogeneous media implementation
#[allow(dead_code)]
pub fn compute_density_gradients(
    rho_arr: &Array3<f64>,
    grid: &Grid,
) -> (Array3<f64>, Array3<f64>, Array3<f64>) {
    let (nx, ny, nz) = (grid.nx, grid.ny, grid.nz);
    let mut grad_x = Array3::zeros((nx, ny, nz));
    let mut grad_y = Array3::zeros((nx, ny, nz));
    let mut grad_z = Array3::zeros((nx, ny, nz));

    let dx_inv = 0.5 / grid.dx;
    let dy_inv = 0.5 / grid.dy;
    let dz_inv = 0.5 / grid.dz;

    for k in 1..nz - 1 {
        for j in 1..ny - 1 {
            for i in 1..nx - 1 {
                grad_x[[i, j, k]] = (rho_arr[[i + 1, j, k]] - rho_arr[[i - 1, j, k]]) * dx_inv;
                grad_y[[i, j, k]] = (rho_arr[[i, j + 1, k]] - rho_arr[[i, j - 1, k]]) * dy_inv;
                grad_z[[i, j, k]] = (rho_arr[[i, j, k + 1]] - rho_arr[[i, j, k - 1]]) * dz_inv;
            }
        }
    }

    (grad_x, grad_y, grad_z)
}
