//! Numerical methods for Kuznetsov equation solver

use crate::domain::grid::Grid;
use crate::domain::math::fft::get_fft_for_grid;
use ndarray::{Array3, Zip};
use num_complex::Complex64;
use std::f64::consts::PI;

/// Compute Laplacian using spectral methods
pub fn compute_laplacian(field: &Array3<f64>, grid: &Grid) -> Array3<f64> {
    let fft = get_fft_for_grid(grid.nx, grid.ny, grid.nz);
    let (nx, ny, nz) = (grid.nx, grid.ny, grid.nz);
    let mut field_hat = Array3::<Complex64>::zeros((nx, ny, nz));
    fft.forward_into(field, &mut field_hat);

    // Compute k-space operators
    let kx_max = PI / grid.dx;
    let ky_max = PI / grid.dy;
    let kz_max = PI / grid.dz;

    // Apply Laplacian in k-space: ∇²f = -(kx² + ky² + kz²) * f_hat
    Zip::indexed(&mut field_hat).for_each(|(i, j, k), f| {
        let kx = if i <= nx / 2 {
            i as f64 * 2.0 * kx_max / nx as f64
        } else {
            (i as f64 - nx as f64) * 2.0 * kx_max / nx as f64
        };
        let ky = if j <= ny / 2 {
            j as f64 * 2.0 * ky_max / ny as f64
        } else {
            (j as f64 - ny as f64) * 2.0 * ky_max / ny as f64
        };
        let kz = if k <= nz / 2 {
            k as f64 * 2.0 * kz_max / nz as f64
        } else {
            (k as f64 - nz as f64) * 2.0 * kz_max / nz as f64
        };

        let k_squared = kx * kx + ky * ky + kz * kz;
        *f = -k_squared * *f;
    });

    let mut out = Array3::<f64>::zeros((nx, ny, nz));
    let mut scratch = Array3::<Complex64>::zeros((nx, ny, nz));
    fft.inverse_into(&field_hat, &mut out, &mut scratch);
    out
}

/// Compute gradient using spectral methods
pub fn compute_gradient(
    field: &Array3<f64>,
    grid: &Grid,
) -> (Array3<f64>, Array3<f64>, Array3<f64>) {
    let fft = get_fft_for_grid(grid.nx, grid.ny, grid.nz);
    let (nx, ny, nz) = (grid.nx, grid.ny, grid.nz);
    let mut field_hat = Array3::<Complex64>::zeros((nx, ny, nz));
    fft.forward_into(field, &mut field_hat);

    let kx_max = PI / grid.dx;
    let ky_max = PI / grid.dy;
    let kz_max = PI / grid.dz;

    let mut grad_x_hat = Array3::<Complex64>::zeros((nx, ny, nz));
    let mut grad_y_hat = Array3::<Complex64>::zeros((nx, ny, nz));
    let mut grad_z_hat = Array3::<Complex64>::zeros((nx, ny, nz));

    // Apply gradient operators in k-space
    Zip::indexed(&mut grad_x_hat)
        .and(&mut grad_y_hat)
        .and(&mut grad_z_hat)
        .and(&field_hat)
        .for_each(|(i, j, k), gx, gy, gz, &f| {
            let kx = if i <= nx / 2 {
                i as f64 * 2.0 * kx_max / nx as f64
            } else {
                (i as f64 - nx as f64) * 2.0 * kx_max / nx as f64
            };
            let ky = if j <= ny / 2 {
                j as f64 * 2.0 * ky_max / ny as f64
            } else {
                (j as f64 - ny as f64) * 2.0 * ky_max / ny as f64
            };
            let kz = if k <= nz / 2 {
                k as f64 * 2.0 * kz_max / nz as f64
            } else {
                (k as f64 - nz as f64) * 2.0 * kz_max / nz as f64
            };

            // Gradient in k-space: ∂f/∂x = i*kx*f_hat
            *gx = Complex64::new(0.0, kx) * f;
            *gy = Complex64::new(0.0, ky) * f;
            *gz = Complex64::new(0.0, kz) * f;
        });

    // Transform back to real space
    let mut grad_x_real = Array3::<f64>::zeros((nx, ny, nz));
    let mut grad_y_real = Array3::<f64>::zeros((nx, ny, nz));
    let mut grad_z_real = Array3::<f64>::zeros((nx, ny, nz));
    let mut scratch = Array3::<Complex64>::zeros((nx, ny, nz));
    fft.inverse_into(&grad_x_hat, &mut grad_x_real, &mut scratch);
    fft.inverse_into(&grad_y_hat, &mut grad_y_real, &mut scratch);
    fft.inverse_into(&grad_z_hat, &mut grad_z_real, &mut scratch);

    (grad_x_real, grad_y_real, grad_z_real)
}
