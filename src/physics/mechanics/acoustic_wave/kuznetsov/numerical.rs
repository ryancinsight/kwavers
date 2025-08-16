//! Numerical methods for Kuznetsov equation solver

use ndarray::{Array3, Zip};
use num_complex::Complex;
use crate::grid::Grid;
use crate::utils::{fft_3d, ifft_3d};
use std::f64::consts::PI;

/// Compute Laplacian using spectral methods
pub fn compute_laplacian(field: &Array3<f64>, grid: &Grid) -> Array3<f64> {
    // Transform to k-space
    let field_hat = fft_3d(field, grid);
    
    // Compute k-space operators
    let (nx, ny, nz) = (grid.nx, grid.ny, grid.nz);
    let kx_max = PI / grid.dx;
    let ky_max = PI / grid.dy;
    let kz_max = PI / grid.dz;
    
    let mut laplacian_hat = Array3::<Complex<f64>>::zeros((nx, ny, nz));
    
    // Apply Laplacian in k-space: ∇²f = -(kx² + ky² + kz²) * f_hat
    Zip::indexed(&mut laplacian_hat)
        .and(&field_hat)
        .for_each(|(i, j, k), lap, &f| {
            let kx = if i <= nx/2 { 
                i as f64 * 2.0 * kx_max / nx as f64 
            } else { 
                (i as f64 - nx as f64) * 2.0 * kx_max / nx as f64 
            };
            let ky = if j <= ny/2 { 
                j as f64 * 2.0 * ky_max / ny as f64 
            } else { 
                (j as f64 - ny as f64) * 2.0 * ky_max / ny as f64 
            };
            let kz = if k <= nz/2 { 
                k as f64 * 2.0 * kz_max / nz as f64 
            } else { 
                (k as f64 - nz as f64) * 2.0 * kz_max / nz as f64 
            };
            
            let k_squared = kx * kx + ky * ky + kz * kz;
            *lap = -k_squared * f;
        });
    
    // Transform back to real space
    ifft_3d(&laplacian_hat, grid)
}

/// Compute gradient using spectral methods
pub fn compute_gradient(field: &Array3<f64>, grid: &Grid) -> (Array3<f64>, Array3<f64>, Array3<f64>) {
    // Transform to k-space
    let field_hat = fft_3d(field, grid);
    
    let (nx, ny, nz) = (grid.nx, grid.ny, grid.nz);
    let kx_max = PI / grid.dx;
    let ky_max = PI / grid.dy;
    let kz_max = PI / grid.dz;
    
    let mut grad_x_hat = Array3::<Complex<f64>>::zeros((nx, ny, nz));
    let mut grad_y_hat = Array3::<Complex<f64>>::zeros((nx, ny, nz));
    let mut grad_z_hat = Array3::<Complex<f64>>::zeros((nx, ny, nz));
    
    // Apply gradient operators in k-space
    Zip::indexed(&mut grad_x_hat)
        .and(&mut grad_y_hat)
        .and(&mut grad_z_hat)
        .and(&field_hat)
        .for_each(|(i, j, k), gx, gy, gz, &f| {
            let kx = if i <= nx/2 { 
                i as f64 * 2.0 * kx_max / nx as f64 
            } else { 
                (i as f64 - nx as f64) * 2.0 * kx_max / nx as f64 
            };
            let ky = if j <= ny/2 { 
                j as f64 * 2.0 * ky_max / ny as f64 
            } else { 
                (j as f64 - ny as f64) * 2.0 * ky_max / ny as f64 
            };
            let kz = if k <= nz/2 { 
                k as f64 * 2.0 * kz_max / nz as f64 
            } else { 
                (k as f64 - nz as f64) * 2.0 * kz_max / nz as f64 
            };
            
            // Gradient in k-space: ∂f/∂x = i*kx*f_hat
            *gx = Complex::new(0.0, kx) * f;
            *gy = Complex::new(0.0, ky) * f;
            *gz = Complex::new(0.0, kz) * f;
        });
    
    // Transform back to real space
    (
        ifft_3d(&grad_x_hat, grid),
        ifft_3d(&grad_y_hat, grid),
        ifft_3d(&grad_z_hat, grid),
    )
}