//! Spectral operator for efficient FFT-based derivative computations
//!
//! This module provides a stateful `SpectralOperator` that pre-allocates
//! workspaces and pre-computes wavenumber vectors for efficient spectral
//! derivative calculations.

use crate::fft::{Fft3d, Ifft3d};
use crate::grid::Grid;
use crate::physics::constants::numerical::FFT_K_SCALING;
use ndarray::{Array1, Array3, Zip};
use num_complex::Complex;
use std::f64::consts::PI;

/// Spectral operator for computing derivatives in Fourier space
#[derive(Debug)]
pub struct SpectralOperator {
    /// Grid dimensions
    nx: usize,
    ny: usize,
    nz: usize,

    /// Pre-computed wavenumber vectors
    kx_vec: Array1<f64>,
    ky_vec: Array1<f64>,
    kz_vec: Array1<f64>,

    /// FFT and IFFT operators
    fft: Fft3d,
    ifft: Ifft3d,

    /// Workspace arrays for complex fields
    field_complex: Array3<Complex<f64>>,
    field_hat: Array3<Complex<f64>>,

    /// Workspace arrays for gradient computation
    grad_x_hat: Array3<Complex<f64>>,
    grad_y_hat: Array3<Complex<f64>>,
    grad_z_hat: Array3<Complex<f64>>,
}

impl SpectralOperator {
    /// Create a new spectral operator for the given grid
    pub fn new(grid: &Grid) -> Self {
        let (nx, ny, nz) = (grid.nx, grid.ny, grid.nz);

        // Pre-compute wavenumber vectors
        let kx_max = PI / grid.dx;
        let ky_max = PI / grid.dy;
        let kz_max = PI / grid.dz;

        let kx_vec: Array1<f64> = (0..nx)
            .map(|i| {
                if i <= nx / 2 {
                    i as f64 * FFT_K_SCALING * kx_max / nx as f64
                } else {
                    (i as f64 - nx as f64) * FFT_K_SCALING * kx_max / nx as f64
                }
            })
            .collect();

        let ky_vec: Array1<f64> = (0..ny)
            .map(|j| {
                if j <= ny / 2 {
                    j as f64 * FFT_K_SCALING * ky_max / ny as f64
                } else {
                    (j as f64 - ny as f64) * FFT_K_SCALING * ky_max / ny as f64
                }
            })
            .collect();

        let kz_vec: Array1<f64> = (0..nz)
            .map(|k| {
                if k <= nz / 2 {
                    k as f64 * FFT_K_SCALING * kz_max / nz as f64
                } else {
                    (k as f64 - nz as f64) * FFT_K_SCALING * kz_max / nz as f64
                }
            })
            .collect();

        // Create FFT operators
        let fft = Fft3d::new(nx, ny, nz);
        let ifft = Ifft3d::new(nx, ny, nz);

        // Pre-allocate workspace arrays
        let field_complex = Array3::<Complex<f64>>::zeros((nx, ny, nz));
        let field_hat = Array3::<Complex<f64>>::zeros((nx, ny, nz));
        let grad_x_hat = Array3::<Complex<f64>>::zeros((nx, ny, nz));
        let grad_y_hat = Array3::<Complex<f64>>::zeros((nx, ny, nz));
        let grad_z_hat = Array3::<Complex<f64>>::zeros((nx, ny, nz));

        Self {
            nx,
            ny,
            nz,
            kx_vec,
            ky_vec,
            kz_vec,
            fft,
            ifft,
            field_complex,
            field_hat,
            grad_x_hat,
            grad_y_hat,
            grad_z_hat,
        }
    }

    /// Compute Laplacian using spectral methods with pre-allocated workspace
    pub fn compute_laplacian_workspace(
        &mut self,
        field: &Array3<f64>,
        laplacian_out: &mut Array3<f64>,
        grid: &Grid,
    ) {
        // Convert real field to complex
        Zip::from(&mut self.field_complex)
            .and(field)
            .for_each(|c, &r| *c = Complex::new(r, 0.0));

        // Transform to k-space
        self.field_hat.assign(&self.field_complex);
        self.fft.process(&mut self.field_hat, grid);

        // Apply Laplacian operator in k-space: ∇²f = -(kx² + ky² + kz²) * f_hat
        Zip::indexed(&mut self.field_hat).for_each(|(i, j, k), f| {
            let kx = self.kx_vec[i];
            let ky = self.ky_vec[j];
            let kz = self.kz_vec[k];

            let k_squared = kx * kx + ky * ky + kz * kz;
            *f = -k_squared * *f;
        });

        // Transform back to real space
        let result = self.ifft.process(&mut self.field_hat, grid);
        laplacian_out.assign(&result);
    }

    /// Compute gradient using spectral methods with pre-allocated workspace
    pub fn compute_gradient_workspace(
        &mut self,
        field: &Array3<f64>,
        grad_x_out: &mut Array3<f64>,
        grad_y_out: &mut Array3<f64>,
        grad_z_out: &mut Array3<f64>,
        grid: &Grid,
    ) {
        // Convert real field to complex
        Zip::from(&mut self.field_complex)
            .and(field)
            .for_each(|c, &r| *c = Complex::new(r, 0.0));

        // Transform to k-space
        self.field_hat.assign(&self.field_complex);
        self.fft.process(&mut self.field_hat, grid);

        // Apply gradient operators in k-space: ∂f/∂x = i*kx*f_hat
        Zip::indexed(&mut self.grad_x_hat)
            .and(&mut self.grad_y_hat)
            .and(&mut self.grad_z_hat)
            .and(&self.field_hat)
            .for_each(|(i, j, k), gx, gy, gz, &f| {
                let kx = self.kx_vec[i];
                let ky = self.ky_vec[j];
                let kz = self.kz_vec[k];

                // Gradient in k-space: ∂f/∂x = i*kx*f_hat
                *gx = Complex::new(0.0, kx) * f;
                *gy = Complex::new(0.0, ky) * f;
                *gz = Complex::new(0.0, kz) * f;
            });

        // Transform back to real space
        let grad_x_result = self.ifft.process(&mut self.grad_x_hat, grid);
        let grad_y_result = self.ifft.process(&mut self.grad_y_hat, grid);
        let grad_z_result = self.ifft.process(&mut self.grad_z_hat, grid);

        grad_x_out.assign(&grad_x_result);
        grad_y_out.assign(&grad_y_result);
        grad_z_out.assign(&grad_z_result);
    }
}
