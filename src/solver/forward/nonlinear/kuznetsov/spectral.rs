//! Spectral operator for efficient FFT-based derivative computations
//!
//! This module provides a stateful `SpectralOperator` that pre-allocates
//! workspaces and pre-computes wavenumber vectors for efficient spectral
//! derivative calculations.

use crate::core::constants::numerical::FFT_K_SCALING;
use crate::domain::grid::Grid;
use crate::math::fft::{get_fft_for_grid, Fft3d};
use ndarray::{Array1, Array3, Zip};
use num_complex::Complex64;
use std::f64::consts::PI;
use std::sync::Arc;

/// Spectral operator for computing derivatives in Fourier space
#[derive(Debug)]
pub struct SpectralOperator {
    /// Grid dimensions
    #[allow(dead_code)] // Grid dimensions for spectral operations
    nx: usize,
    #[allow(dead_code)]
    ny: usize,
    #[allow(dead_code)]
    nz: usize,

    /// Pre-computed wavenumber vectors
    kx_vec: Array1<f64>,
    ky_vec: Array1<f64>,
    kz_vec: Array1<f64>,

    /// FFT and IFFT operators
    fft: Arc<Fft3d>,

    /// Workspace arrays for complex fields
    field_hat: Array3<Complex64>,
    scratch_hat: Array3<Complex64>,

    /// Workspace arrays for gradient computation
    grad_x_hat: Array3<Complex64>,
    grad_y_hat: Array3<Complex64>,
    grad_z_hat: Array3<Complex64>,
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

        let fft = get_fft_for_grid(nx, ny, nz);

        // Pre-allocate workspace arrays
        let field_hat = Array3::<Complex64>::zeros((nx, ny, nz));
        let scratch_hat = Array3::<Complex64>::zeros((nx, ny, nz));
        let grad_x_hat = Array3::<Complex64>::zeros((nx, ny, nz));
        let grad_y_hat = Array3::<Complex64>::zeros((nx, ny, nz));
        let grad_z_hat = Array3::<Complex64>::zeros((nx, ny, nz));

        Self {
            nx,
            ny,
            nz,
            kx_vec,
            ky_vec,
            kz_vec,
            fft,
            field_hat,
            scratch_hat,
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
        _grid: &Grid,
    ) {
        self.fft.forward_into(field, &mut self.field_hat);

        // Apply Laplacian operator in k-space: ∇²f = -(kx² + ky² + kz²) * f_hat
        Zip::indexed(&mut self.field_hat).for_each(|(i, j, k), f| {
            let kx = self.kx_vec[i];
            let ky = self.ky_vec[j];
            let kz = self.kz_vec[k];

            let k_squared = kx * kx + ky * ky + kz * kz;
            *f = -k_squared * *f;
        });

        self.fft
            .inverse_into(&self.field_hat, laplacian_out, &mut self.scratch_hat);
    }

    /// Compute gradient using spectral methods with pre-allocated workspace
    pub fn compute_gradient_workspace(
        &mut self,
        field: &Array3<f64>,
        grad_x_out: &mut Array3<f64>,
        grad_y_out: &mut Array3<f64>,
        grad_z_out: &mut Array3<f64>,
        _grid: &Grid,
    ) {
        self.fft.forward_into(field, &mut self.field_hat);

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
                *gx = Complex64::new(0.0, kx) * f;
                *gy = Complex64::new(0.0, ky) * f;
                *gz = Complex64::new(0.0, kz) * f;
            });

        self.fft
            .inverse_into(&self.grad_x_hat, grad_x_out, &mut self.scratch_hat);
        self.fft
            .inverse_into(&self.grad_y_hat, grad_y_out, &mut self.scratch_hat);
        self.fft
            .inverse_into(&self.grad_z_hat, grad_z_out, &mut self.scratch_hat);
    }
}
