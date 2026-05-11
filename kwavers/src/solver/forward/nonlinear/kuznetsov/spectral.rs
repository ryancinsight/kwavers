//! Spectral operator for efficient FFT-based derivative computations
//!
//! This module provides a stateful `SpectralOperator` that pre-allocates
//! workspaces and pre-computes wavenumber vectors for efficient spectral
//! derivative calculations.

use crate::domain::grid::Grid;
use crate::math::fft::{get_fft_for_grid, Fft3d};
use ndarray::{Array1, Array3, Zip};
use num_complex::Complex64;
use std::f64::consts::PI;
use std::sync::Arc;

/// Spectral operator for computing derivatives in Fourier space
#[derive(Debug)]
pub struct SpectralOperator {
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

        // Discrete wavenumbers for an N-point DFT with physical spacing d:
        //   k[i] = 2π·i / (N·d)   for i = 0, …, N/2
        //   k[i] = 2π·(i−N) / (N·d)  for i = N/2+1, …, N−1
        // Equivalently, k[i] = 2·k_Nyquist·i / N  where k_Nyquist = π/d.
        // Note: the factor is 2, not 2π.  Using 2π here inflates each wavenumber
        // by an extra factor of π, causing the spectral Laplacian — and therefore
        // the effective wave speed — to be π² ≈ 9.87× too large.
        let kx_nyquist = PI / grid.dx;
        let ky_nyquist = PI / grid.dy;
        let kz_nyquist = PI / grid.dz;

        let kx_vec: Array1<f64> = (0..nx)
            .map(|i| {
                if i <= nx / 2 {
                    2.0 * kx_nyquist * i as f64 / nx as f64
                } else {
                    2.0 * kx_nyquist * (i as f64 - nx as f64) / nx as f64
                }
            })
            .collect();

        let ky_vec: Array1<f64> = (0..ny)
            .map(|j| {
                if j <= ny / 2 {
                    2.0 * ky_nyquist * j as f64 / ny as f64
                } else {
                    2.0 * ky_nyquist * (j as f64 - ny as f64) / ny as f64
                }
            })
            .collect();

        let kz_vec: Array1<f64> = (0..nz)
            .map(|k| {
                if k <= nz / 2 {
                    2.0 * kz_nyquist * k as f64 / nz as f64
                } else {
                    2.0 * kz_nyquist * (k as f64 - nz as f64) / nz as f64
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
    /// # Panics
    /// - Panics if `kx_vec contiguous`.
    /// - Panics if `ky_vec contiguous`.
    /// - Panics if `kz_vec contiguous`.
    ///
    pub fn compute_laplacian_workspace(
        &mut self,
        field: &Array3<f64>,
        laplacian_out: &mut Array3<f64>,
        _grid: &Grid,
    ) {
        self.fft.forward_into(field, &mut self.field_hat);

        // Apply Laplacian operator in k-space: ∇²f = -(kx² + ky² + kz²) * f_hat
        let kx_s = self.kx_vec.as_slice().expect("kx_vec contiguous");
        let ky_s = self.ky_vec.as_slice().expect("ky_vec contiguous");
        let kz_s = self.kz_vec.as_slice().expect("kz_vec contiguous");
        Zip::indexed(&mut self.field_hat).par_for_each(|(i, j, k), f| {
            let k_sq = kz_s[k].mul_add(kz_s[k], kx_s[i].mul_add(kx_s[i], ky_s[j] * ky_s[j]));
            *f = -k_sq * *f;
        });

        self.fft
            .inverse_into(&self.field_hat, laplacian_out, &mut self.scratch_hat);
    }

    /// Compute gradient using spectral methods with pre-allocated workspace
    /// # Panics
    /// - Panics if `kx_vec contiguous`.
    /// - Panics if `ky_vec contiguous`.
    /// - Panics if `kz_vec contiguous`.
    ///
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
        let kx_s = self.kx_vec.as_slice().expect("kx_vec contiguous");
        let ky_s = self.ky_vec.as_slice().expect("ky_vec contiguous");
        let kz_s = self.kz_vec.as_slice().expect("kz_vec contiguous");
        Zip::indexed(&mut self.grad_x_hat)
            .and(&mut self.grad_y_hat)
            .and(&mut self.grad_z_hat)
            .and(&self.field_hat)
            .par_for_each(|(i, j, k), gx, gy, gz, &f| {
                *gx = Complex64::new(0.0, kx_s[i]) * f;
                *gy = Complex64::new(0.0, ky_s[j]) * f;
                *gz = Complex64::new(0.0, kz_s[k]) * f;
            });

        self.fft
            .inverse_into(&self.grad_x_hat, grad_x_out, &mut self.scratch_hat);
        self.fft
            .inverse_into(&self.grad_y_hat, grad_y_out, &mut self.scratch_hat);
        self.fft
            .inverse_into(&self.grad_z_hat, grad_z_out, &mut self.scratch_hat);
    }
}
