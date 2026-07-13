//! Spectral operator for efficient FFT-based derivative computations
//!
//! This module provides a stateful `KuznetsovSpectralOperator` that pre-allocates
//! workspaces and pre-computes wavenumber vectors for efficient spectral
//! derivative calculations.

use kwavers_grid::Grid;
use kwavers_math::fft::Complex64;
use kwavers_math::fft::{get_fft_for_grid, Fft3d, Fft3dInOutExt};
use leto::Array3;
use leto::{Array1 as LetoArray1, Array3 as LetoArray3};
use std::f64::consts::PI;
use std::sync::Arc;

/// Spectral operator for computing derivatives in Fourier space
#[derive(Debug)]
pub struct KuznetsovSpectralOperator {
    /// Pre-computed wavenumber vectors
    kx_vec: LetoArray1<f64>,
    ky_vec: LetoArray1<f64>,
    kz_vec: LetoArray1<f64>,

    /// FFT and IFFT operators
    fft: Arc<Fft3d>,

    /// Workspace arrays for complex fields
    field_hat: LetoArray3<Complex64>,
    scratch_hat: LetoArray3<Complex64>,

    /// Workspace arrays for gradient computation
    grad_x_hat: LetoArray3<Complex64>,
    grad_y_hat: LetoArray3<Complex64>,
    grad_z_hat: LetoArray3<Complex64>,
}

impl KuznetsovSpectralOperator {
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

        let kx_vec = LetoArray1::from_iter((0..nx).map(|i| {
            if i <= nx / 2 {
                2.0 * kx_nyquist * i as f64 / nx as f64
            } else {
                2.0 * kx_nyquist * (i as f64 - nx as f64) / nx as f64
            }
        }));

        let ky_vec = LetoArray1::from_iter((0..ny).map(|j| {
            if j <= ny / 2 {
                2.0 * ky_nyquist * j as f64 / ny as f64
            } else {
                2.0 * ky_nyquist * (j as f64 - ny as f64) / ny as f64
            }
        }));

        let kz_vec = LetoArray1::from_iter((0..nz).map(|k| {
            if k <= nz / 2 {
                2.0 * kz_nyquist * k as f64 / nz as f64
            } else {
                2.0 * kz_nyquist * (k as f64 - nz as f64) / nz as f64
            }
        }));

        let fft = get_fft_for_grid(nx, ny, nz);

        // Pre-allocate workspace arrays
        let field_hat = LetoArray3::<Complex64>::from_elem([nx, ny, nz], Complex64::default());
        let scratch_hat = LetoArray3::<Complex64>::from_elem([nx, ny, nz], Complex64::default());
        let grad_x_hat = LetoArray3::<Complex64>::from_elem([nx, ny, nz], Complex64::default());
        let grad_y_hat = LetoArray3::<Complex64>::from_elem([nx, ny, nz], Complex64::default());
        let grad_z_hat = LetoArray3::<Complex64>::from_elem([nx, ny, nz], Complex64::default());

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
        let [nx, ny, nz] = self.field_hat.shape();
        let field_leto = LetoArray3::from_shape_vec([nx, ny, nz], field.iter().copied().collect())
            .expect("Kuznetsov field shape must match its Leto FFT shape");
        self.fft.forward_into(&field_leto, &mut self.field_hat);

        // Apply Laplacian operator in k-space: ∇²f = -(kx² + ky² + kz²) * f_hat
        let kx_s = self.kx_vec.as_slice().expect("kx_vec contiguous");
        let ky_s = self.ky_vec.as_slice().expect("ky_vec contiguous");
        let kz_s = self.kz_vec.as_slice().expect("kz_vec contiguous");
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let k_sq =
                        kz_s[k].mul_add(kz_s[k], kx_s[i].mul_add(kx_s[i], ky_s[j] * ky_s[j]));
                    self.field_hat[[i, j, k]] *= -k_sq;
                }
            }
        }

        let mut laplacian = LetoArray3::<f64>::zeros([nx, ny, nz]);
        self.fft
            .inverse_into(&self.field_hat, &mut laplacian, &mut self.scratch_hat);
        copy_leto_real_to_ndarray(&laplacian, laplacian_out);
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
        let [nx, ny, nz] = self.field_hat.shape();
        let field_leto = LetoArray3::from_shape_vec([nx, ny, nz], field.iter().copied().collect())
            .expect("Kuznetsov field shape must match its Leto FFT shape");
        self.fft.forward_into(&field_leto, &mut self.field_hat);

        // Apply gradient operators in k-space: ∂f/∂x = i*kx*f_hat
        let kx_s = self.kx_vec.as_slice().expect("kx_vec contiguous");
        let ky_s = self.ky_vec.as_slice().expect("ky_vec contiguous");
        let kz_s = self.kz_vec.as_slice().expect("kz_vec contiguous");
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let f = self.field_hat[[i, j, k]];
                    self.grad_x_hat[[i, j, k]] = Complex64::new(0.0, kx_s[i]) * f;
                    self.grad_y_hat[[i, j, k]] = Complex64::new(0.0, ky_s[j]) * f;
                    self.grad_z_hat[[i, j, k]] = Complex64::new(0.0, kz_s[k]) * f;
                }
            }
        }

        let grad_x = self.fft.inverse(&self.grad_x_hat);
        let grad_y = self.fft.inverse(&self.grad_y_hat);
        let grad_z = self.fft.inverse(&self.grad_z_hat);
        copy_leto_real_to_ndarray(&grad_x, grad_x_out);
        copy_leto_real_to_ndarray(&grad_y, grad_y_out);
        copy_leto_real_to_ndarray(&grad_z, grad_z_out);
    }
}

fn copy_leto_real_to_ndarray(source: &LetoArray3<f64>, out: &mut Array3<f64>) {
    let [nx, ny, nz] = source.shape();
    debug_assert_eq!(out.shape(), [nx, ny, nz]);
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                out[[i, j, k]] = source[[i, j, k]];
            }
        }
    }
}
