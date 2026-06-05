//! Numerical methods for Kuznetsov equation solver
//!
//! ## Theorem (Fourier spectral Laplacian and gradient)
//!
//! **Statement**: For an N-periodic real grid function `u[i,j,k]` on a uniform
//! grid of spacing `(Δx, Δy, Δz)`, the spectral Laplacian and gradient are:
//!
//! ```text
//! ∇²u = F⁻¹[ −(kₓ² + kᵧ² + k_z²) · Û[m,n,p] ]
//! ∂u/∂x = F⁻¹[ i·kₓ · Û[m,n,p] ]
//! ```
//!
//! where the wavenumber components follow the standard DFT convention:
//! ```text
//! kₓ[m] = 2πm/(Nₓ·Δx)   for m = 0, …, Nₓ/2
//! kₓ[m] = 2π(m−Nₓ)/(Nₓ·Δx) for m = Nₓ/2+1, …, Nₓ−1
//! ```
//! (same for y and z dimensions).
//!
//! **Proof**: Differentiation of `exp(ikₓ·x)` once gives `iₓ·exp(iₓ·x)`;
//! twice gives `−kₓ²·exp(ikₓ·x)`. The DFT decomposes `u` into complex
//! exponentials, differentiation multiplies each Fourier coefficient by the
//! appropriate power of `i·kₓ`, and the IDFT reassembles the result. This is
//! exact (to floating-point rounding) for band-limited signals satisfying the
//! sampling theorem.
//!
//! (Trefethen 2000, §3; Canuto et al. 2006, §2.)
//!
//! ## Reference
//!
//! - Trefethen LN (2000). Spectral Methods in MATLAB. SIAM.
//! - Canuto C et al. (2006). Spectral Methods: Fundamentals in Single Domains.
//!   Springer.

use kwavers_grid::Grid;
use kwavers_math::fft::{get_fft_for_grid, Fft3dInOutExt};
use ndarray::{Array3, Zip};
use num_complex::Complex64;
use std::f64::consts::PI;

/// Compute Laplacian using spectral methods
#[must_use]
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
    Zip::indexed(&mut field_hat).par_for_each(|(i, j, k), f| {
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
        .par_for_each(|(i, j, k), gx, gy, gz, &f| {
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
