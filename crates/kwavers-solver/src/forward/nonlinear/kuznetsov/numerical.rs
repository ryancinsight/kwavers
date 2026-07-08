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
use leto::Array3 as LetoArray3;
use ndarray::Array3;
use kwavers_math::fft::Complex64;
use std::f64::consts::PI;

/// Compute Laplacian using spectral methods
#[must_use]
pub fn compute_laplacian(field: &Array3<f64>, grid: &Grid) -> Array3<f64> {
    let fft = get_fft_for_grid(grid.nx, grid.ny, grid.nz);
    let (nx, ny, nz) = (grid.nx, grid.ny, grid.nz);
    let field_leto = LetoArray3::from_shape_vec([nx, ny, nz], field.iter().copied().collect())
        .expect("Kuznetsov field shape must match its Leto FFT shape");
    let mut field_hat = LetoArray3::<Complex64>::from_elem([nx, ny, nz], Complex64::default());
    fft.forward_into(&field_leto, &mut field_hat);

    // Compute k-space operators
    let kx_max = PI / grid.dx;
    let ky_max = PI / grid.dy;
    let kz_max = PI / grid.dz;

    // Apply Laplacian in k-space: ∇²f = -(kx² + ky² + kz²) * f_hat
    for i in 0..nx {
        let kx = if i <= nx / 2 {
            i as f64 * 2.0 * kx_max / nx as f64
        } else {
            (i as f64 - nx as f64) * 2.0 * kx_max / nx as f64
        };
        for j in 0..ny {
            let ky = if j <= ny / 2 {
                j as f64 * 2.0 * ky_max / ny as f64
            } else {
                (j as f64 - ny as f64) * 2.0 * ky_max / ny as f64
            };
            for k in 0..nz {
                let kz = if k <= nz / 2 {
                    k as f64 * 2.0 * kz_max / nz as f64
                } else {
                    (k as f64 - nz as f64) * 2.0 * kz_max / nz as f64
                };
                let k_squared = kx * kx + ky * ky + kz * kz;
                field_hat[[i, j, k]] *= -k_squared;
            }
        }
    }

    Array3::from_shape_vec((nx, ny, nz), fft.inverse(&field_hat).into_vec())
        .expect("Kuznetsov Laplacian output shape must match the solver grid")
}

/// Compute gradient using spectral methods
pub fn compute_gradient(
    field: &Array3<f64>,
    grid: &Grid,
) -> (Array3<f64>, Array3<f64>, Array3<f64>) {
    let fft = get_fft_for_grid(grid.nx, grid.ny, grid.nz);
    let (nx, ny, nz) = (grid.nx, grid.ny, grid.nz);
    let field_leto = LetoArray3::from_shape_vec([nx, ny, nz], field.iter().copied().collect())
        .expect("Kuznetsov field shape must match its Leto FFT shape");
    let field_hat = fft.forward(&field_leto);

    let kx_max = PI / grid.dx;
    let ky_max = PI / grid.dy;
    let kz_max = PI / grid.dz;

    let mut grad_x_hat = LetoArray3::<Complex64>::from_elem([nx, ny, nz], Complex64::default());
    let mut grad_y_hat = LetoArray3::<Complex64>::from_elem([nx, ny, nz], Complex64::default());
    let mut grad_z_hat = LetoArray3::<Complex64>::from_elem([nx, ny, nz], Complex64::default());

    // Apply gradient operators in k-space
    for i in 0..nx {
        let kx = if i <= nx / 2 {
            i as f64 * 2.0 * kx_max / nx as f64
        } else {
            (i as f64 - nx as f64) * 2.0 * kx_max / nx as f64
        };
        for j in 0..ny {
            let ky = if j <= ny / 2 {
                j as f64 * 2.0 * ky_max / ny as f64
            } else {
                (j as f64 - ny as f64) * 2.0 * ky_max / ny as f64
            };
            for k in 0..nz {
                let kz = if k <= nz / 2 {
                    k as f64 * 2.0 * kz_max / nz as f64
                } else {
                    (k as f64 - nz as f64) * 2.0 * kz_max / nz as f64
                };
                let f = field_hat[[i, j, k]];
                grad_x_hat[[i, j, k]] = Complex64::new(0.0, kx) * f;
                grad_y_hat[[i, j, k]] = Complex64::new(0.0, ky) * f;
                grad_z_hat[[i, j, k]] = Complex64::new(0.0, kz) * f;
            }
        }
    }

    // Transform back to real space
    let grad_x_real = Array3::from_shape_vec((nx, ny, nz), fft.inverse(&grad_x_hat).into_vec())
        .expect("Kuznetsov x-gradient output shape must match the solver grid");
    let grad_y_real = Array3::from_shape_vec((nx, ny, nz), fft.inverse(&grad_y_hat).into_vec())
        .expect("Kuznetsov y-gradient output shape must match the solver grid");
    let grad_z_real = Array3::from_shape_vec((nx, ny, nz), fft.inverse(&grad_z_hat).into_vec())
        .expect("Kuznetsov z-gradient output shape must match the solver grid");

    (grad_x_real, grad_y_real, grad_z_real)
}
