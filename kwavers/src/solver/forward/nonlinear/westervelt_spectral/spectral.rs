//! Spectral operations for Westervelt solver

use crate::domain::grid::Grid;
use crate::math::fft::{
    fft_3d_array, fft_3d_array_into, ifft_3d_array, ifft_3d_array_into, Complex64,
};
use ndarray::{Array3, Zip};
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

/// Compute Laplacian using spectral method (allocating convenience wrapper).
///
/// # Theorem: Spectral Laplacian
/// For a periodic function `f` on a uniform grid, the spectral Laplacian is exact:
/// ```text
///   ∇²f = IFFT(−|k|² · FFT(f))
/// ```
/// where `|k|² = kx² + ky² + kz²` stored in `k_squared`.
/// Convergence is exponential in the number of grid points for smooth `f`.
///
/// Allocates two `Array3<Complex64>` per call. For hot-path use, prefer
/// [`compute_laplacian_spectral_into`] which reuses caller-supplied scratch.
#[must_use]
pub fn compute_laplacian_spectral(field: &Array3<f64>, k_squared: &Array3<f64>) -> Array3<f64> {
    // Transform to k-space
    let field_k = fft_3d_array(field);

    // Apply Laplacian in k-space: ∇²f = -k²f
    let laplacian_k = &field_k * &k_squared.mapv(|k2| Complex::new(-k2, 0.0));

    // Transform back to real space
    ifft_3d_array(&laplacian_k)
}

/// Zero-allocation spectral Laplacian via caller-supplied scratch buffers.
///
/// # Theorem: Spectral Laplacian (same as [`compute_laplacian_spectral`])
/// ```text
///   ∇²f = IFFT(−|k|² · FFT(f))
/// ```
///
/// # Algorithm
/// 1. `fft_3d_array_into(field, fft_scratch)` — real→complex DFT in-place into scratch
/// 2. `fft_scratch[i] *= −k_squared[i]` — element-wise Laplacian multiply (parallel)
/// 3. `ifft_3d_array_into(fft_scratch, out)` — complex→real IDFT, real part → `out`
///
/// After return, `fft_scratch` contains the complex IDFT result (overwritten); only
/// `out` carries the valid real Laplacian. `fft_scratch` is safe to reuse.
///
/// # Preconditions
/// - `fft_scratch.dim() == field.dim()`
/// - `out.dim() == field.dim()`
/// - `k_squared.dim() == field.dim()`
/// # Panics
/// - Panics if an internal precondition is violated.
///
pub fn compute_laplacian_spectral_into(
    field: &Array3<f64>,
    k_squared: &Array3<f64>,
    fft_scratch: &mut Array3<Complex64>,
    out: &mut Array3<f64>,
) {
    debug_assert_eq!(fft_scratch.dim(), field.dim(), "fft_scratch shape mismatch");
    debug_assert_eq!(out.dim(), field.dim(), "laplacian output shape mismatch");
    debug_assert_eq!(k_squared.dim(), field.dim(), "k_squared shape mismatch");

    // Step 1: real→complex DFT into scratch (no allocation)
    fft_3d_array_into(field, fft_scratch);

    // Step 2: multiply by −|k|² in-place (Laplacian operator in spectral domain)
    Zip::from(fft_scratch.view_mut())
        .and(k_squared.view())
        .par_for_each(|c, &k2| *c *= Complex64::new(-k2, 0.0));

    // Step 3: IDFT + extract real part into `out` (no allocation)
    ifft_3d_array_into(fft_scratch, out);
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

                let k_squared = kz_val.mul_add(kz_val, kx_val.mul_add(kx_val, ky_val * ky_val));

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

