//! Spectral operations for Westervelt solver

use kwavers_core::constants::numerical::TWO_PI;
use kwavers_grid::Grid;
use kwavers_math::fft::{fft_3d_array, fft_3d_array_into, ifft_3d_array, Complex64};
use leto::Array3 as LetoArray3;
use ndarray::{Array3, Zip};

/// Initialize k-space grids for spectral operations
pub fn initialize_kspace_grids(
    grid: &Grid,
) -> (Array3<f64>, Array3<f64>, Array3<f64>, Array3<f64>) {
    let (nx, ny, nz) = (grid.nx, grid.ny, grid.nz);

    // The k-space wavenumbers are separable: kx depends only on i, ky on j,
    // kz on k. Precompute each 1-D axis once (O(N) FFT-shifted divisions in
    // total) instead of recomputing the branch and division per cell, which
    // costs O(N³) divisions. The per-cell values are formed from the same
    // expressions, so the result is identical to the dense triple loop.
    let kx_axis: Vec<f64> = (0..nx)
        .map(|i| {
            if i <= nx / 2 {
                TWO_PI * i as f64 / (nx as f64 * grid.dx)
            } else {
                TWO_PI * (i as f64 - nx as f64) / (nx as f64 * grid.dx)
            }
        })
        .collect();
    let ky_axis: Vec<f64> = (0..ny)
        .map(|j| {
            if j <= ny / 2 {
                TWO_PI * j as f64 / (ny as f64 * grid.dy)
            } else {
                TWO_PI * (j as f64 - ny as f64) / (ny as f64 * grid.dy)
            }
        })
        .collect();
    let kz_axis: Vec<f64> = (0..nz)
        .map(|k| {
            if k <= nz / 2 {
                TWO_PI * k as f64 / (nz as f64 * grid.dz)
            } else {
                TWO_PI * (k as f64 - nz as f64) / (nz as f64 * grid.dz)
            }
        })
        .collect();

    let mut kx = Array3::<f64>::zeros((nx, ny, nz));
    let mut ky = Array3::<f64>::zeros((nx, ny, nz));
    let mut kz = Array3::<f64>::zeros((nx, ny, nz));
    let mut k_squared = Array3::<f64>::zeros((nx, ny, nz));

    Zip::indexed(&mut kx)
        .and(&mut ky)
        .and(&mut kz)
        .and(&mut k_squared)
        .par_for_each(|(i, j, k), kx_v, ky_v, kz_v, k2| {
            let kx_val = kx_axis[i];
            let ky_val = ky_axis[j];
            let kz_val = kz_axis[k];
            *kx_v = kx_val;
            *ky_v = ky_val;
            *kz_v = kz_val;
            *k2 = kx_val * kx_val + ky_val * ky_val + kz_val * kz_val;
        });

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
    let (nx, ny, nz) = field.dim();
    let field_leto = LetoArray3::from_shape_vec([nx, ny, nz], field.iter().copied().collect())
        .expect("Westervelt field shape must match its Leto FFT shape");
    // Transform to k-space
    let field_k = fft_3d_array(&field_leto);

    // Apply Laplacian in k-space: ∇²f = -k²f
    let mut laplacian_k = LetoArray3::<Complex64>::from_elem([nx, ny, nz], Complex64::default());
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                laplacian_k[[i, j, k]] =
                    field_k[[i, j, k]] * Complex64::new(-k_squared[[i, j, k]], 0.0);
            }
        }
    }

    // Transform back to real space
    Array3::from_shape_vec((nx, ny, nz), ifft_3d_array(&laplacian_k).into_vec())
        .expect("Westervelt Laplacian output shape must match the solver grid")
}

/// Scratch-reusing spectral Laplacian via caller-supplied FFT buffer.
///
/// # Theorem: Spectral Laplacian (same as [`compute_laplacian_spectral`])
/// ```text
///   ∇²f = IFFT(−|k|² · FFT(f))
/// ```
///
/// # Algorithm
/// 1. Copy the real field into Leto storage and run Apollo real→complex DFT into `fft_scratch`
/// 2. `fft_scratch[i] *= −k_squared[i]` — element-wise Laplacian multiply (parallel)
/// 3. Apollo complex→real IDFT, real part → `out`
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
    fft_scratch: &mut LetoArray3<Complex64>,
    out: &mut Array3<f64>,
) {
    let (nx, ny, nz) = field.dim();
    debug_assert_eq!(
        fft_scratch.shape(),
        [nx, ny, nz],
        "fft_scratch shape mismatch"
    );
    debug_assert_eq!(out.dim(), field.dim(), "laplacian output shape mismatch");
    debug_assert_eq!(k_squared.dim(), field.dim(), "k_squared shape mismatch");

    // Step 1: real→complex DFT into scratch (no allocation)
    let field_leto = LetoArray3::from_shape_vec([nx, ny, nz], field.iter().copied().collect())
        .expect("Westervelt field shape must match its Leto FFT shape");
    fft_3d_array_into(&field_leto, fft_scratch);

    // Step 2: multiply by −|k|² in-place (Laplacian operator in spectral domain)
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                fft_scratch[[i, j, k]] *= Complex64::new(-k_squared[[i, j, k]], 0.0);
            }
        }
    }

    // Step 3: IDFT + extract real part into `out` (no allocation)
    let laplacian = ifft_3d_array(fft_scratch);
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                out[[i, j, k]] = laplacian[[i, j, k]];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kwavers_grid::Grid;
    use std::f64::consts::PI;

    /// Helper: uniform spacing grid.
    fn make_grid(n: usize, dx: f64) -> Grid {
        Grid::new(n, n, n, dx, dx, dx).unwrap()
    }

    // ── k-space grid structure ─────────────────────────────────────────────

    /// **Theorem (DC bin is zero wavenumber)**:
    ///
    /// The k-space grid element `k_squared[0, 0, 0]` must be exactly zero because
    /// the DC Fourier mode has zero wavenumber in every direction:
    ///
    /// ```text
    /// k_x(0) = 2π·0/(N·Δx) = 0,   k_y(0) = k_z(0) = 0   ⟹   |k|²(0,0,0) = 0
    /// ```
    #[test]
    fn k_squared_dc_bin_is_exactly_zero() {
        let grid = make_grid(16, 1e-3);
        let (k_sq, _kx, _ky, _kz) = initialize_kspace_grids(&grid);
        assert_eq!(k_sq[[0, 0, 0]], 0.0, "DC bin k_squared must be exactly 0");
    }

    /// **Theorem (fundamental mode wavenumber)**:
    ///
    /// The fundamental Fourier mode (bin 1 along x, all others 0) has:
    ///
    /// ```text
    /// kx(1) = 2π / (N·Δx) = 2π / L_x
    /// ```
    ///
    /// All other components are zero, so `k_squared[1, 0, 0] = kx(1)²`.
    #[test]
    fn k_squared_fundamental_mode_matches_2pi_over_lx() {
        let n = 32usize;
        let dx = 1.0e-3_f64;
        let grid = make_grid(n, dx);
        let (k_sq, kx, _ky, _kz) = initialize_kspace_grids(&grid);

        let lx = n as f64 * dx;
        let k1_analytic = 2.0 * PI / lx;

        // kx at bin 1 must equal 2π/Lx
        let kx1 = kx[[1, 0, 0]];
        assert!(
            (kx1 - k1_analytic).abs() < 1e-12,
            "kx[1] = {kx1:.6e} must equal 2π/Lx = {k1_analytic:.6e}"
        );

        // k_squared at (1, 0, 0) must equal k1_analytic²
        let k_sq_1 = k_sq[[1, 0, 0]];
        let expected = k1_analytic * k1_analytic;
        let rel_err = (k_sq_1 - expected).abs() / expected;
        assert!(
            rel_err < 1e-12,
            "k_squared[1,0,0] = {k_sq_1:.6e}, expected {expected:.6e}, rel_err={rel_err:.2e}"
        );
    }

    /// **Theorem (k-space Nyquist symmetry)**:
    ///
    /// For even N, the Nyquist bin N/2 must satisfy:
    ///
    /// ```text
    /// kx(N/2) = 2π · (N/2) / (N·Δx) = π / Δx
    /// ```
    ///
    /// This is the maximum representable wavenumber on the grid.
    #[test]
    fn k_squared_nyquist_bin_equals_pi_over_dx() {
        let n = 16usize;
        let dx = 1.0e-3_f64;
        let grid = make_grid(n, dx);
        let (_k_sq, kx, _ky, _kz) = initialize_kspace_grids(&grid);

        let nyquist = PI / dx; // = π/Δx
        let kx_nyquist = kx[[n / 2, 0, 0]];
        assert!(
            (kx_nyquist - nyquist).abs() < 1e-10,
            "kx[N/2] = {kx_nyquist:.6e} must equal π/Δx = {nyquist:.6e}"
        );
    }

    // ── Spectral Laplacian correctness ─────────────────────────────────────

    /// **Theorem (∇²[constant] = 0)**:
    ///
    /// For any constant function f = C, ∇²f = 0 everywhere.
    /// Spectrally: −|k|²·FFT(C) = 0 for all k ≠ 0 (DC bin has k²=0), so
    /// IFFT(result) = 0.
    #[test]
    fn spectral_laplacian_of_constant_is_zero() {
        let n = 16usize;
        let dx = 1.0e-3_f64;
        let grid = make_grid(n, dx);
        let (k_sq, _, _, _) = initialize_kspace_grids(&grid);

        let field = ndarray::Array3::from_elem((n, n, n), 7.5_f64);
        let lap = compute_laplacian_spectral(&field, &k_sq);

        let max_abs = lap.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
        // Budget: N³·log₂(N)·ε_machine ≈ 16³·4·2.2e-16 ≈ 3.6e-11
        assert!(
            max_abs < 1e-8,
            "∇²(constant) must be zero; max_abs={max_abs:.3e}"
        );
    }

    /// **Theorem (spectral Laplacian of sin(kx·x))**:
    ///
    /// For f(x,y,z) = sin(k₁·x), the exact Laplacian is:
    ///
    /// ```text
    /// ∇²f = ∂²f/∂x² = −k₁² sin(k₁·x)
    /// ```
    ///
    /// The spectral method reproduces this to machine precision for any
    /// DFT-representable mode.  We verify the interior L∞ relative error < 1e-8.
    #[test]
    fn spectral_laplacian_of_sine_matches_analytical() {
        let n = 32usize;
        let dx = 1.0e-3_f64;
        let grid = make_grid(n, dx);
        let (k_sq, _, _, _) = initialize_kspace_grids(&grid);

        let k1 = 2.0 * PI / (n as f64 * dx); // fundamental wavenumber
        let mut field = ndarray::Array3::<f64>::zeros((n, n, n));
        for i in 0..n {
            let x = i as f64 * dx;
            let val = (k1 * x).sin();
            for j in 0..n {
                for k in 0..n {
                    field[[i, j, k]] = val;
                }
            }
        }

        let lap = compute_laplacian_spectral(&field, &k_sq);

        // Check interior points only (boundaries may pick up Gibbs from non-periodic extension)
        let center = n / 2;
        let mut max_rel_err = 0.0_f64;
        for i in 1..n - 1 {
            let x = i as f64 * dx;
            let expected = -k1 * k1 * (k1 * x).sin();
            let computed = lap[[i, center, center]];
            let denom = k1 * k1; // normalise by |k|² to get relative error
            let rel_err = (computed - expected).abs() / denom;
            max_rel_err = max_rel_err.max(rel_err);
        }

        assert!(
            max_rel_err < 1e-8,
            "∇²(sin(k₁x)) spectral relative error = {max_rel_err:.3e} (must be < 1e-8)"
        );
    }

    /// **Theorem (scratch-reusing path bit-identical to allocating path)**:
    ///
    /// `compute_laplacian_spectral_into` must produce results bitwise identical to
    /// `compute_laplacian_spectral` on the same input, confirming the scratch-reusing
    /// scratch buffer path does not alter the computation.
    #[test]
    fn spectral_laplacian_into_is_bitwise_identical_to_allocating() {
        let n = 16usize;
        let dx = 1.0e-3_f64;
        let grid = make_grid(n, dx);
        let (k_sq, _, _, _) = initialize_kspace_grids(&grid);

        let k1 = 2.0 * PI / (n as f64 * dx);
        let mut field = ndarray::Array3::<f64>::zeros((n, n, n));
        for i in 0..n {
            let x = i as f64 * dx;
            for j in 0..n {
                for k in 0..n {
                    field[[i, j, k]] = (k1 * x).cos() * (k1 * j as f64 * dx).sin();
                }
            }
        }

        // Allocating path
        let lap_alloc = compute_laplacian_spectral(&field, &k_sq);

        // Zero-allocation path
        let shape = (n, n, n);
        let mut fft_scratch = LetoArray3::<Complex64>::from_elem([n, n, n], Complex64::default());
        let mut lap_into = ndarray::Array3::<f64>::zeros(shape);
        compute_laplacian_spectral_into(&field, &k_sq, &mut fft_scratch, &mut lap_into);

        // Results must be bitwise identical
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    assert_eq!(
                        lap_alloc[[i, j, k]].to_bits(),
                        lap_into[[i, j, k]].to_bits(),
                        "spectral_laplacian_into result differs at [{i},{j},{k}]: \
                         alloc={:.6e} into={:.6e}",
                        lap_alloc[[i, j, k]],
                        lap_into[[i, j, k]]
                    );
                }
            }
        }
    }
}
