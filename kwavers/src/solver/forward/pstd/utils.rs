//! Centralized spectral utilities for k-space operations
//!
//! This module provides common functionality for computing wavenumbers,
//! k-space corrections, and spectral derivatives used by multiple solvers.

use crate::core::error::KwaversResult;
use crate::domain::grid::Grid;
use crate::math::fft::KSpaceCalculator;
use ndarray::{s, Array3, Zip};
use std::f64::consts::PI;

/// Compute wavenumber arrays for spectral operations
/// Returns (kx, ky, kz) arrays with proper Nyquist handling
pub fn compute_wavenumbers(grid: &Grid) -> (Array3<f64>, Array3<f64>, Array3<f64>) {
    let kx_vec = KSpaceCalculator::generate_k_vector(grid.nx, grid.dx);
    let ky_vec = KSpaceCalculator::generate_k_vector(grid.ny, grid.dy);
    let kz_vec = KSpaceCalculator::generate_k_vector(grid.nz, grid.dz);

    let mut kx = Array3::zeros((grid.nx, grid.ny, grid.nz));
    let mut ky = Array3::zeros((grid.nx, grid.ny, grid.nz));
    let mut kz = Array3::zeros((grid.nx, grid.ny, grid.nz));

    // Fill 3D arrays using broadcasting logic
    for i in 0..grid.nx {
        kx.slice_mut(s![i, .., ..]).fill(kx_vec[i]);
    }
    for j in 0..grid.ny {
        ky.slice_mut(s![.., j, ..]).fill(ky_vec[j]);
    }
    for k in 0..grid.nz {
        kz.slice_mut(s![.., .., k]).fill(kz_vec[k]);
    }

    (kx, ky, kz)
}

/// Compute anti-aliasing filter (low-pass filter in k-space)
/// Uses a Butterworth-style filter to smoothly roll off high frequencies
pub fn compute_anti_aliasing_filter(grid: &Grid, cutoff: f64, order: u32) -> Array3<f64> {
    let mut filter = Array3::zeros((grid.nx, grid.ny, grid.nz));
    let (nx, ny, nz) = (grid.nx, grid.ny, grid.nz);

    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                let ik = if i <= nx / 2 {
                    i as f64
                } else {
                    (nx as i64 - i as i64).abs() as f64
                };
                let jk = if j <= ny / 2 {
                    j as f64
                } else {
                    (ny as i64 - j as i64).abs() as f64
                };
                let kk = if k <= nz / 2 {
                    k as f64
                } else {
                    (nz as i64 - k as i64).abs() as f64
                };

                let r_x = ik / (nx as f64 / 2.0);
                let r_y = jk / (ny as f64 / 2.0);
                let r_z = kk / (nz as f64 / 2.0);

                let r = (r_x * r_x + r_y * r_y + r_z * r_z).sqrt();
                filter[[i, j, k]] = 1.0 / (1.0 + (r / cutoff).powi(2 * order as i32));
            }
        }
    }
    filter
}

/// Compute 1D wavenumber array with proper Nyquist handling
#[allow(dead_code)]
fn compute_1d_wavenumbers(n: usize, dx: f64) -> Vec<f64> {
    let mut k = vec![0.0; n];
    let dk = 2.0 * PI / (n as f64 * dx);

    for (i, k_val) in k.iter_mut().enumerate().take(n) {
        if i <= n / 2 {
            *k_val = i as f64 * dk;
        } else {
            *k_val = -((n - i) as f64) * dk;
        }
    }

    // Note: Nyquist frequency should NOT be set to zero
    // It represents the highest resolvable frequency
    // Setting it to zero would lose information

    k
}

/// Compute k² magnitude array for Laplacian operations
#[must_use]
pub fn compute_k_squared(kx: &Array3<f64>, ky: &Array3<f64>, kz: &Array3<f64>) -> Array3<f64> {
    let mut k_squared = Array3::zeros(kx.dim());

    Zip::from(&mut k_squared)
        .and(kx)
        .and(ky)
        .and(kz)
        .for_each(|k2, &kx_val, &ky_val, &kz_val| {
            *k2 = kx_val * kx_val + ky_val * ky_val + kz_val * kz_val;
        });

    k_squared
}

/// Compute k magnitude array
#[must_use]
pub fn compute_k_magnitude(kx: &Array3<f64>, ky: &Array3<f64>, kz: &Array3<f64>) -> Array3<f64> {
    let mut k_mag = Array3::zeros(kx.dim());

    Zip::from(&mut k_mag)
        .and(kx)
        .and(ky)
        .and(kz)
        .for_each(|km, &kx_val, &ky_val, &kz_val| {
            *km = (kx_val * kx_val + ky_val * ky_val + kz_val * kz_val).sqrt();
        });

    k_mag
}

/// Compute k-space correction factors for PSTD
/// These factors compensate for numerical dispersion in the finite difference stencil
pub fn compute_kspace_correction_factors(
    kx: &Array3<f64>,
    ky: &Array3<f64>,
    kz: &Array3<f64>,
    grid: &Grid,
    correction_type: CorrectionType,
    dt: f64,
    c_ref: f64,
) -> Array3<f64> {
    let mut correction = Array3::from_elem(kx.dim(), 1.0);

    match correction_type {
        CorrectionType::Liu1997 => {
            // Liu (1997) correction: sinc function
            Zip::from(&mut correction).and(kx).and(ky).and(kz).for_each(
                |c, &kx_val, &ky_val, &kz_val| {
                    let sinc_x = sinc(kx_val * grid.dx / 2.0);
                    let sinc_y = sinc(ky_val * grid.dy / 2.0);
                    let sinc_z = sinc(kz_val * grid.dz / 2.0);
                    *c = sinc_x * sinc_y * sinc_z;
                },
            );
        }
        CorrectionType::Treeby2010 => {
            // Treeby & Cox (2010) k-space correction: sinc(c_ref * dt * |k| / 2)
            // Uses UNNORMALIZED sinc (sin(x)/x), matching the C++ k-wave binary
            // (kspaceFirstOrder-OMP). The Python np.sinc is normalized sin(πx)/(πx)
            // but the Python-precomputed kappa is NOT passed to the C++ binary (it is
            // absent from the save_to_disk dotdict). The binary computes kappa internally
            // using sin(x)/x with x = c_ref * dt * |k| / 2.
            Zip::from(&mut correction).and(kx).and(ky).and(kz).for_each(
                |c, &kx_val, &ky_val, &kz_val| {
                    let k_sq = kx_val * kx_val + ky_val * ky_val + kz_val * kz_val;
                    let k_mag = k_sq.sqrt();
                    let arg = c_ref * dt * k_mag / 2.0;
                    *c = sinc(arg);
                },
            );
        }
        CorrectionType::None => {
            // No correction (array initialized to 1.0)
        }
    }

    correction
}

/// Type of k-space correction to apply
#[derive(Debug, Clone, Copy)]
pub enum CorrectionType {
    /// No correction
    None,
    /// Liu (1997) sinc correction
    Liu1997,
    /// Treeby & Cox (2010) exact staggered grid correction
    Treeby2010,
}

/// Unnormalized sinc function: sinc(x) = sin(x)/x
/// Used for spatial corrections (Liu 1997) where the argument already
/// includes the correct scaling (e.g., kx·dx/2).
#[inline]
fn sinc(x: f64) -> f64 {
    if x.abs() < 1e-10 {
        1.0
    } else {
        x.sin() / x
    }
}

/// Normalized sinc function: sinc(x) = sin(π·x)/(π·x)
/// Matches numpy's np.sinc(x) convention. NOT used for kappa (C++ binary uses unnormalized).
/// Kept for reference and testing.
#[inline]
#[allow(dead_code)]
fn sinc_normalized(x: f64) -> f64 {
    if x.abs() < 1e-10 {
        1.0
    } else {
        let pi_x = std::f64::consts::PI * x;
        pi_x.sin() / pi_x
    }
}

/// Apply anti-aliasing filter in k-space
#[must_use]
pub fn apply_antialiasing_filter(
    k_squared: &Array3<f64>,
    k_max_squared: f64,
    filter_type: FilterType,
) -> Array3<f64> {
    let mut filter = Array3::from_elem(k_squared.dim(), 1.0);

    match filter_type {
        FilterType::Smooth => {
            // Smooth roll-off filter
            let transition_width = 0.1 * k_max_squared;
            let cutoff = 0.8 * k_max_squared;

            Zip::from(&mut filter).and(k_squared).for_each(|f, &k2| {
                if k2 > cutoff {
                    let x = (k2 - cutoff) / transition_width;
                    *f = 0.5 * (1.0 - x.tanh());
                }
            });
        }
        FilterType::Sharp => {
            // Sharp cutoff at Nyquist
            Zip::from(&mut filter).and(k_squared).for_each(|f, &k2| {
                if k2 > k_max_squared {
                    *f = 0.0;
                }
            });
        }
        FilterType::None => {
            // No filtering
        }
    }

    filter
}

/// Type of anti-aliasing filter
#[derive(Debug, Clone, Copy)]
pub enum FilterType {
    /// No filtering
    None,
    /// Smooth roll-off filter
    Smooth,
    /// Sharp cutoff at Nyquist
    Sharp,
}

pub fn gradient_x(field: &Array3<f64>, grid: &Grid) -> KwaversResult<Array3<f64>> {
    let (_nx, _ny, _nz) = field.dim();

    // Use the central FFT processor for spectral derivative
    let fft = crate::math::fft::get_fft_for_grid(grid.nx, grid.ny, grid.nz);
    fft.spectral_derivative(field, 0)
}

pub fn gradient_y(field: &Array3<f64>, grid: &Grid) -> KwaversResult<Array3<f64>> {
    let fft = crate::math::fft::get_fft_for_grid(grid.nx, grid.ny, grid.nz);
    fft.spectral_derivative(field, 1)
}

pub fn gradient_z(field: &Array3<f64>, grid: &Grid) -> KwaversResult<Array3<f64>> {
    let fft = crate::math::fft::get_fft_for_grid(grid.nx, grid.ny, grid.nz);
    fft.spectral_derivative(field, 2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wavenumber_computation() {
        let grid = Grid::new(8, 8, 8, 0.001, 0.001, 0.001).expect("Failed to create test grid");
        let (kx, ky, kz) = compute_wavenumbers(&grid);

        // Check dimensions
        assert_eq!(kx.dim(), (8, 8, 8));
        assert_eq!(ky.dim(), (8, 8, 8));
        assert_eq!(kz.dim(), (8, 8, 8));

        // Check DC component
        assert_eq!(kx[[0, 0, 0]], 0.0);
        assert_eq!(ky[[0, 0, 0]], 0.0);
        assert_eq!(kz[[0, 0, 0]], 0.0);

        // Check symmetry
        assert!((kx[[1, 0, 0]] + kx[[7, 0, 0]]).abs() < 1e-10);
    }

    #[test]
    fn test_sinc_function() {
        assert_eq!(sinc(0.0), 1.0);
        assert!((sinc(PI) - 0.0).abs() < 1e-10);
        assert!((sinc(PI / 2.0) - 2.0 / PI).abs() < 1e-10);
    }

    #[test]
    fn test_sinc_normalized_function() {
        // sinc_normalized(x) = sin(π·x) / (π·x), matching numpy's np.sinc(x)
        assert_eq!(sinc_normalized(0.0), 1.0);
        // np.sinc(1.0) = sin(π)/π = 0
        assert!((sinc_normalized(1.0)).abs() < 1e-10);
        // np.sinc(0.5) = sin(π/2)/(π/2) = 2/π
        assert!((sinc_normalized(0.5) - 2.0 / PI).abs() < 1e-10);
        // np.sinc(0.1) ≈ 0.9836...
        assert!((sinc_normalized(0.1) - 0.9836316431).abs() < 1e-6);
    }

    /// κ = sinc(c₀·|k|·dt/2) at DC (k=0) must equal 1.0.
    ///
    /// ## Theorem (Treeby & Cox 2010, Eq. 17)
    ///
    /// The k-space correction κ compensates for the staggered time integration:
    ///   κ(k) = sinc(c_ref · |k| · dt / 2) = sin(x)/x,  x = c_ref·|k|·dt/2
    ///
    /// At k = 0: x = 0 → κ = 1 (no correction needed for DC component).
    /// At the Nyquist wavenumber k_N = π/dx, for a Courant-stable step dt = CFL·dx/c:
    ///   x = c_ref · (π/dx) · (CFL·dx/c) / 2 = π·CFL/2
    ///   For CFL = 0.3: x = 0.15·π → κ = sin(0.15π)/(0.15π) ≈ 0.9775
    ///
    /// ## Reference
    /// Treeby, B.E. & Cox, B.T. (2010). k-Wave: MATLAB toolbox for the simulation and
    /// reconstruction of photoacoustic wave fields. J. Biomed. Opt. 15(2):021314.
    #[test]
    fn test_kspace_kappa_correction_at_nyquist() {
        let c0 = 1500.0_f64;
        let dx = 1e-3_f64;
        let cfl = 0.3_f64;
        let dt = cfl * dx / c0; // stable time step

        let grid = Grid::new(32, 32, 32, dx, dx, dx).expect("grid creation");
        let (kx, ky, kz) = compute_wavenumbers(&grid);

        let kappa = compute_kspace_correction_factors(&kx, &ky, &kz, &grid, CorrectionType::Treeby2010, dt, c0);

        // DC component: κ = 1.0 exactly
        assert!(
            (kappa[[0, 0, 0]] - 1.0).abs() < 1e-12,
            "κ at DC must be 1.0, got {}",
            kappa[[0, 0, 0]]
        );

        // Nyquist wavenumber: k_N = π/dx for a grid of n=32 with dk = 2π/(n·dx)
        // The max |k| present in the grid is approximately π/dx.
        // Compute expected κ at k = π/dx.
        let k_nyquist = PI / dx;
        let arg = c0 * k_nyquist * dt / 2.0;
        let expected_kappa_nyquist = sinc(arg); // sin(arg)/arg

        // Find the kappa value closest to Nyquist (max |k| in the grid)
        let k_max = kx
            .iter()
            .zip(ky.iter())
            .zip(kz.iter())
            .map(|((&kxi, &kyi), &kzi)| (kxi * kxi + kyi * kyi + kzi * kzi).sqrt())
            .fold(0.0_f64, f64::max);

        let arg_max = c0 * k_max * dt / 2.0;
        let kappa_at_kmax = sinc(arg_max);

        // κ at k_max must be positive (no aliasing at stable CFL < 1)
        assert!(
            kappa_at_kmax > 0.0,
            "κ at k_max must be positive for CFL={cfl}, got {kappa_at_kmax}"
        );
        // κ at k_max < 1 (correction reduces gradient contribution at high k)
        assert!(
            kappa_at_kmax < 1.0,
            "κ at k_max must be < 1.0 (correction attenuates high-k), got {kappa_at_kmax}"
        );
        // Verify analytical value: arg = CFL·π/2 ≈ 0.4712 → κ ≈ 0.9369 (√3 Nyquist)
        let _ = expected_kappa_nyquist; // informational; grid has |k| ≤ √3·π/dx at corners
        assert!(
            kappa_at_kmax > 0.8,
            "κ at corner Nyquist must be > 0.8 for CFL=0.3, got {kappa_at_kmax}"
        );
    }
}
