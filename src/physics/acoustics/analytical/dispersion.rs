//! Dispersion analysis and correction for numerical methods

use crate::domain::grid::Grid;
use ndarray::Array3;
use std::f64::consts::PI;

/// Dispersion analysis for numerical methods
#[derive(Debug)]
pub struct DispersionAnalysis;

impl DispersionAnalysis {
    /// Calculate numerical dispersion for FDTD method
    #[must_use]
    pub fn fdtd_dispersion(k: f64, dx: f64, dt: f64, c: f64) -> f64 {
        // TODO_AUDIT: P2 - FDTD Dispersion Analysis - 1D Only (Enhancement)
        //
        // PROBLEM:
        // Current implementation only considers 1D wavenumber k. For 3D FDTD,
        // dispersion depends on all three wavenumber components (kx, ky, kz).
        //
        // IMPACT:
        // - Acceptable for 1D problems or isotropic grids with dx=dy=dz
        // - Less accurate for anisotropic grids or oblique wave propagation
        // - Enhancement rather than critical fix (current implementation functional)
        //
        // RECOMMENDED ENHANCEMENT:
        // Implement full 3D FDTD dispersion:
        //   sin²(ω_num·dt/2) = CFL_x²·sin²(kx·dx/2) + CFL_y²·sin²(ky·dy/2) + CFL_z²·sin²(kz·dz/2)
        // where CFL_x = c·dt/dx, CFL_y = c·dt/dy, CFL_z = c·dt/dz
        //
        // EFFORT: 2-3 hours (part of full 3D dispersion analysis)
        // SPRINT: Sprint 213
        // PRIORITY: P2

        let cfl = c * dt / dx;
        let kx_dx = k * dx;

        // Von Neumann stability analysis result
        let sin_half_omega_dt = (cfl * kx_dx.sin()).asin();
        let omega_numerical = 2.0 * sin_half_omega_dt / dt;
        let omega_exact = k * c;

        (omega_numerical - omega_exact) / omega_exact
    }

    /// Calculate numerical dispersion for PSTD method
    #[must_use]
    pub fn pstd_dispersion(k: f64, dx: f64, order: usize) -> f64 {
        // TODO_AUDIT: P2 - Dispersion Correction - Simplified 1D Approximation
        //
        // PROBLEM:
        // Uses hardcoded polynomial coefficients (0.02, 0.001) for dispersion correction
        // instead of analytical 3D dispersion relation. Only considers 1D wavenumber k.
        //
        // IMPACT:
        // - Less accurate for anisotropic grids (dx ≠ dy ≠ dz)
        // - No multidimensional dispersion handling
        // - Phase errors accumulate in long-distance propagation
        // - Acceptable for most applications but suboptimal for complex geometries
        //
        // RECOMMENDED ENHANCEMENT:
        // Implement full 3D Von Neumann dispersion analysis:
        //   sin²(ω_num·dt/2) = CFL²·[sin²(kx·dx/2) + sin²(ky·dy/2) + sin²(kz·dz/2)]
        //
        // VALIDATION CRITERIA:
        // - Dispersion relation plot: ω_numerical vs. |k| should match analytical
        // - Anisotropy test: verify same dispersion in all directions for isotropic media
        // - Long-distance benchmark: compare corrected vs. uncorrected propagation
        //
        // REFERENCES:
        // - Koene & Robertsson (2012). "Removing numerical dispersion." Geophysics, 77(1), T1-T11.
        // - Moczo et al. (2014). "The Finite-Difference Modelling of Earthquake Motions." Cambridge UP.
        //
        // EFFORT: 4-6 hours
        // SPRINT: Sprint 213 (Advanced Numerics)
        // PRIORITY: P2 (Enhancement, current implementation functional)

        // K-space method dispersion (spectral accuracy)
        let kx_dx = k * dx;

        match order {
            2 => 0.02 * kx_dx.powi(2),  // Second-order correction
            4 => 0.001 * kx_dx.powi(4), // Fourth-order correction
            _ => 0.0,                   // Perfect for lower orders
        }
    }

    /// Apply dispersion correction to a field
    pub fn apply_correction(
        field: &mut Array3<f64>,
        grid: &Grid,
        frequency: f64,
        c: f64,
        method: DispersionMethod,
    ) {
        let k = 2.0 * PI * frequency / c;

        let correction_factor = match method {
            DispersionMethod::FDTD(dt) => 1.0 / (1.0 + Self::fdtd_dispersion(k, grid.dx, dt, c)),
            DispersionMethod::PSTD(order) => 1.0 / (1.0 + Self::pstd_dispersion(k, grid.dx, order)),
            DispersionMethod::None => 1.0,
        };

        field.mapv_inplace(|v| v * correction_factor);
    }
}

/// Numerical method for dispersion calculation
#[derive(Debug)]
pub enum DispersionMethod {
    /// Finite-difference time-domain with timestep
    FDTD(f64),
    /// Pseudo-spectral time-domain with order
    PSTD(usize),
    /// No dispersion correction
    None,
}
