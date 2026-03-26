//! FDTD Numerical Dispersion Analysis
//!
//! ## Mathematical Foundation
//!
//! ### 1D Von Neumann Dispersion
//! ```text
//! sin(ω_num dt/2) = CFL · sin(k·dx/2)
//! ```
//!
//! ### 3D Von Neumann Dispersion
//! ```text
//! sin²(ω_num·dt/2) = CFL_x²·sin²(kx·dx/2) + CFL_y²·sin²(ky·dy/2) + CFL_z²·sin²(kz·dz/2)
//! ```
//!
//! ### Stability Condition (3D isotropic)
//! ```text
//! c·dt/h ≤ 1/√3 ≈ 0.577
//! ```
//!
//! ## References
//!
//! - Taflove & Hagness (2005) "Computational Electrodynamics" (3rd ed.)
//! - Koene & Robertsson (2012) Geophysics 77(1):T1-T11

use super::DispersionAnalysis;

impl DispersionAnalysis {
    /// Calculate numerical dispersion for FDTD method (1D)
    #[must_use]
    pub fn fdtd_dispersion(k: f64, dx: f64, dt: f64, c: f64) -> f64 {
        let cfl = c * dt / dx;
        let kx_dx = k * dx;

        let sin_half_omega_dt = (cfl * kx_dx.sin()).asin();
        let omega_numerical = 2.0 * sin_half_omega_dt / dt;
        let omega_exact = k * c;

        (omega_numerical - omega_exact) / omega_exact
    }

    /// Calculate numerical dispersion for FDTD method in 3D
    ///
    /// Full 3D Von Neumann stability analysis accounting for anisotropic grids
    /// and oblique wave propagation directions.
    #[must_use]
    pub fn fdtd_dispersion_3d(
        kx: f64,
        ky: f64,
        kz: f64,
        dx: f64,
        dy: f64,
        dz: f64,
        dt: f64,
        c: f64,
    ) -> f64 {
        let cfl_x = c * dt / dx;
        let cfl_y = c * dt / dy;
        let cfl_z = c * dt / dz;

        let sin_kx_dx_half = (0.5 * kx * dx).sin();
        let sin_ky_dy_half = (0.5 * ky * dy).sin();
        let sin_kz_dz_half = (0.5 * kz * dz).sin();

        let sin_squared_omega_dt_half = cfl_x * cfl_x * sin_kx_dx_half * sin_kx_dx_half
            + cfl_y * cfl_y * sin_ky_dy_half * sin_ky_dy_half
            + cfl_z * cfl_z * sin_kz_dz_half * sin_kz_dz_half;

        let sin_squared_clamped = sin_squared_omega_dt_half.clamp(0.0, 1.0);
        let sin_half_omega_dt = sin_squared_clamped.sqrt();
        let omega_numerical = 2.0 * sin_half_omega_dt.asin() / dt;

        let k_magnitude = (kx * kx + ky * ky + kz * kz).sqrt();
        let omega_exact = k_magnitude * c;

        if omega_exact.abs() < 1e-14 {
            return 0.0;
        }

        (omega_numerical - omega_exact) / omega_exact
    }
}
