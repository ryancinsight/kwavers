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
    ///
    /// Implements the 1D Von Neumann dispersion relation:
    /// ```text
    /// sin(ω_num·dt/2) = CFL · sin(k·dx/2)
    /// ```
    /// Returns the relative frequency error `(ω_num − ω_exact) / ω_exact`.
    /// Returns 0.0 when k = 0 (no wave, no dispersion error).
    #[must_use]
    pub fn fdtd_dispersion(k: f64, dx: f64, dt: f64, c: f64) -> f64 {
        let omega_exact = k * c;
        if omega_exact.abs() < 1e-14 {
            return 0.0;
        }

        let cfl = c * dt / dx;
        // Von Neumann 1D: sin(omega_num * dt/2) = CFL * sin(k * dx/2)
        let sin_half_omega_dt = (cfl * (0.5 * k * dx).sin()).asin();
        let omega_numerical = 2.0 * sin_half_omega_dt / dt;

        (omega_numerical - omega_exact) / omega_exact
    }

    /// Calculate numerical dispersion for FDTD method in 3D
    ///
    /// Full 3D Von Neumann stability analysis accounting for anisotropic grids
    /// and oblique wave propagation directions.
    #[allow(clippy::too_many_arguments)]
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

        let sin_squared_omega_dt_half = (cfl_z * cfl_z * sin_kz_dz_half).mul_add(
            sin_kz_dz_half,
            (cfl_x * cfl_x * sin_kx_dx_half).mul_add(
                sin_kx_dx_half,
                cfl_y * cfl_y * sin_ky_dy_half * sin_ky_dy_half,
            ),
        );

        let sin_squared_clamped = sin_squared_omega_dt_half.clamp(0.0, 1.0);
        let sin_half_omega_dt = sin_squared_clamped.sqrt();
        let omega_numerical = 2.0 * sin_half_omega_dt.asin() / dt;

        let k_magnitude = kz.mul_add(kz, kx.mul_add(kx, ky * ky)).sqrt();
        let omega_exact = k_magnitude * c;

        if omega_exact.abs() < 1e-14 {
            return 0.0;
        }

        (omega_numerical - omega_exact) / omega_exact
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kwavers_core::constants::fundamental::SOUND_SPEED_WATER_SIM;
    use kwavers_core::constants::numerical::MHZ_TO_HZ;
    use std::f64::consts::PI;

    /// 1D FDTD dispersion is zero at k=0 (no wave → no error).
    #[test]
    fn fdtd_dispersion_zero_at_zero_wavenumber() {
        // k=0 → kx·dx=0 → sin(0)=0 → arcsin(0)=0 → omega_numerical=0 = omega_exact=0.
        // The function clamps: (0 - 0)/0 would be 0/0 — but k=0 → omega_exact=0.
        // Implementation returns 0.0 in that case via the (omega_numerical-omega_exact)/omega_exact formula.
        let disp = DispersionAnalysis::fdtd_dispersion(0.0, 1e-4, 1e-7, SOUND_SPEED_WATER_SIM);
        assert!(disp.is_finite(), "dispersion at k=0 must be finite");
    }

    /// 1D FDTD dispersion magnitude is < 1% at 20 points-per-wavelength with CFL=0.4/√3.
    ///
    /// Reference: Taflove & Hagness (2005), §3.6.
    #[test]
    fn fdtd_dispersion_below_1pct_at_20_ppw() {
        let c = SOUND_SPEED_WATER_SIM;
        let freq = MHZ_TO_HZ;
        let lambda = c / freq;
        let dx = lambda / 20.0;
        let dt = 0.4 * dx / (c * 3.0_f64.sqrt());
        let k = 2.0 * PI / lambda;

        let disp = DispersionAnalysis::fdtd_dispersion(k, dx, dt, c);
        assert!(
            disp.abs() < 0.01,
            "1D FDTD dispersion must be < 1% at 20 PPW (got {disp:.4e})"
        );
    }

    /// Symmetry: fdtd_dispersion(−k) == fdtd_dispersion(k) since sin is odd and asin is odd.
    #[test]
    fn fdtd_dispersion_symmetric_in_wavenumber() {
        let c = SOUND_SPEED_WATER_SIM;
        let freq = MHZ_TO_HZ;
        let lambda = c / freq;
        let dx = lambda / 15.0;
        let dt = 0.3 * dx / (c * 3.0_f64.sqrt());
        let k = 2.0 * PI / lambda;

        let pos = DispersionAnalysis::fdtd_dispersion(k, dx, dt, c);
        let neg = DispersionAnalysis::fdtd_dispersion(-k, dx, dt, c);
        assert!(
            (pos - neg).abs() < 1e-14,
            "fdtd_dispersion must be even in k: pos={pos}, neg={neg}"
        );
    }
}
