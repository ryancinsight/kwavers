//! PSTD Numerical Dispersion Analysis
//!
//! ## Mathematical Foundation
//!
//! PSTD with spectral spatial derivatives and 2nd-order time stepping:
//! ```text
//! ω_num = (2/dt)·arcsin(c·dt·|k|/2)
//! ```
//!
//! ## References
//!
//! - Liu (1997) Microwave Opt. Tech. Lett. 15(3):158-165

use super::DispersionAnalysis;

impl DispersionAnalysis {
    /// Calculate numerical dispersion for PSTD method (1D)
    #[must_use]
    pub fn pstd_dispersion(k: f64, dx: f64, order: usize) -> f64 {
        let kx_dx = k * dx;
        match order {
            2 => 0.02 * kx_dx.powi(2),
            4 => 0.001 * kx_dx.powi(4),
            _ => 0.0,
        }
    }

    /// Calculate numerical dispersion for PSTD method in 3D
    #[allow(clippy::too_many_arguments)]
    #[must_use]
    pub fn pstd_dispersion_3d(
        kx: f64,
        ky: f64,
        kz: f64,
        dx: f64,
        dy: f64,
        dz: f64,
        dt: f64,
        c: f64,
        order: usize,
    ) -> f64 {
        let k_magnitude = kz.mul_add(kz, kx.mul_add(kx, ky * ky)).sqrt();

        let c_dt_k_half = 0.5 * c * dt * k_magnitude;
        let sin_arg = c_dt_k_half.clamp(-1.0, 1.0);
        let omega_numerical = 2.0 * sin_arg.asin() / dt;

        let omega_exact = k_magnitude * c;

        if omega_exact.abs() < 1e-14 {
            return 0.0;
        }

        let base_error = (omega_numerical - omega_exact) / omega_exact;

        let kx_dx = kx * dx;
        let ky_dy = ky * dy;
        let kz_dz = kz * dz;
        let k_h_magnitude = kz_dz
            .mul_add(kz_dz, kx_dx.mul_add(kx_dx, ky_dy * ky_dy))
            .sqrt();

        let anisotropy_correction = match order {
            2 => 0.02 * k_h_magnitude.powi(2),
            4 => 0.001 * k_h_magnitude.powi(4),
            _ => 0.0,
        };

        base_error + anisotropy_correction
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::constants::fundamental::SOUND_SPEED_WATER_SIM;
    use std::f64::consts::PI;

    /// pstd_dispersion order=2 follows the 0.02·(k·dx)² analytical model.
    ///
    /// Reference: Liu (1997) — 2nd-order PSTD phase error ≈ 0.02·(k·Δx)².
    #[test]
    fn pstd_dispersion_order2_matches_analytical_formula() {
        let c = SOUND_SPEED_WATER_SIM;
        let freq = 1e6_f64;
        let lambda = c / freq;
        let dx = lambda / 10.0;
        let k = 2.0 * PI / lambda;
        let kx_dx = k * dx;

        let disp = DispersionAnalysis::pstd_dispersion(k, dx, 2);
        let expected = 0.02 * kx_dx.powi(2);
        assert!(
            (disp - expected).abs() < 1e-15,
            "order-2 dispersion={disp:.6e}, expected={expected:.6e}"
        );
    }

    /// pstd_dispersion order=4 follows the 0.001·(k·dx)⁴ analytical model.
    #[test]
    fn pstd_dispersion_order4_matches_analytical_formula() {
        let c = SOUND_SPEED_WATER_SIM;
        let freq = 1e6_f64;
        let lambda = c / freq;
        let dx = lambda / 8.0;
        let k = 2.0 * PI / lambda;
        let kx_dx = k * dx;

        let disp = DispersionAnalysis::pstd_dispersion(k, dx, 4);
        let expected = 0.001 * kx_dx.powi(4);
        assert!(
            (disp - expected).abs() < 1e-15,
            "order-4 dispersion={disp:.6e}, expected={expected:.6e}"
        );
    }

    /// Unknown order returns 0.0.
    #[test]
    fn pstd_dispersion_unknown_order_returns_zero() {
        let disp = DispersionAnalysis::pstd_dispersion(1.0, 1e-4, 99);
        assert_eq!(disp, 0.0, "unknown order must return 0.0");
    }

    /// 4th-order PSTD has strictly lower dispersion than 2nd-order at same grid resolution.
    #[test]
    fn pstd_order4_lower_than_order2_at_same_resolution() {
        let c = SOUND_SPEED_WATER_SIM;
        let freq = 2e6_f64;
        let lambda = c / freq;
        let dx = lambda / 8.0;
        let k = 2.0 * PI / lambda;

        let d2 = DispersionAnalysis::pstd_dispersion(k, dx, 2);
        let d4 = DispersionAnalysis::pstd_dispersion(k, dx, 4);
        assert!(
            d4.abs() < d2.abs(),
            "order-4 error={d4:.4e} must be smaller than order-2 error={d2:.4e}"
        );
    }
}
