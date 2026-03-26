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
        let k_magnitude = (kx * kx + ky * ky + kz * kz).sqrt();

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
        let k_h_magnitude = (kx_dx * kx_dx + ky_dy * ky_dy + kz_dz * kz_dz).sqrt();

        let anisotropy_correction = match order {
            2 => 0.02 * k_h_magnitude.powi(2),
            4 => 0.001 * k_h_magnitude.powi(4),
            _ => 0.0,
        };

        base_error + anisotropy_correction
    }
}
