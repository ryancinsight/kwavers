//! Acoustic initial-value problem helpers.
//!
//! # Theorem: Exact staggered velocity start from an initial pressure field
//!
//! For a homogeneous acoustic medium with initial pressure `p0` and zero
//! explicit initial velocity, the exact leapfrog-compatible half-step velocity
//! start is
//!
//! ```text
//!   u(x, -Δt/2) = IFFT( -i k̂ · sin(c₀ |k| Δt / 2) / (ρ₀ c₀) · FFT(p0) )
//! ```
//!
//! The scalar scale factor can be written using the k-space correction factor
//! `κ = cos(0.5 · c₀ · Δt · |k|)` as
//!
//! ```text
//!   sin(c₀ |k| Δt / 2) / (ρ₀ c₀ |k|)
//!   = (Δt / (2ρ₀)) · sinc(arccos(κ))
//! ```
//!
//! with `sinc(x) = sin(x)/x` and the removable singularity `sinc(0) = 1`.
//! This identity avoids materializing a separate `|k|` array when `κ` is already
//! available from the k-space operator.

use crate::core::error::{KwaversError, KwaversResult};
use ndarray::Array3;

/// Compute the spectral IVP velocity scale from the k-space correction factor.
///
/// The result is `Δt / (2ρ₀) * sinc(arccos(κ))` with the `κ = 1` limit handled
/// analytically. This is the scalar multiplier used by the exact initial-pressure
/// velocity start in the PSTD and k-space FDTD solvers.
pub fn spectral_velocity_scale_from_kappa(
    kappa: &Array3<f64>,
    dt: f64,
    rho0_ref: f64,
) -> KwaversResult<Array3<f64>> {
    if !dt.is_finite() || dt <= 0.0 {
        return Err(KwaversError::InvalidInput(format!(
            "dt must be finite and positive, got {dt}"
        )));
    }
    if !rho0_ref.is_finite() || rho0_ref <= 0.0 {
        return Err(KwaversError::InvalidInput(format!(
            "rho0_ref must be finite and positive, got {rho0_ref}"
        )));
    }

    let scale_prefactor = dt / (2.0 * rho0_ref);
    Ok(kappa.mapv(|kap| {
        let theta = kap.clamp(-1.0, 1.0).acos();
        if theta < 1e-30 {
            scale_prefactor
        } else {
            scale_prefactor * theta.sin() / theta
        }
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::Array3;
    use std::f64::consts::FRAC_PI_4;

    #[test]
    fn scale_matches_sinc_identity() {
        let dt = 2.0e-7;
        let rho0_ref = 1000.0;
        let theta = FRAC_PI_4;
        let kappa = Array3::from_elem((2, 2, 2), theta.cos());
        let scale = spectral_velocity_scale_from_kappa(&kappa, dt, rho0_ref).unwrap();
        let expected = dt / (2.0 * rho0_ref) * theta.sin() / theta;
        assert_abs_diff_eq!(scale[[0, 0, 0]], expected, epsilon = 1e-15);
    }

    #[test]
    fn scale_uses_removable_limit_at_zero() {
        let dt = 2.0e-7;
        let rho0_ref = 1000.0;
        let kappa = Array3::from_elem((1, 1, 1), 1.0);
        let scale = spectral_velocity_scale_from_kappa(&kappa, dt, rho0_ref).unwrap();
        assert_abs_diff_eq!(scale[[0, 0, 0]], dt / (2.0 * rho0_ref), epsilon = 1e-18);
    }
}
