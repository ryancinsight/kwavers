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
//! The scalar scale factor can be written using the source-injection phase
//! factor `κ_src = cos(0.5 · c₀ · Δt · |k|)` as
//!
//! ```text
//!   sin(c₀ |k| Δt / 2) / (ρ₀ c₀ |k|)
//!   = (Δt / (2ρ₀)) · sinc(arccos(κ_src))
//! ```
//!
//! with `sinc(x) = sin(x)/x` and the removable singularity `sinc(0) = 1`.
//! This identity avoids materializing a separate `|k|` array when `κ_src` is
//! already available from the source-injection operator.

use kwavers_core::error::{KwaversError, KwaversResult};
use leto::Array3;

/// Compute the spectral IVP velocity scale from the source-injection phase factor.
///
/// The result is `Δt / (2ρ₀) * sinc(arccos(κ_src))` with the `κ_src = 1`
/// limit handled analytically. This is the scalar multiplier used by the exact
/// initial-pressure velocity start in the PSTD and k-space FDTD solvers.
/// # Errors
/// - Returns [`KwaversError::InvalidInput`] if the precondition for invalid or out-of-range input parameters is violated.
///
pub fn spectral_velocity_scale_from_source_kappa(
    source_kappa: &Array3<f64>,
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
    Ok(source_kappa.mapv(|kap| {
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
    use kwavers_core::constants::fundamental::DENSITY_WATER_NOMINAL;
    use leto::Array3;
    use std::f64::consts::FRAC_PI_4;

    #[test]
    fn scale_matches_sinc_identity() {
        let dt = 2.0e-7;
        let rho0_ref = DENSITY_WATER_NOMINAL;
        let theta = FRAC_PI_4;
        let source_kappa = Array3::from_elem([2, 2, 2], theta.cos());
        let scale = spectral_velocity_scale_from_source_kappa(&source_kappa, dt, rho0_ref).unwrap();
        let expected = dt / (2.0 * rho0_ref) * theta.sin() / theta;
        assert_abs_diff_eq!(scale[[0, 0, 0]], expected, epsilon = 1e-15);
    }

    #[test]
    fn scale_uses_removable_limit_at_zero() {
        let dt = 2.0e-7;
        let rho0_ref = DENSITY_WATER_NOMINAL;
        let source_kappa = Array3::from_elem([1, 1, 1], 1.0);
        let scale = spectral_velocity_scale_from_source_kappa(&source_kappa, dt, rho0_ref).unwrap();
        assert_abs_diff_eq!(scale[[0, 0, 0]], dt / (2.0 * rho0_ref), epsilon = 1e-18);
    }
}
