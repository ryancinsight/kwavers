//! CBS scattering-potential identities.
//!
//! # Contracts
//!
//! For reference slowness `s0`, model slowness `s`, and angular frequency
//! `omega`, the real Helmholtz scattering potential is
//! `V = omega^2 (s^2 - s0^2)`.
//!
//! The shifted CBS potential is `V_s = V - i epsilon`, with
//! `epsilon >= ||V||_infinity`. This gives every voxel a nonzero imaginary
//! component, so pointwise CBS preconditioners remain finite.

use crate::core::error::{KwaversError, KwaversResult};
use ndarray::Array3;
use num_complex::Complex64;

/// Compute the real Helmholtz scattering potential.
///
/// # Errors
/// Returns an error if frequency or slowness values are outside the physical
/// finite positive domain.
pub fn real_scattering_potential(
    omega_rad_s: f64,
    slowness_s_per_m: &Array3<f64>,
    reference_slowness_s_per_m: f64,
) -> KwaversResult<Vec<f64>> {
    if !omega_rad_s.is_finite() || omega_rad_s <= 0.0 {
        return Err(KwaversError::InvalidInput(format!(
            "CBS omega must be positive and finite, got {omega_rad_s}"
        )));
    }
    if !reference_slowness_s_per_m.is_finite() || reference_slowness_s_per_m <= 0.0 {
        return Err(KwaversError::InvalidInput(format!(
            "CBS reference slowness must be positive and finite, got {reference_slowness_s_per_m}"
        )));
    }
    if slowness_s_per_m.is_empty() {
        return Err(KwaversError::InvalidInput(
            "CBS scattering potential requires a nonempty slowness volume".to_owned(),
        ));
    }

    let reference_squared = reference_slowness_s_per_m * reference_slowness_s_per_m;
    let omega_squared = omega_rad_s * omega_rad_s;
    slowness_s_per_m
        .iter()
        .map(|&slowness| {
            if !slowness.is_finite() || slowness <= 0.0 {
                Err(KwaversError::InvalidInput(format!(
                    "CBS slowness must be positive and finite, got {slowness}"
                )))
            } else {
                Ok(omega_squared * (slowness * slowness - reference_squared))
            }
        })
        .collect()
}

/// Return `epsilon = max(||V||_infinity, f64::EPSILON)`.
///
/// # Errors
/// Returns an error if the potential is empty or nonfinite.
pub fn convergence_epsilon(real_potential: &[f64]) -> KwaversResult<f64> {
    if real_potential.is_empty() {
        return Err(KwaversError::InvalidInput(
            "CBS epsilon requires a nonempty scattering potential".to_owned(),
        ));
    }
    let mut max_abs = 0.0_f64;
    for &value in real_potential {
        if !value.is_finite() {
            return Err(KwaversError::InvalidInput(format!(
                "CBS scattering potential must be finite, got {value}"
            )));
        }
        max_abs = max_abs.max(value.abs());
    }
    Ok(max_abs.max(f64::EPSILON))
}

/// Compute the shifted CBS potential `V_s = V - i epsilon`.
///
/// # Errors
/// Returns an error if `epsilon` is invalid or the potential is invalid.
pub fn shifted_potential(real_potential: &[f64], epsilon: f64) -> KwaversResult<Vec<Complex64>> {
    if !epsilon.is_finite() || epsilon <= 0.0 {
        return Err(KwaversError::InvalidInput(format!(
            "CBS epsilon must be positive and finite, got {epsilon}"
        )));
    }
    convergence_epsilon(real_potential)?;
    Ok(real_potential
        .iter()
        .map(|&value| Complex64::new(value, -epsilon))
        .collect())
}

/// Compute the pointwise CBS preconditioner `gamma = i epsilon / V_s`.
///
/// # Errors
/// Returns an error if inputs violate the shifted-potential contract.
pub fn pointwise_preconditioner(
    shifted_potential: &[Complex64],
    epsilon: f64,
) -> KwaversResult<Vec<Complex64>> {
    if shifted_potential.is_empty() {
        return Err(KwaversError::InvalidInput(
            "CBS preconditioner requires a nonempty shifted potential".to_owned(),
        ));
    }
    if !epsilon.is_finite() || epsilon <= 0.0 {
        return Err(KwaversError::InvalidInput(format!(
            "CBS epsilon must be positive and finite, got {epsilon}"
        )));
    }
    let numerator = Complex64::new(0.0, epsilon);
    shifted_potential
        .iter()
        .map(|&value| {
            if !value.re.is_finite() || !value.im.is_finite() || value.norm_sqr() == 0.0 {
                Err(KwaversError::InvalidInput(format!(
                    "CBS shifted potential must be finite and nonzero, got {value}"
                )))
            } else {
                Ok(numerator / value)
            }
        })
        .collect()
}
