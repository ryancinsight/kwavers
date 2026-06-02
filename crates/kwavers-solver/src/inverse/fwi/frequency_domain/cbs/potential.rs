//! CBS scattering-potential identities.
//!
//! # Contracts
//!
//! For reference slowness `s0`, model slowness `s`, and angular frequency
//! `omega`, the real Helmholtz scattering potential is
//! `V = omega^2 (s^2 - s0^2)`.
//!
//! For the PSTD leapfrog Green operator, the temporal mass term is the exact
//! discrete symbol `Omega_dt^2 = 4 sin^2(omega dt / 2) / dt^2`, so the matching
//! scattering potential is `V = Omega_dt^2 (s^2 - s0^2)`. Using the continuous
//! `omega^2` contrast with the PSTD denominator mixes two different discrete
//! equations and breaks the adjoint theorem.
//!
//! The shifted CBS potential is `V_s = V - i epsilon`, with
//! `epsilon >= ||V||_infinity`. This gives every voxel a nonzero imaginary
//! component, so pointwise CBS preconditioners remain finite.

use super::green::GreenOperatorKind;
use kwavers_core::error::{KwaversError, KwaversResult};
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
    real_scattering_potential_from_factor(
        helmholtz_frequency_factor(omega_rad_s)?,
        slowness_s_per_m,
        reference_slowness_s_per_m,
    )
}

/// Compute the real PSTD leapfrog scattering potential.
///
/// The finite-difference time discretization replaces the continuous
/// Helmholtz mass term `omega^2 s^2` with
/// `4 sin^2(omega dt / 2) s^2 / dt^2`. This function returns the corresponding
/// contrast against the reference slowness.
///
/// # Errors
/// Returns an error if frequency, timestep, or slowness values are outside the
/// physical finite positive domain.
pub fn real_pstd_scattering_potential(
    omega_rad_s: f64,
    time_step_s: f64,
    slowness_s_per_m: &Array3<f64>,
    reference_slowness_s_per_m: f64,
) -> KwaversResult<Vec<f64>> {
    real_scattering_potential_from_factor(
        pstd_temporal_angular_frequency_squared(omega_rad_s, time_step_s)?,
        slowness_s_per_m,
        reference_slowness_s_per_m,
    )
}

/// Compute the scattering potential matching a selected CBS Green operator.
///
/// # Errors
/// Returns an error when the operator or slowness contract is invalid.
pub fn real_scattering_potential_for_operator(
    omega_rad_s: f64,
    slowness_s_per_m: &Array3<f64>,
    reference_slowness_s_per_m: f64,
    operator: GreenOperatorKind,
) -> KwaversResult<Vec<f64>> {
    match operator {
        GreenOperatorKind::SpectralPstdPeriodic { time_step_s, .. } => {
            real_pstd_scattering_potential(
                omega_rad_s,
                time_step_s,
                slowness_s_per_m,
                reference_slowness_s_per_m,
            )
        }
        GreenOperatorKind::DenseFreeSpace | GreenOperatorKind::SpectralPeriodic { .. } => {
            real_scattering_potential(omega_rad_s, slowness_s_per_m, reference_slowness_s_per_m)
        }
    }
}

/// Compute the discrete PSTD temporal angular-frequency square.
///
/// `Omega_dt^2 = 4 sin^2(omega dt / 2) / dt^2` is the exact mass coefficient of
/// the second-order leapfrog time discretization at angular frequency `omega`.
///
/// # Errors
/// Returns an error if frequency or timestep is outside the finite positive
/// domain.
pub fn pstd_temporal_angular_frequency_squared(
    omega_rad_s: f64,
    time_step_s: f64,
) -> KwaversResult<f64> {
    validate_positive_finite(omega_rad_s, "CBS omega")?;
    validate_positive_finite(time_step_s, "PSTD time step")?;
    Ok(4.0 * (0.5 * omega_rad_s * time_step_s).sin().powi(2) / time_step_s.powi(2))
}

/// Return the multiplicative coefficient in `dV/ds = factor * s`.
///
/// # Errors
/// Returns an error when the selected operator has an invalid frequency-domain
/// contract.
pub fn scattering_slowness_derivative_factor_for_operator(
    omega_rad_s: f64,
    operator: GreenOperatorKind,
) -> KwaversResult<f64> {
    let frequency_factor = match operator {
        GreenOperatorKind::SpectralPstdPeriodic { time_step_s, .. } => {
            pstd_temporal_angular_frequency_squared(omega_rad_s, time_step_s)?
        }
        GreenOperatorKind::DenseFreeSpace | GreenOperatorKind::SpectralPeriodic { .. } => {
            helmholtz_frequency_factor(omega_rad_s)?
        }
    };
    Ok(2.0 * frequency_factor)
}

fn helmholtz_frequency_factor(omega_rad_s: f64) -> KwaversResult<f64> {
    validate_positive_finite(omega_rad_s, "CBS omega")?;
    Ok(omega_rad_s * omega_rad_s)
}

fn real_scattering_potential_from_factor(
    frequency_factor: f64,
    slowness_s_per_m: &Array3<f64>,
    reference_slowness_s_per_m: f64,
) -> KwaversResult<Vec<f64>> {
    validate_positive_finite(frequency_factor, "CBS frequency factor")?;
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
    slowness_s_per_m
        .iter()
        .map(|&slowness| {
            if !slowness.is_finite() || slowness <= 0.0 {
                Err(KwaversError::InvalidInput(format!(
                    "CBS slowness must be positive and finite, got {slowness}"
                )))
            } else {
                Ok(frequency_factor * (slowness * slowness - reference_squared))
            }
        })
        .collect()
}

fn validate_positive_finite(value: f64, label: &str) -> KwaversResult<()> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(KwaversError::InvalidInput(format!(
            "{label} must be positive and finite, got {value}"
        )))
    }
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
