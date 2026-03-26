//! Acoustic Saturation Physics
//!
//! Saturation determines the hard upper limit on achievable acoustic pressure
//! regardless of how much power is pumped into the transducer. As amplitude
//! increases, nonlinear energy shifting to harmonics (which attenuate faster)
//! forms a hard limit.

use super::NonlinearParameters;
use std::f64::consts::PI;

/// Calculates the acoustic saturation pressure
///
/// As initial pressure increases to infinity, the pressure at a distance z
/// approaches an asymptotic limit (saturation) due to nonlinear dissipation.
///
/// Returns maximum un-focused pressure [Pa]
#[must_use]
pub fn acoustic_saturation_pressure(
    frequency: f64,
    distance: f64,
    params: &NonlinearParameters,
) -> f64 {
    let omega = 2.0 * PI * frequency;

    // Saturation pressure P_sat = (ρ₀ * c₀³) / (β * ω * z)
    let p_sat = (params.density * params.sound_speed.powi(3)) / (params.beta * omega * distance);

    // Ensure valid physical result
    p_sat.max(0.0)
}

/// Estimates the nonlinear threshold for cavitation
///
/// Very high intensity pulses (which have formed shocks) have enhanced
/// potential for cavitation due to the rapid pressure transitions.
#[must_use]
pub fn nonlinear_cavitation_threshold(
    frequency: f64,
    distance: f64,
    params: &NonlinearParameters,
) -> f64 {
    // Mechanical Index concept suggests PI scales with 1/sqrt(f)
    // M.I. = P_neg (MPa) / sqrt(f_c (MHz))

    // Baseline threshold in water ~ 1 MPa at 1 MHz
    let f_mhz = frequency / 1e6;
    let base_threshold = 1e6 * f_mhz.sqrt();

    // Saturation can cap achievable negative pressure
    let p_sat = acoustic_saturation_pressure(frequency, distance, params);

    // The shock process steepens the positive phase more than the negative phase
    // but the rapid transitions can lower the apparent threshold
    let shock_enhancement = 0.8; // Lower threshold by 20% if shocks form

    let effective_threshold = base_threshold * shock_enhancement;

    // It's possible for saturation to cap pressure below theoretical threshold
    if p_sat < effective_threshold {
        // Very hard to cavitate here
        effective_threshold * 2.0
    } else {
        effective_threshold
    }
}

/// Calculate the nonlinear radiation force (Langevin) on bubbles
#[must_use]
pub fn nonlinear_radiation_force(
    acoustic_intensity: f64,
    bubble_radius: f64,
    params: &NonlinearParameters,
) -> f64 {
    // Radiation force F_rad = (2αI)/c₀
    // For bubbles, cross-section replaces α in the continuum limit.

    let c_0 = params.sound_speed;

    // Classical primary radiation force on a scattering particle
    // F = π * a^2 * Y_p * (I / c₀)
    // Y_p is acoustic radiation force function

    let cross_section = PI * bubble_radius.powi(2);

    // Assume Y_p ≈ 1.0 (highly idealized geometric scattering limit)
    let y_p = 1.0;

    let f_rad_linear = cross_section * y_p * (acoustic_intensity / c_0);

    // Nonlinear correction (often minimal for F_rad, but included for completeness)
    // Higher harmonics scatter differently
    let fnl_correction = 1.0 + (params.beta - 1.0) * 0.1;

    f_rad_linear * fnl_correction
}
