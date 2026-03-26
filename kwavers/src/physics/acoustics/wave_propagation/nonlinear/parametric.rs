//! Parametric Acoustics (Parametric Arrays)
//!
//! Models the nonlinear mixing of two primary frequencies to
//! generate sum and difference frequencies (parametric array).
//!
//! Extremely useful for generating highly directional low-frequency sound.

use super::NonlinearParameters;
use std::f64::consts::PI;

/// Calculates the amplitude of the difference frequency (parametric array)
///
/// Mixing f1 and f2 creates f_diff = |f1 - f2|.
/// The virtual array created by nonlinear interaction allows a highly
/// directional low-frequency beam from a small transducer.
///
/// # Mathematical Model
/// Westervelt's solution for the difference frequency pressure on axis:
/// ```text
/// P_diff(R) = (W * β * f_diff²) / (2 * ρ₀ * c₀⁴ * R * α_T)
/// ```
/// Where:
/// - W: Primary acoustical power
/// - f_diff: Difference frequency
/// - α_T: Total attenuation of primaries
#[must_use]
pub fn difference_frequency_amplitude(
    p1: f64,
    p2: f64,
    f1: f64,
    f2: f64,
    distance: f64,
    params: &NonlinearParameters,
) -> f64 {
    let f_diff = (f1 - f2).abs();
    let omega_diff = 2.0 * PI * f_diff;

    // Need an estimate for primary attenuation (Np/m)
    // Assume f1 and f2 are close, take average attenuation
    let f_avg = (f1 + f2) / 2.0;
    let alpha_t = params.attenuation_at_frequency(f_avg);

    let rho_0 = params.density;
    let c_0 = params.sound_speed;
    let beta = params.beta;

    // Amplitude in the far field (Westervelt model variant)
    let p_diff = (p1 * p2 * beta * omega_diff.powi(2))
        / (2.0 * rho_0 * c_0.powi(4) * distance * alpha_t);

    // Self-demodulation limits actual performance
    let self_demod = self_demodulation(p1.max(p2), f_avg, distance, params);

    // Real parametric arrays are often bounded by self-demodulation
    p_diff.min(self_demod * 2.0)
}

/// Calculates the amplitude of the sum frequency
///
/// Sum frequency f_sum = f1 + f2 is highly attenuated since
/// absorption scales with frequency.
#[must_use]
pub fn sum_frequency_amplitude(
    p1: f64,
    p2: f64,
    f1: f64,
    f2: f64,
    distance: f64,
    params: &NonlinearParameters,
) -> f64 {
    let f_sum = f1 + f2;
    let omega_sum = 2.0 * PI * f_sum;

    let rho_0 = params.density;
    let c_0 = params.sound_speed;

    // Fundamental growth factor (similar to second harmonic)
    let coefficient = params.beta * omega_sum / (2.0 * rho_0 * c_0.powi(3));

    // Linear growth estimate (ignoring severe attenuation for a moment)
    let p_sum_raw = coefficient * p1 * p2 * distance;

    // Attenuation of the sum frequency determines survival
    let alpha_sum = params.attenuation_at_frequency(f_sum);
    let survival = (-alpha_sum * distance).exp();

    p_sum_raw * survival
}

/// Estimates self-demodulation (generation of LF envelope)
///
/// A pulsed wave effectively acts as a primary pair, producing low
/// frequencies matching its envelope due to parametric interaction.
#[must_use]
pub fn self_demodulation(
    pulse_pressure: f64,
    _center_freq: f64,
    distance: f64,
    params: &NonlinearParameters,
) -> f64 {
    // Westervelt / Berktay far-field formula (simplified bounding)
    

    (params.beta * pulse_pressure.powi(2))
        / (16.0 * params.density * params.sound_speed.powi(2) * distance)
}
