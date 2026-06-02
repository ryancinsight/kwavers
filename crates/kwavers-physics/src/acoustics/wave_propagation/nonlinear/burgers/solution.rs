//! Fubini-Blackstock analytical Burgers solution.

use super::bessel::bessel_j;
use kwavers_core::constants::numerical::TWO_PI;
use crate::acoustics::wave_propagation::nonlinear::NonlinearParameters;
use std::f64::consts::PI;

/// Normalized amplitude of the nth harmonic from the Fubini-Blackstock solution.
///
/// Returns `|P_n|/P0` for a lossless plane wave at dimensionless propagation
/// distance `sigma = z / z_shock`.
///
/// ```text
/// B_n(sigma) = (2/(n sigma)) J_n(n sigma)    0 < sigma < 1
/// B_n(sigma) = 2/(n pi sigma)                sigma >= 1
/// ```
/// # Panics
/// - Panics if assertion fails: `harmonic order must be >= 1`.
///
#[must_use]
pub fn fubini_harmonic_amplitude(n: u32, sigma: f64) -> f64 {
    assert!(n >= 1, "harmonic order must be >= 1");
    if sigma <= 0.0 {
        return if n == 1 { 1.0 } else { 0.0 };
    }
    if sigma < 1.0 {
        let arg = n as f64 * sigma;
        2.0 * bessel_j(n, arg) / arg
    } else {
        2.0 / (n as f64 * PI * sigma)
    }
}

/// Fundamental pressure amplitude from the Fubini-Blackstock solution with
/// independent thermoviscous attenuation applied to the fundamental.
///
/// ```text
/// z_shock = rho0 c0^3 / (beta omega P0)
/// sigma = z / z_shock
/// P1(z) = P0 B1(sigma) exp(-alpha(f0) z)
/// ```
#[must_use]
pub fn burgers_equation(
    initial_pressure: f64,
    frequency: f64,
    distance: f64,
    params: &NonlinearParameters,
) -> f64 {
    let omega = TWO_PI * frequency;
    let z_shock =
        params.density * params.sound_speed.powi(3) / (params.beta * omega * initial_pressure);
    let sigma = distance / z_shock;
    let b1 = fubini_harmonic_amplitude(1, sigma);
    let alpha = params.attenuation_at_frequency(frequency);

    initial_pressure * b1 * (-alpha * distance).exp()
}
