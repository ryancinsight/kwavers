//! Burgers' Equation for 1D Nonlinear Acoustics
//!
//! Models propagation of finite-amplitude waves in thermoviscous fluids.
//!
//! # Mathematical Foundation
//!
//! The generalized Burgers' equation:
//! ```text
//! ∂P/∂z - (β/(ρ₀c₀³)) P ∂P/∂τ - (δ/(2c₀³)) ∂²P/∂τ² = 0
//! ```

use super::NonlinearParameters;
use std::f64::consts::PI;

/// Calculates acoustic pressure evolution using generalized Burgers' equation
///
/// Simplified analytical solution for early-stage nonlinear propagation
/// before shock formation.
///
/// # Arguments
/// * `initial_pressure` - Source pressure amplitude [Pa]
/// * `frequency` - Fundamental frequency [Hz]
/// * `distance` - Propagation distance [m]
/// * `params` - Medium nonlinear parameters
///
/// # Returns
/// Expected pressure amplitude at given distance, or 0.0 if shock
/// formation strongly violates the weak-shock assumptions.
#[must_use]
pub fn burgers_equation(
    initial_pressure: f64,
    frequency: f64,
    distance: f64,
    params: &NonlinearParameters,
) -> f64 {
    let omega = 2.0 * PI * frequency;
    // Wave number: let k = omega / params.sound_speed;

    // Shock formation distance (l_s)
    let shock_distance = params.density * params.sound_speed.powi(3)
        / (params.beta * omega * initial_pressure);

    // If beyond shock distance, simple analytical solutions break down
    if distance >= shock_distance {
        // Return 0 indicating invalid regime for this simplified model
        // (Real implementation needs numerical solver or weak-shock theory)
        return 0.0;
    }

    // Attenuation factor
    let alpha = params.attenuation_at_frequency(frequency);
    let attenuation_factor = (-alpha * distance).exp();

    // Fubini solution (first term approximation)
    // Valid for sigma = distance/shock_distance < 1
    let sigma = distance / shock_distance;

    // J_1 is Bessel function of first kind, order 1
    // For small arguments, J_1(x) ~ x/2
    let bessel_approx = sigma / 2.0;

    // Amplitude of fundamental frequency
    let fundamental = initial_pressure * attenuation_factor * (2.0 / sigma) * bessel_approx;

    // Account for energy transfer to harmonics (simplified)
    let energy_loss_factor = 1.0 - (sigma * sigma / 8.0).min(0.5);

    fundamental * energy_loss_factor
}
