//! Harmonic Generation and Tissue Harmonic Imaging (THI)
//!
//! Models the creation of higher harmonics (2f, 3f...) due to nonlinear
//! propagation. THI relies on the second harmonic to improve image resolution
//! and reduce artifact clutter.

use super::{NonlinearParameters, TissueHarmonicProperties};
use std::f64::consts::PI;

/// Calculates the amplitude of the second harmonic generated during propagation
///
/// # Theory
///
/// Using perturbation theory on the lossless Burgers' equation, the second harmonic
/// pressure amplitude P2 grows linearly with distance initially:
///
/// ```text
/// P2(z) = (β * ω / (2 * ρ₀ * c₀³)) * P1² * z
/// ```
///
/// Where P1 is the fundamental pressure amplitude.
///
/// # Note on validity
///
/// Automatically limits the output to physical bounds (P2 max ~ P1/2 for saw-tooth).
#[must_use]
pub fn second_harmonic_amplitude(
    fundamental_pressure: f64,
    frequency: f64,
    distance: f64,
    params: &NonlinearParameters,
) -> f64 {
    let omega = 2.0 * PI * frequency;

    // Linear growth coefficient for second harmonic
    let coefficient = params.beta * omega / (2.0 * params.density * params.sound_speed.powi(3));

    // Calculate unrestricted amplitude
    let p2_unrestricted = coefficient * fundamental_pressure.powi(2) * distance;

    // Apply physical limits (energy conservation)
    // The second harmonic cannot exceed a certain fraction of the fundamental energy.
    // In a fully developed sawtooth wave, the nth harmonic amplitude goes as 1/n.
    // Therefore, max P2 is roughly P1/2 (relative to the initial fundamental).
    let max_p2 = fundamental_pressure * 0.5;

    // Smooth limitation
    let ratio = p2_unrestricted / max_p2;
    if ratio < 0.1 {
        // Linear regime
        p2_unrestricted
    } else {
        // Saturation regime (tanh acts as a smooth limiter)
        max_p2 * ratio.tanh()
    }
}

/// Estimates the efficiency of tissue harmonic generation
///
/// THI involves a trade-off: higher frequencies generate harmonics faster
/// due to nonlinearity, but those harmonics are attenuated much faster.
///
/// Returns a dimensionless metric [0, 1] representing relative efficiency.
#[must_use]
pub fn tissue_harmonic_efficiency(
    props: &TissueHarmonicProperties,
    params: &NonlinearParameters,
) -> f64 {
    let f1 = props.fundamental_frequency;
    let f2 = 2.0 * f1; // Second harmonic

    // Attenuation at both frequencies [Np/m]
    let alpha1 = params.attenuation_at_frequency(f1);
    let alpha2 = params.attenuation_at_frequency(f2);

    // Harmonic generation is proportional to beta, f1, and P1
    let generation_factor = params.beta * f1 * props.fundamental_pressure;

    // The harmonic must survive attenuation from focus back to transducer.
    // Simplified model: generation occurs strongly only near the focus (z ≈ F).
    let survival_factor = (-alpha2 * props.focal_depth).exp();

    // Fundamental attenuation to the focus limits the source pressure for harmonics
    let pumping_factor = (-alpha1 * props.focal_depth).exp();

    // Combine factors and normalize to a typical maximum value to get [0, 1] range
    // Normalization constant is empirical/arbitrary for this qualitative metric
    const NORMALIZATION: f64 = 1e-12;

    let efficiency = generation_factor * pumping_factor * survival_factor * NORMALIZATION;

    // Bound to [0.0, 1.0]
    efficiency.clamp(0.0, 1.0)
}

/// Optimizes fundamental frequency for maximum harmonic return
///
/// Find the frequency that maximizes the trade-off between nonlinear
/// generation (favors high f) and attenuation (favors low f).
#[must_use]
pub fn optimal_harmonic_frequency(
    depth: f64,
    params: &NonlinearParameters,
) -> f64 {
    // Analytically, taking the derivative d/df of the efficiency equation
    // and setting to zero yields (assuming f^1 attenuation):
    // f_opt = 1 / ( (2^y + 1) * alpha_0 * depth * y )
    // where y is the attenuation exponent

    let y = params.attenuation_exponent;
    let alpha_0 = params.attenuation_coeff * 1e-6; // per Hz factor (approximate)

    let denominator = (2.0f64.powf(y) + 1.0) * alpha_0 * depth * y;

    if denominator <= 0.0 {
        return 2.0e6; // Default to 2 MHz if parameters are invalid
    }

    let f_opt = 1.0 / denominator;

    // Clamp to realistic medical ultrasound range (1 - 15 MHz)
    f_opt.clamp(1.0e6, 15.0e6)
}

/// Simple model for contrast agent harmonic response
///
/// Microbubbles generate much stronger harmonics than tissue due to
/// volumetric resonance.
#[must_use]
pub fn contrast_harmonic_response(
    pressure: f64,
    frequency: f64,
    bubble_resonance: f64,
) -> f64 {
    // Driving frequency ratio
    let omega_ratio = frequency / bubble_resonance;

    // Simple nonlinear resonance model (enhanced near resonance)
    // Response scales with square of pressure for second harmonic
    let resonance_enhancement = 1.0 / ((1.0 - omega_ratio.powi(2)).powi(2) + 0.1 * omega_ratio.powi(2)).sqrt();

    // Scale factor for bubbles is orders of magnitude higher than tissue
    const BUBBLE_NONLINEARITY_SCALE: f64 = 1e-6;

    BUBBLE_NONLINEARITY_SCALE * pressure.powi(2) * resonance_enhancement
}
