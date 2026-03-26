//! Shock Wave Formation and Physics
//!
//! Models the nonlinear steepening of acoustic waves into shocks.

use super::NonlinearParameters;
use std::f64::consts::PI;

/// Calculates the spatial location where a shock wave forms
///
/// Due to nonlinear steepening, the positive phase of an acoustic wave travels
/// slightly faster than the local sound speed (c = c₀ + βu), and the negative phase
/// travels slightly slower. This causes the wave to steepen until a shock forms.
///
/// Returns distance [m]
#[must_use]
pub fn shock_formation_distance(
    initial_pressure: f64,
    frequency: f64,
    params: &NonlinearParameters,
) -> f64 {
    let omega = 2.0 * PI * frequency;
    let beta = params.beta;
    let rho_0 = params.density;
    let c_0 = params.sound_speed;

    // Shock formation distance l_s = (ρ₀ * c₀³) / (β * ω * P₀)
    let shock_dist = (rho_0 * c_0.powi(3)) / (beta * omega * initial_pressure);

    // Limit to reasonable physical values (prevent Infinity for very low pressures)
    shock_dist.clamp(0.0, 1e6)
}

/// Calculate the shock front thickness
///
/// Thermoviscous dissipation counteracts nonlinear steepening, resulting in a
/// finite shock thickness.
///
/// Returns thickness [m]
#[must_use]
pub fn shock_thickness(
    shock_pressure: f64,
    params: &NonlinearParameters,
) -> f64 {
    // Basic theoretical shock thickness (weak shock theory)
    // l_shock = (ρ₀ c₀³ δ) / (β P_shock)
    // Here we use attenuation coefficient at 1 MHz as a proxy for diffusivity (δ)
    let diffusivity = params.attenuation_coeff * 2.0 * params.sound_speed.powi(3);

    let thickness = diffusivity / (params.beta * shock_pressure);

    // Bounded below by mean free path for liquids (~1e-10 m)
    thickness.max(1e-10)
}

/// Estimates peak pressure considering shock dissipation
#[must_use]
pub fn shock_wave_amplitude(
    initial_pressure: f64,
    frequency: f64,
    distance: f64,
    params: &NonlinearParameters,
) -> f64 {
    let shock_dist = shock_formation_distance(initial_pressure, frequency, params);

    // Acoustic saturation limit
    let p_sat = super::saturation::acoustic_saturation_pressure(frequency, distance, params);

    if distance < shock_dist {
        // Pre-shock regime: nonlinear growth but dominated by fundamental
        // P(z) ~ P_0 (ignoring linear attenuation for this estimate)
        initial_pressure.min(p_sat)
    } else {
        // Post-shock regime: rapid dissipation, amplitude limits to saturation
        // Classic sawtooth wave dissipation: P(z) = P_0 / (1 + z/l_s)
        let sawtooth_amp = initial_pressure / (1.0 + distance / shock_dist);
        sawtooth_amp.min(p_sat)
    }
}
