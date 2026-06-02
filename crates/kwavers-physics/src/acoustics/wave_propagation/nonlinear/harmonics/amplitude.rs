use super::super::NonlinearParameters;
use crate::acoustics::wave_propagation::nonlinear::burgers::fubini_harmonic_amplitude;
use crate::acoustics::wave_propagation::nonlinear::shock::shock_formation_distance;

/// Second harmonic pressure amplitude at axial distance z (Pa).
///
/// ## Algorithm
/// 1. Shock formation distance: z_shock = ρ₀c₀³/(β ω P₀)
/// 2. Gol'dberg number: σ = z / z_shock
/// 3. Fubini coefficient B₂(σ)
/// 4. Apply second-harmonic attenuation: exp(−α(2f₀) · z)
///
/// ## References
/// - Aanonsen et al. (1984) J. Acoust. Soc. Am. 75(3), eq. (6).
/// - Hamilton & Blackstock (1998) Nonlinear Acoustics §4.3, eq. (4.3.7).
#[must_use]
pub fn second_harmonic_amplitude(
    fundamental_pressure: f64,
    frequency: f64,
    distance: f64,
    params: &NonlinearParameters,
) -> f64 {
    nth_harmonic_amplitude(fundamental_pressure, frequency, distance, 2, params)
}

/// Nth harmonic pressure amplitude at axial distance z (Pa).
///
/// ```text
/// Pₙ(z) = P₀ × Bₙ(σ) × exp(−α(n·f₀) · z)
/// ```
///
/// ## References
/// - Aanonsen et al. (1984) J. Acoust. Soc. Am. 75(3), eq. (6).
/// - Hamilton & Blackstock (1998) §4.3, eq. (4.3.7).
/// # Panics
/// - Panics if assertion fails: `harmonic order must be ≥ 1`.
///
#[must_use]
pub fn nth_harmonic_amplitude(
    fundamental_pressure: f64,
    frequency: f64,
    distance: f64,
    harmonic: u32,
    params: &NonlinearParameters,
) -> f64 {
    assert!(harmonic >= 1, "harmonic order must be ≥ 1");

    let z_shock = shock_formation_distance(fundamental_pressure, frequency, params);
    let sigma = distance / z_shock;

    let bn = fubini_harmonic_amplitude(harmonic, sigma);
    let alpha_n = params.attenuation_at_frequency(harmonic as f64 * frequency);

    fundamental_pressure * bn * (-alpha_n * distance).exp()
}
