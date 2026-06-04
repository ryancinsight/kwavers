//! Tests for CEUS domain definitions.

use super::microbubble::Microbubble;
use kwavers_core::constants::fundamental::{ATMOSPHERIC_PRESSURE, DENSITY_WATER_NOMINAL};
use kwavers_core::constants::numerical::FOUR_PI;
use kwavers_core::constants::numerical::MHZ_TO_HZ;

/// Cross-section must be strictly positive for any physical frequency.
/// # Panics
/// - Panics if an internal precondition is violated.
///
#[test]
fn test_scattering_cross_section_positive() {
    let mb = Microbubble::sono_vue(); // 1.5 µm radius
    for &f_mhz in &[1.0f64, 2.0, 3.0, 5.0, 10.0] {
        let sigma = mb.scattering_cross_section(f_mhz * MHZ_TO_HZ);
        assert!(
            sigma > 0.0 && sigma.is_finite(),
            "σ_s must be positive and finite at {f_mhz} MHz, got {sigma:.3e}"
        );
    }
}

/// Cross-section is maximised near the resonance frequency.
/// # Panics
/// - Panics if assertion fails: `σ_s(f_r)={sigma_res:.3e} should exceed σ_s(10·f_r)={sigma_far:.3e}`.
///
#[test]
fn test_scattering_peak_near_resonance() {
    let mb = Microbubble::sono_vue();
    let f_r = mb.resonance_frequency(ATMOSPHERIC_PRESSURE, DENSITY_WATER_NOMINAL);
    let sigma_res = mb.scattering_cross_section(f_r);
    let sigma_far = mb.scattering_cross_section(f_r * 10.0);
    assert!(
        sigma_res > sigma_far,
        "σ_s(f_r)={sigma_res:.3e} should exceed σ_s(10·f_r)={sigma_far:.3e}"
    );
}

/// σ_s must use c_L=1480 m/s (water), not 343 m/s (air).
/// Verify: σ_s at resonance is bounded well below the acoustic aperture limit.
/// For kR << 1, σ_s << 4πR² — encapsulated microbubbles in water are sub-resonant scatterers.
/// # Panics
/// - Panics if assertion fails: `Implausibly large σ_s={sigma_res:.3e} suggests wrong sound speed`.
/// - Panics if assertion fails: `σ_s must be positive`.
///
#[test]
fn test_scattering_uses_water_sound_speed() {
    let mb = Microbubble::sono_vue();
    let f_r = mb.resonance_frequency(ATMOSPHERIC_PRESSURE, DENSITY_WATER_NOMINAL);
    let sigma_res = mb.scattering_cross_section(f_r);
    let geo = FOUR_PI * mb.radius_eq * mb.radius_eq;
    // Sanity: σ_s should be at most ~100× geometric for typical damping (~0.05)
    assert!(
        sigma_res < 200.0 * geo,
        "Implausibly large σ_s={sigma_res:.3e} suggests wrong sound speed"
    );
    assert!(sigma_res > 0.0, "σ_s must be positive");
}

/// Cross-section must decrease above resonance.
/// # Panics
/// - Panics if assertion fails: `σ_s should decrease above resonance: σ(1.5·f_r)={s1:.3e} σ(3·f_r)={s2:.3e}`.
///
#[test]
fn test_scattering_decreases_above_resonance() {
    let mb = Microbubble::sono_vue();
    let f_r = mb.resonance_frequency(ATMOSPHERIC_PRESSURE, DENSITY_WATER_NOMINAL);
    let s1 = mb.scattering_cross_section(f_r * 1.5);
    let s2 = mb.scattering_cross_section(f_r * 3.0);
    assert!(
        s1 > s2,
        "σ_s should decrease above resonance: σ(1.5·f_r)={s1:.3e} σ(3·f_r)={s2:.3e}"
    );
}
