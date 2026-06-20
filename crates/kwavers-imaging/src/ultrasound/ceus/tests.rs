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

/// PHY-13: at resonance the Hoff (2000) Lorentzian denominator collapses to
/// `δ_tot²` (the `(1−Ω²)²` term vanishes at Ω=1), so the peak cross-section has
/// the closed form `σ_s(ω₀) = 4π R² (ω₀R/c_L)² / δ_tot²` with `δ_tot` the sum of
/// the radiation/liquid-viscous/shell-viscous dimensionless damping (Church 1995).
/// This re-derives δ_tot independently and checks the assembled Lorentzian peak.
#[test]
fn test_scattering_resonance_matches_closed_form() {
    use kwavers_core::constants::cavitation::VISCOSITY_WATER;
    use kwavers_core::constants::fundamental::SOUND_SPEED_WATER;
    use std::f64::consts::PI;

    let mb = Microbubble::sono_vue();
    let r = mb.radius_eq;
    let c_l = SOUND_SPEED_WATER;
    let rho_l = DENSITY_WATER_NOMINAL;
    let mu_l = VISCOSITY_WATER;
    let omega0 = 2.0 * PI * mb.resonance_frequency(ATMOSPHERIC_PRESSURE, rho_l);

    // Independent δ_tot (Church 1995 A3–A5), matching the documented model.
    let delta_rad = omega0 * r / c_l;
    let delta_vis = 4.0 * mu_l / (omega0 * rho_l * r * r);
    let delta_sh = 4.0 * mb.shell_thickness * mb.shell_viscosity / (omega0 * rho_l * r * r * r);
    let delta_tot = delta_rad + delta_vis + delta_sh;

    let ka0 = omega0 * r / c_l;
    let expected = FOUR_PI * r * r * ka0 * ka0 / (delta_tot * delta_tot);
    let f_r = mb.resonance_frequency(ATMOSPHERIC_PRESSURE, rho_l);
    let actual = mb.scattering_cross_section(f_r);
    assert!(
        (actual - expected).abs() / expected < 1e-9,
        "resonance σ_s {actual:.6e} vs closed form {expected:.6e}"
    );
}

/// PHY-13: well below resonance the Lorentzian denominator → 1, so the model
/// reduces to the geometric coupling `σ_s ≈ 4π R² (ωR/c_L)² ∝ ω²` — doubling the
/// frequency quadruples the cross-section. Needs no internal parameters.
#[test]
fn test_scattering_low_frequency_omega_squared_scaling() {
    let mb = Microbubble::sono_vue();
    let f_r = mb.resonance_frequency(ATMOSPHERIC_PRESSURE, DENSITY_WATER_NOMINAL);
    // Far below resonance (Ω ≪ 1/δ_tot) ⇒ both the (1−Ω²)² and (δ_tot·Ω)²
    // denominator terms → their Ω→0 limits ⇒ denom ≈ 1 ⇒ σ_s ≈ 4πR²(ka)² ∝ ω².
    let f = f_r / 1000.0;
    let s1 = mb.scattering_cross_section(f);
    let s2 = mb.scattering_cross_section(2.0 * f);
    assert!(
        (s2 / s1 - 4.0).abs() < 0.01,
        "low-frequency σ_s must scale as ω² (ratio≈4): got {}",
        s2 / s1
    );
}
