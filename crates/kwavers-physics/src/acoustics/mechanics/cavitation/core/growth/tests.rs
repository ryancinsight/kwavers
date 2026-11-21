use super::*;
use eunomia::assert_relative_eq;
use kwavers_core::constants::fundamental::{ATMOSPHERIC_PRESSURE, GAS_CONSTANT};
use kwavers_core::constants::numerical::MHZ_TO_HZ;
use kwavers_core::constants::thermodynamic::T_AMBIENT;

// Standard air-in-water parameters (Eller & Flynn 1965 regime)
const R0: f64 = 5e-6; // 5 µm bubble radius
const P0: f64 = ATMOSPHERIC_PRESSURE; // atmospheric pressure [Pa]
const P_AC: f64 = 120_000.0; // ~1.18 atm acoustic amplitude
const FREQ: f64 = MHZ_TO_HZ; // 1 MHz ultrasound
const D_AIR: f64 = 2.0e-9; // O₂ diffusivity in water [m²/s]
const C_SAT: f64 = 0.26; // dissolved O₂ at saturation [mol/m³]

/// Isothermal ideal-gas dn/dR for a bubble (Laplace term included);
/// converts a molar flux [mol/s] to dR/dt [m/s]. Same conversion as
/// the Stage 7 cross-check (smith_wang_diffusion.py).
fn dn_dr_mol_per_m(radius: f64) -> f64 {
    (FOUR_PI * radius * radius * P0
        + (16.0 / 3.0) * std::f64::consts::PI * radius * SURFACE_TENSION)
        / (GAS_CONSTANT * T_AMBIENT)
}

fn flux_to_um_per_min(flux: f64, radius: f64) -> f64 {
    flux / dn_dr_mol_per_m(radius) * 1e6 * 60.0
}

#[test]
fn test_rectified_diffusion_positive_flux() {
    // Above the rectified-diffusion threshold the net flux is
    // positive: 30 µm at 35 kPa, 26.6 kHz is above threshold
    // (P_T ≈ 27 kPa there) and grows at ≈ +0.75 µm/min.
    let flux = rectified_diffusion_rate(30e-6, P0, 35e3, 26.6e3, D_AIR, C_SAT);
    assert!(
        flux > 0.0,
        "Above threshold, acoustic pressure must yield positive molar flux"
    );
    // Physical reasonableness: flux should be very small (pico- to nano-moles/s)
    assert!(
        flux < 1.0,
        "Molar flux should be << 1 mol/s for microbubbles"
    );
    // The growth term is positive whenever P_ac > 0, even below
    // threshold: the 5 µm / 120 kPa / 1 MHz point is below
    // threshold (net ≈ −11.6 µm/min) but its rectified term is
    // ≈ +25 µm/min.
    let terms = rectified_diffusion_terms(R0, P0, P_AC, FREQ, D_AIR, C_SAT);
    assert!(
        terms.growth_flux > 0.0,
        "Rectified growth term must be positive for P_ac > 0"
    );
    assert!(
        terms.net_flux < 0.0,
        "5 µm / 120 kPa / 1 MHz is below threshold (net dissolution), got {}",
        terms.net_flux
    );
}

#[test]
fn test_rectified_diffusion_amplitude_scaling() {
    // The genuine rectified term is second order in the oscillation
    // amplitude (∝ ε² ∝ P_ac²), NOT linear in P_ac as the old formula
    // had it. At small amplitude (30 µm, 26.6 kHz, 5 vs 10 kPa,
    // ε ≈ 0.012/0.024) the growth flux must quadruple when P_ac
    // doubles. The NET flux does not scale this way (the Laplace
    // dissolution term is P_ac-independent), so the scaling is
    // asserted on the growth term.
    let r_ell = 30e-6;
    let f_ell = 26.6e3;
    let g5 = rectified_diffusion_terms(r_ell, P0, 5e3, f_ell, D_AIR, C_SAT).growth_flux;
    let g10 = rectified_diffusion_terms(r_ell, P0, 10e3, f_ell, D_AIR, C_SAT).growth_flux;
    assert_relative_eq!(g10 / g5, 4.0, epsilon = 0.02);

    // Flux is linear in concentration (both terms scale with C).
    let base = rectified_diffusion_rate(R0, P0, P_AC, FREQ, D_AIR, C_SAT);
    let doubled_c = rectified_diffusion_rate(R0, P0, P_AC, FREQ, D_AIR, 2.0 * C_SAT);
    assert_relative_eq!(doubled_c / base, 2.0, epsilon = 1e-10);
}

#[test]
fn test_rectified_diffusion_zero_acoustic_pressure() {
    // With no sound there is no rectified term — but the bubble still
    // dissolves by Laplace-driven diffusion, so the NET flux is
    // negative, not zero. (The old test asserted zero flux, which
    // encoded the missing dissolution term flagged by the Stage 6
    // audit.) The growth term itself must be exactly zero.
    let terms = rectified_diffusion_terms(R0, P0, 0.0, FREQ, D_AIR, C_SAT);
    assert_eq!(terms.growth_flux, 0.0, "No sound => no rectified growth");
    assert_eq!(terms.epsilon, 0.0, "No sound => zero oscillation amplitude");
    assert!(
        terms.net_flux < 0.0,
        "No sound => net Laplace dissolution, got {}",
        terms.net_flux
    );
    assert_relative_eq!(terms.net_flux, terms.dissolution_flux, epsilon = 1e-12);
}

#[test]
fn test_rectified_diffusion_zero_diffusivity() {
    // No diffusion => no flux of either sign.
    let flux = rectified_diffusion_rate(R0, P0, P_AC, FREQ, 0.0, C_SAT);
    assert_eq!(
        flux, 0.0,
        "Zero diffusivity must yield zero flux, got {flux}"
    );
    let flux_c0 = rectified_diffusion_rate(R0, P0, P_AC, FREQ, D_AIR, 0.0);
    assert_eq!(flux_c0, 0.0, "Zero concentration must yield zero flux");
}

#[test]
fn test_rectified_diffusion_checked_accepts_in_regime() {
    // P_AC = 120 kPa is P_ac/P0 = 1.18, inside the small-amplitude regime.
    let flux = rectified_diffusion_rate_checked(R0, P0, P_AC, FREQ, D_AIR, C_SAT)
        .expect("in-regime input must be accepted");
    assert_relative_eq!(
        flux,
        rectified_diffusion_rate(R0, P0, P_AC, FREQ, D_AIR, C_SAT),
        epsilon = 1e-12
    );
}

#[test]
fn test_rectified_diffusion_checked_rejects_burst_pressures() {
    // Stage 6 audit: the law was applied at P_ac/P0 = 66 (Iwasaki burst,
    // 6.7 MPa) and 146 (Kuji burst, 14.7 MPa), producing 986/1904 µm
    // growth artifacts. The checked form must refuse those inputs.
    for p_ac in [6.7e6, 14.7e6] {
        let err = rectified_diffusion_rate_checked(R0, P0, p_ac, FREQ, D_AIR, C_SAT)
            .expect_err("burst pressure must be rejected");
        assert!(
            matches!(
                err,
                KwaversError::Physics(PhysicsError::InvalidParameter { .. })
            ),
            "expected InvalidParameter, got {err:?}"
        );
    }
}

#[test]
fn test_rectified_diffusion_checked_rejects_nonpositive_ambient() {
    let err = rectified_diffusion_rate_checked(R0, 0.0, P_AC, FREQ, D_AIR, C_SAT)
        .expect_err("zero ambient pressure must be rejected");
    assert!(
        matches!(
            err,
            KwaversError::Physics(PhysicsError::InvalidParameter { .. })
        ),
        "expected InvalidParameter, got {err:?}"
    );
}

#[test]
fn test_thermal_phi_limits() {
    // Adiabatic limit X → 0: Φ → 3γ, κ → γ, δ_th → 0.
    let (re, im) = thermal_phi(1e-6, GAMMA_AIR);
    assert_relative_eq!(re / 3.0, GAMMA_AIR, epsilon = 0.01);
    assert!(
        im / re < 0.02,
        "thermal damping must vanish, got {}",
        im / re
    );
    // Isothermal limit X → ∞: Φ → 3, κ → 1, δ_th → 0.
    let (re, im) = thermal_phi(100.0, GAMMA_AIR);
    assert_relative_eq!(re / 3.0, 1.0, epsilon = 0.01);
    assert!(
        im / re < 0.02,
        "thermal damping must vanish, got {}",
        im / re
    );
    // Devin (1959) simplified forms: δ_th ≈ 3(γ−1)χ/2 for χ ≤ 0.08
    // (eq. (2.50)) and δ_th ≈ (γ−1)/(15γχ) for χ ≥ 0.5 (eq. (2.52)),
    // with χ = √X. Agreement is ~30% at small χ, ~1% at χ ≥ 1.
    let (re, im) = thermal_phi(0.05_f64.powi(2), GAMMA_AIR);
    let d_th = im / re;
    assert!(
        (0.02..0.05).contains(&d_th),
        "δ_th at χ=0.05 must be ~0.03 (Devin (2.50)), got {d_th}"
    );
    let (re, im) = thermal_phi(1.0_f64.powi(2), GAMMA_AIR);
    let d_th = im / re;
    assert!(
        (0.01..0.03).contains(&d_th),
        "δ_th at χ=1 must be ~0.019 (Devin (2.52)), got {d_th}"
    );
}

#[test]
fn test_eller_1969_benchmark() {
    // Eller (1969) conditions (Smith & Wang 2022, §4.2): 26.6 kHz,
    // P_ac = 20 kPa, air-saturated water, D = 2e-9 m²/s. Benchmark
    // constants C = 0.26 mol/m³ as in the Stage 7 cross-check.
    // This test checks the steady theory. Figure 3 of Smith & Wang
    // shows a radius-dependent empirical growth rate.
    // The old formula gave 515–527 µm/min here.
    // Expected values (κ-faithful Crum–Hansen, cross-checked against
    // an independent Python implementation to 6+ digits):
    //   R0:  20 µm   30 µm   40 µm   50 µm
    //   net: −1.65   −0.50   −0.13   +0.05 µm/min
    //   gro: +0.94   +0.66   +0.53   +0.47 µm/min
    //   ε:   0.064   0.065   0.066   0.069
    let f_ell = 26.6e3;
    let p_ell = 20e3;
    for (r_um, net_lo, net_hi) in [
        (20.0, -2.5, -0.5),
        (30.0, -1.0, 0.0),
        (40.0, -0.5, 0.2),
        (50.0, -0.2, 0.3),
    ] {
        let r = r_um * 1e-6;
        let terms = rectified_diffusion_terms(r, P0, p_ell, f_ell, D_AIR, C_SAT);
        let net_um_min = flux_to_um_per_min(terms.net_flux, r);
        let gro_um_min = flux_to_um_per_min(terms.growth_flux, r);
        // Saturated steady theory crosses zero near R0 = 47 µm.
        assert!(
            net_um_min > net_lo && net_um_min < net_hi,
            "R={r_um} µm: net rate must be small (|net| ≲ 2 µm/min), got {net_um_min:.2} µm/min"
        );
        // The separate rectified contribution is 0.47–0.94 µm/min.
        assert!(
            gro_um_min > 0.2 && gro_um_min < 1.2,
            "R={r_um} µm: steady rectified term outside reference range, got {gro_um_min:.2}"
        );
        // Oscillation amplitude is small (ε ≈ 0.065), as the law assumes.
        assert!(
            terms.epsilon > 0.05 && terms.epsilon < 0.08,
            "R={r_um} µm: ε must be ≈0.065, got {}",
            terms.epsilon
        );
    }
    // The old 40× overprediction is gone: the net flux magnitude at
    // 30 µm is >100× smaller than the old formula's 4.2e-12 mol/s.
    let old_flux =
        FOUR_PI * 30e-6 * D_AIR * C_SAT * (p_ell / P0) * (30e-6 * 30e-6 * f_ell / D_AIR).sqrt();
    let new_flux = rectified_diffusion_rate(30e-6, P0, p_ell, f_ell, D_AIR, C_SAT).abs();
    assert!(
        new_flux < old_flux / 100.0,
        "new |flux| {new_flux:.2e} must be >100× smaller than old {old_flux:.2e}"
    );
}

#[test]
fn test_eller_threshold_pressure() {
    // Threshold pressure at R0 = 30 µm is ≈ 27 kPa in this theory
    // (net rate crosses zero between 25 and 30 kPa at 26.6 kHz).
    let r = 30e-6;
    let f_ell = 26.6e3;
    let below = rectified_diffusion_rate(r, P0, 25e3, f_ell, D_AIR, C_SAT);
    let above = rectified_diffusion_rate(r, P0, 30e3, f_ell, D_AIR, C_SAT);
    assert!(
        below < 0.0,
        "25 kPa must still dissolve at 30 µm, got {below:.2e}"
    );
    assert!(above > 0.0, "30 kPa must grow at 30 µm, got {above:.2e}");
}
