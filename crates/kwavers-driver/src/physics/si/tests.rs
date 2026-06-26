//! Slice-wide tests for the [`crate::physics::si`] vertical slice (Phase 3f).
//!
//! Phase 3f consolidated the previously-inline `mod tests { ... }` blocks of the `src/si.rs`
//! flat-file into a single `crate::physics::si::tests` module, mirroring the Phase 2c
//! `place::tests` / `cost::tests` / `route::tests` consolidation pattern: one slice-wide
//! test file collects the existing test surface + the 3 new slice APIs (impedance_target,
//! channel_operating_margin_db, return_loss_db) so an external reviewer can grep `tests.rs`
//! once and see the entire si-slice behavioural contract.

use crate::physics::si::{
    channel_operating_margin_db, crosstalk_coupling, differential_microstrip_impedance,
    impedance_target, microstrip_delay_s_per_m, microstrip_impedance, return_loss_db,
    risetime_degradation_ps_per_m, stripline_impedance, within_skew,
};

// -------- Existing fixtures (lifted from src/si.rs::tests) --------

#[test]
fn fr4_50_ohm_geometry() {
    // FR4 (εr≈4.3): w/h ≈ 2 is the classic ~50 Ω microstrip.
    let z = microstrip_impedance(2.0, 1.0, 4.3);
    assert!(
        (47.0..=52.0).contains(&z),
        "w/h=2 on FR4 should be ~50 Ω, got {z:.1}"
    );
    // Wider trace ⇒ lower impedance.
    assert!(microstrip_impedance(4.0, 1.0, 4.3) < z);
}

#[test]
fn crosstalk_falls_with_spacing() {
    // Coupling drops sharply as traces separate; equal spacing/height ⇒ 0.5.
    assert!((crosstalk_coupling(0.1e-3, 0.1e-3) - 0.5).abs() < 1e-9);
    assert!(crosstalk_coupling(0.3e-3, 0.1e-3) < crosstalk_coupling(0.1e-3, 0.1e-3));
    assert!(crosstalk_coupling(1.0e-3, 0.1e-3) < 0.02);
}

#[test]
fn skew_budget_against_5ns() {
    // 5 ns delay resolution is generous: even a 100 mm length mismatch (~0.7 ns on FR4) is fine.
    let d = microstrip_delay_s_per_m(0.3, 1.0, 4.3);
    assert!(d > 5.0e-9 && d < 7.5e-9, "FR4 delay ~6 ns/m, got {:.2e}", d);
    assert!(
        within_skew(0.10, 0.0, d, 5.0e-9),
        "100 mm skew is within a 5 ns budget"
    );
    assert!(!within_skew(1.0, 0.0, d, 5.0e-9), "1 m skew exceeds 5 ns");
}

#[test]
fn stripline_50_ohm_geometry() {
    // FR4: er=4.3. A stripline with b=1.0 mm, w≈0.6 mm, t=35µm gives ~50 Ω.
    // At w=0.5, b=1.0, t=0 in normalised units:  Z = 60/√4.3·ln(4/(0.67π·0.4)) ≈ 50 Ω.
    let z = stripline_impedance(0.5, 0.0, 1.0, 4.3);
    assert!(
        (45.0..=65.0).contains(&z),
        "stripline should be near 50 Ω, got {z:.1}"
    );
    // Wider trace ⇒ lower impedance.
    assert!(stripline_impedance(1.0, 0.0, 1.0, 4.3) < z);
}

#[test]
fn differential_pair_impedance_is_below_single_ended() {
    // For s/h = 1 (moderate coupling): k = 0.5 ⇒ Z_diff = 2·Z₀·(1−0.5) = Z₀.
    // For s/h = 5 (loose): k ≈ 0.038 ⇒ Z_diff ≈ 2·Z₀·0.962 ≈ slightly below 2·Z₀.
    let z_se = microstrip_impedance(2.0, 1.0, 4.3);
    let z_diff_close = differential_microstrip_impedance(2.0, 1.0, 1.0, 4.3);
    let z_diff_loose = differential_microstrip_impedance(2.0, 1.0, 5.0, 4.3);
    assert!(
        z_diff_close < 2.0 * z_se,
        "tight coupling must reduce differential impedance vs 2·Z₀"
    );
    assert!(
        z_diff_loose < 2.0 * z_se,
        "loose coupling still has some reduction"
    );
    assert!(
        z_diff_close < z_diff_loose,
        "tighter spacing gives lower differential Z"
    );
}

#[test]
fn risetime_degradation_is_positive_and_small_at_2mhz() {
    // At 2 MHz on FR4 (er=4.3, tan_δ=0.02), 1 oz copper — skin effect is negligible
    // (δ≈46µm > 35µm foil), so degradation is dominated by dielectric loss, ~ps/m range.
    let deg = risetime_degradation_ps_per_m(0.3, 1.0, 4.3, 0.02, 2.0e6, 1.0);
    assert!(deg > 0.0, "degradation must be positive");
    // Higher frequency ⇒ more skin effect loss ⇒ more degradation.
    let deg_100mhz = risetime_degradation_ps_per_m(0.3, 1.0, 4.3, 0.02, 100.0e6, 1.0);
    assert!(
        deg_100mhz > deg,
        "higher frequency must degrade more: {deg_100mhz:.1} > {deg:.1} ps/m"
    );
}

// -------- Phase 3f NEW APIs: impedance_target, channel_operating_margin_db, return_loss_db --------

#[test]
fn impedance_target_matches_at_zero_reflection_and_scales_with_gamma() {
    // Γ_max = 0 ⇒ Z_target = Z_driver (perfect match, no tolerance for reflection).
    assert!(
        (impedance_target(50.0, 0.0) - 50.0).abs() < 1e-12,
        "Γ_max=0 ⇒ exact match, no up-shift"
    );
    // Γ_max = 0.1 ⇒ Z_target = 50·(1.1)/(0.9) ≈ 61.111 Ω (10% mismatch tolerance).
    assert!(
        (impedance_target(50.0, 0.10) - 50.0 * 1.1 / 0.9).abs() < 1e-9,
        "Γ_max=0.1 ⇒ 1.1/0.9 multiplier"
    );
    // Γ_max → 1 ⇒ Z_target → ∞; the function clamps to Z_driver for safety.
    assert!(
        (impedance_target(50.0, f64::NAN) - 50.0).abs() < 1e-12,
        "NaN Γ_max ⇒ degenerate; returns Z_driver"
    );
    // Direction: larger Γ tolerance ⇒ larger Z_target (the conventional digital-impedance
    // upward branch).
    assert!(
        impedance_target(50.0, 0.5) > impedance_target(50.0, 0.1),
        "looser Γ tolerance ⇒ larger Z_target"
    );
}

#[test]
fn channel_operating_margin_zero_at_threshold_and_one_decade_per_20db() {
    // signal == noise ⇒ COM = 0 dB (link at threshold).
    assert!(
        channel_operating_margin_db(1.0, 1.0).abs() < 1e-12,
        "equal signal/noise ⇒ 0 dB"
    );
    // 10× signal-to-noise ⇒ 20 dB COM (one-decade amplitude ratio).
    assert!(
        (channel_operating_margin_db(10.0, 1.0) - 20.0).abs() < 1e-12,
        "10× S/N ⇒ 20 dB COM"
    );
    // 100× signal-to-noise ⇒ 40 dB COM (two-decade amplitude ratio).
    assert!(
        (channel_operating_margin_db(100.0, 1.0) - 40.0).abs() < 1e-12,
        "100× S/N ⇒ 40 dB COM"
    );
    // Sub-threshold link (signal < noise) ⇒ negative COM, fails eye-mask.
    let com = channel_operating_margin_db(0.5, 1.0);
    assert!(com < 0.0, "sub-threshold link ⇒ negative COM, got {com}");
    assert!(
        (com - (-6.0206)).abs() < 1e-3,
        "0.5 S/N ratio ⇒ -6.02 dB COM"
    );
}

#[test]
fn return_loss_db_is_infinity_at_perfect_match_and_zero_at_full_mismatch() {
    // Perfect match ⇒ Γ = 0 ⇒ RL = +∞ (encoded as f64::INFINITY).
    let rl_inf = return_loss_db(50.0, 50.0);
    assert!(rl_inf.is_infinite() && rl_inf.is_sign_positive(), "matched ⇒ +∞ dB");
    // 2:1 mismatch (line = 2·driver or driver = 2·line) ⇒ |Γ| = 1/3 ⇒ RL = -20·log10(1/3)
    // ≈ 9.542 dB.
    let rl_2to1_up = return_loss_db(50.0, 100.0);
    assert!(
        (rl_2to1_up - 9.542).abs() < 1e-2,
        "Z_line = 2·Z_driver ⇒ RL ≈ 9.54 dB, got {rl_2to1_up:.3}"
    );
    let rl_2to1_down = return_loss_db(100.0, 50.0);
    assert!(
        (rl_2to1_down - rl_2to1_up).abs() < 1e-9,
        "Γ is symmetric in Z swap"
    );
    // 3:1 mismatch (line = 3·driver) ⇒ |Γ| = 0.5 ⇒ RL = -20·log10(0.5) ≈ 6.021 dB.
    let rl_3to1 = return_loss_db(50.0, 150.0);
    assert!(
        (rl_3to1 - 6.021).abs() < 1e-2,
        "Z_line = 3·Z_driver ⇒ RL ≈ 6.02 dB, got {rl_3to1:.3}"
    );
    // Open-circuit equivalent (line → ∞) ⇒ RL → 0 dB.
    let rl_open = return_loss_db(50.0, 1.0e9);
    assert!(
        rl_open.abs() < 0.01,
        "near-open line ⇒ RL ≈ 0 dB, got {rl_open:.3}"
    );
    // Short-circuit equivalent (line → 0) ⇒ RL = 0 dB.
    let rl_short = return_loss_db(50.0, oof_tiny_resistance());
    assert!(
        rl_short.abs() < 0.5,
        "near-short line ⇒ RL ≈ 0 dB, got {rl_short:.3}"
    );
}

/// Helper: a tiny line resistance (10 µΩ) used as the "short-circuit" approximation.
fn oof_tiny_resistance() -> f64 {
    1.0e-5
}

#[test]
fn ssot_distinction_pdn_target_impedance_is_separate() {
    // The slice-level `impedance_target` (here) computes the SIGNAL-LINE branching-match
    // impedance driver·(1+Γ)/(1-Γ) — for controlled-impedance signal routing.
    //
    // The PDN power-rail `crate::physics::pdn::target_impedance_ohm` computes the
    // V_tolerance / I_step target — for power-delivery decoupling.
    //
    // This test pins the SSOT distinction: feeding a typical PDN-rail pair (V_tol=0.05 V,
    // I_step=10 A ⇒ 0.005 Ω) to `impedance_target` is the WRONG signature shape — the
    // function returns its `z_driver` degenerate fallback because the Γ_max input is the
    // first ratio-form. The two functions solve different physical problems and must not
    // be substituted for each other at a call site.
    let pdn_target = crate::physics::pdn::target_impedance_ohm(0.05, 10.0);
    assert!(
        (pdn_target - 0.005).abs() < 1e-9,
        "PDN rail target = 0.05 V / 10 A = 5 mΩ, got {pdn_target}"
    );
    // The SI form takes driver Z + tolerated Γ — a 50 Ω driver tolerating 10% mismatch
    // gives 50·1.1/0.9 ≈ 61 Ω, NOT a power-rail target.
    let si_signal_target = impedance_target(50.0, 0.10);
    assert!(
        (si_signal_target - 61.111).abs() < 1e-2,
        "SI signal-line target — driver·(1+Γ)/(1-Γ), got {si_signal_target:.3}"
    );
    // And the two values must be far apart (mΩ-range PDN vs tens-of-Ω SI).
    assert!(
        si_signal_target > 100.0 * pdn_target,
        "SI/PDN targets must operate at vastly different scales; \
         got SI={si_signal_target:.3} PDN={pdn_target:.6}"
    );
}
