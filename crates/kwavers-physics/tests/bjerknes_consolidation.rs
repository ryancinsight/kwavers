//! KW-PHYS-009 items 1–4 consolidation acceptance (2026-11-27).
//!
//! Items 1–3 (Bjerknes sign unification, primary Bjerknes
//! consolidation, Wood sound-speed duplicate removal) and item 4
//! (Epstein–Plesset stability module rename), exercised through the
//! public API. This file lives outside the lib-test target because
//! that target does not compile as found (the unrelated, incomplete
//! `growth/` workstream — see the Fig. 4 benchmark test header).
//!
//! ## The pinned physics (item 1)
//!
//! The secondary Bjerknes scalar is the line-of-centres component of
//! the force on one bubble due to the other, positive away from the
//! other bubble: `F = −(ρ₀/(4πd²))·⟨V̇₁V̇₂⟩` (Crum LA (1975),
//! *J. Acoust. Soc. Am.* 57(2), 536–538; Leighton TG (1994), *The
//! Acoustic Bubble*, §3.4; Lanoy M et al. (2015), *Phys. Rev. E* 91,
//! 011101(R), Eq. (3) after Crum — the vector form
//! `F⃗₁→₂ ∝ (r⃗₁ − r⃗₂)ξ₁ξ₂cos(φ₁ − φ₂)` points from bubble 2 toward
//! bubble 1 for in-phase pairs). The standard result this pins:
//! **bubbles oscillating in phase attract (F < 0); in antiphase they
//! repel (F > 0).** Before the consolidation, the harmonic
//! calculator returned the attraction-positive strength — the exact
//! negative of this — while `BjerknesForceComputer::secondary`
//! returned the Crum component; both now evaluate one kernel.

use kwavers_physics::acoustics::bubble_dynamics::bjerknes_forces::BjerknesForceData;
use kwavers_physics::acoustics::bubble_dynamics::epstein_plesset_stability::OscillationType;
use kwavers_physics::acoustics::bubble_dynamics::{
    primary_bjerknes_force, secondary_bjerknes_force_checked,
    secondary_bjerknes_force_instantaneous, secondary_bjerknes_force_mean, wood_sound_speed,
    BjerknesCalculator, BjerknesConfig, BjerknesForceComputer, BjerknesInteractionType,
    BubbleParameters, BubbleState,
};

const RHO: f64 = 1000.0;

fn calculator() -> BjerknesCalculator {
    BjerknesCalculator::new(BjerknesConfig::default())
}

/// Item 1 acceptance: an in-phase pair ATTRACTS and an antiphase
/// pair REPELS, on the harmonic calculator path, the regime-checked
/// wrapper path, and the instantaneous kernel path alike.
#[test]
fn in_phase_pair_attracts_on_every_path() {
    let calc = calculator();

    // Harmonic calculator, φ = 0 (in phase).
    let in_phase: BjerknesForceData = calc
        .secondary_bjerknes_force(5e-6, 5e-6, 1e-15, 1e-15, 0.0, 10e-6)
        .unwrap();
    assert_eq!(
        in_phase.interaction_type,
        BjerknesInteractionType::Attractive
    );
    assert!(
        in_phase.secondary < 0.0,
        "in-phase pair must attract: secondary < 0, got {}",
        in_phase.secondary
    );

    // Harmonic calculator, φ = π (antiphase).
    let antiphase = calc
        .secondary_bjerknes_force(5e-6, 5e-6, 1e-15, 1e-15, std::f64::consts::PI, 10e-6)
        .unwrap();
    assert_eq!(
        antiphase.interaction_type,
        BjerknesInteractionType::Repulsive
    );
    assert!(
        antiphase.secondary > 0.0,
        "antiphase pair must repel: secondary > 0, got {}",
        antiphase.secondary
    );
    // Magnitudes are symmetric under φ → φ + π.
    assert!((in_phase.secondary + antiphase.secondary).abs() < 1e-12 * in_phase.secondary.abs());

    // Regime-checked wrapper (small-amplitude pair, inside the
    // linear regime per the module's own test parameters) carries
    // the same sign and value as the direct calculator call.
    let r0 = 5e-6;
    let dv_small = 4.0 * std::f64::consts::PI * r0 * r0 * (0.05 * r0);
    let checked =
        secondary_bjerknes_force_checked(&calc, r0, r0, dv_small, dv_small, 0.0, 50e-6).unwrap();
    assert!(
        checked.secondary < 0.0,
        "checked path must attract in phase"
    );
    let direct = calc
        .secondary_bjerknes_force(r0, r0, dv_small, dv_small, 0.0, 50e-6)
        .unwrap();
    assert_eq!(checked.secondary, direct.secondary);

    // Instantaneous kernel: same-sense volume rates attract.
    assert!(secondary_bjerknes_force_instantaneous(1e-9, 1e-9, 50e-6, RHO) < 0.0);
    assert!(secondary_bjerknes_force_instantaneous(-1e-9, -1e-9, 50e-6, RHO) < 0.0);
    assert!(secondary_bjerknes_force_instantaneous(1e-9, -1e-9, 50e-6, RHO) > 0.0);
}

/// Item 1 consistency: the cycle average of the canonical
/// instantaneous form over a synthesized harmonic cycle equals the
/// calculator's harmonic value — the two public forms are one law.
#[test]
fn instantaneous_cycle_average_matches_calculator() {
    let calc = calculator();
    let cfg = calc.config();
    let omega = std::f64::consts::TAU * cfg.frequency;
    let (dv1, dv2, d) = (1e-15, 3e-15, 40e-6);

    for phase in [0.0_f64, 0.7, std::f64::consts::PI, 4.0] {
        // V_i(t) = ΔV_i cos(ωt + φ_i) ⇒ V̇_i = −ωΔV_i sin(ωt + φ_i).
        let n = 8192usize;
        let mut sum = 0.0;
        for k in 0..n {
            let t = (k as f64) / (n as f64) / cfg.frequency;
            let v1_dot = -omega * dv1 * (omega * t).sin();
            let v2_dot = -omega * dv2 * (omega * t + phase).sin();
            sum += secondary_bjerknes_force_instantaneous(v1_dot, v2_dot, d, cfg.rho);
        }
        let averaged = sum / (n as f64);
        let via_calculator = calc
            .secondary_bjerknes_force(5e-6, 5e-6, dv1, dv2, phase, d)
            .unwrap()
            .secondary;
        let rel = (averaged - via_calculator).abs() / via_calculator.abs();
        assert!(
            rel < 1e-6,
            "cycle average {averaged:e} vs calculator {via_calculator:e} at phase {phase}: rel {rel:e}"
        );

        // And the mean-kernel form agrees with the calculator exactly.
        let mean_product = 0.5 * omega * omega * dv1 * dv2 * phase.cos();
        let via_kernel = secondary_bjerknes_force_mean(mean_product, d, cfg.rho);
        assert_eq!(via_kernel, via_calculator);
    }
}

/// Item 1 delegation: `BjerknesForceComputer::secondary` (the
/// instantaneous path) is the canonical kernel applied to
/// `V̇_i = 4πR_i²Ṙ_i`, and its classification uses the canonical
/// interaction-type enum.
#[test]
fn interactions_secondary_delegates_to_canonical_kernel() {
    let params = BubbleParameters::default();
    let mut b1 = BubbleState::new(&params);
    let mut b2 = BubbleState::new(&params);
    b1.radius = 2.0e-6;
    b2.radius = 3.0e-6;
    let d = 1.0e-4;

    for (w1, w2) in [(0.5, 0.4), (0.5, -0.4), (-0.5, -0.4)] {
        b1.wall_velocity = w1;
        b2.wall_velocity = w2;
        let via_computer = BjerknesForceComputer::secondary(&b1, &b2, d, RHO);
        let v1_dot = 4.0 * std::f64::consts::PI * b1.radius.powi(2) * w1;
        let v2_dot = 4.0 * std::f64::consts::PI * b2.radius.powi(2) * w2;
        let via_kernel = secondary_bjerknes_force_instantaneous(v1_dot, v2_dot, d, RHO);
        assert_eq!(via_computer, via_kernel);
        // Same-sense wall motion attracts.
        if w1 * w2 > 0.0 {
            assert!(via_computer < 0.0, "in-phase walls must attract");
            assert_eq!(
                BjerknesForceComputer::interaction_type(&b1, &b2),
                BjerknesInteractionType::Attractive
            );
        } else {
            assert!(via_computer > 0.0, "antiphase walls must repel");
            assert_eq!(
                BjerknesForceComputer::interaction_type(&b1, &b2),
                BjerknesInteractionType::Repulsive
            );
        }
    }
    // Non-positive distance: kernel guard, no pair force.
    assert_eq!(BjerknesForceComputer::secondary(&b1, &b2, 0.0, RHO), 0.0);
}

/// Item 2 acceptance: one canonical primary Bjerknes law
/// `F = −V·∇p`; the calculator and the interactions computer both
/// evaluate it.
#[test]
fn primary_bjerknes_single_law_all_sites() {
    let calc = calculator();
    let radius: f64 = 5e-6;
    let grad: f64 = 1e9;
    let volume = (4.0 * std::f64::consts::PI / 3.0) * radius.powi(3);

    let canonical = primary_bjerknes_force(volume, grad);
    // The calculator method's middle argument is an unused legacy
    // amplitude slot (documented as such in primary.rs).
    let via_calculator = calc.primary_bjerknes_force(radius, 0.0, grad).unwrap();
    let via_computer = BjerknesForceComputer::primary(volume, grad);
    assert_eq!(canonical, via_calculator);
    assert_eq!(canonical, via_computer);
    // Physics: a bubble is pushed down the pressure gradient.
    assert_eq!(canonical, -volume * grad);
    assert!(primary_bjerknes_force(volume, grad) < 0.0);
    assert!(primary_bjerknes_force(volume, -grad) > 0.0);
}

/// Item 3 acceptance: the canonical guarded Wood sound speed is the
/// surviving implementation (the unguarded `interactions` duplicate
/// with the transposed argument order was removed).
#[test]
fn wood_sound_speed_canonical_guarded() {
    // β = 0 returns the liquid sound speed exactly.
    assert_eq!(wood_sound_speed(0.0, 1480.0, 1000.0, 343.0, 1.2), 1480.0);
    // 1% void fraction collapses the sound speed (Wood's equation).
    let c = wood_sound_speed(0.01, 1480.0, 1000.0, 343.0, 1.2);
    assert!(c < 1480.0 && c > 50.0, "1% void fraction: c = {c}");
    // Guard: non-positive inputs fall back to the liquid speed
    // instead of NaN (the removed duplicate returned NaN here).
    assert_eq!(wood_sound_speed(0.01, 1480.0, 0.0, 343.0, 1.2), 1480.0);
    assert_eq!(wood_sound_speed(0.01, 1480.0, 1000.0, 343.0, 0.0), 1480.0);
}

/// Item 4 acceptance: the stability analysis is reachable under its
/// disambiguated module path, distinct from the dissolution law's
/// `epstein_plesset_*` functions in `super::dissolution`.
#[test]
fn epstein_plesset_stability_path_disambiguated() {
    // The enum is importable from the renamed module (this import is
    // the acceptance: the old `epstein_plesset::` path no longer
    // resolves), and the solver stays re-exported at the
    // bubble_dynamics level under its unchanged name.
    let _ = OscillationType::StableHarmonic;
    let _solver = std::marker::PhantomData::<
        kwavers_physics::acoustics::bubble_dynamics::EpsteinPlessetStabilitySolver,
    >;
    let _results = std::marker::PhantomData::<
        kwavers_physics::acoustics::bubble_dynamics::epstein_plesset_stability::EpsteinPlessetStabilityValidationResults,
    >;
}
