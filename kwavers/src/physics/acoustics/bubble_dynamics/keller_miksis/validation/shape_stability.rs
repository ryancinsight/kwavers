//! Shape-stability integration tests for the Keller-Miksis model.
//!
//! Validates Plesset-Prosperetti mode coupling, breakup detection, collapse-
//! driven mode growth, and capillary boundedness at rest of the symplectic
//! shape-mode integrator.

use crate::physics::acoustics::bubble_dynamics::bubble_state::{BubbleParameters, BubbleState};
use crate::physics::acoustics::bubble_dynamics::keller_miksis::KellerMiksisModel;

/// A freshly constructed model must have shape modes seeded and the bubble
/// must not be immediately flagged as unstable at equilibrium radius.
///
/// Rationale: seed amplitude = 1 Å ≪ 0.3 × R₀ ≈ 1.5 µm for a 5 µm bubble.
/// # Panics
/// - Panics if assertion fails: `BubbleState::is_shape_unstable must default to false`.
///
#[test]
fn test_shape_modes_seeded_but_stable_at_equilibrium() {
    let params = BubbleParameters::default();
    let model = KellerMiksisModel::new(params.clone());
    let state = BubbleState::new(&params);

    assert!(
        !model.is_shape_unstable(state.radius),
        "Freshly seeded shape modes must not trigger instability at equilibrium"
    );
    assert!(
        !state.is_shape_unstable,
        "BubbleState::is_shape_unstable must default to false"
    );
}

/// `update_shape_stability` must set `state.is_shape_unstable = true` when
/// the shape modes are forcibly grown beyond the breakup threshold.
///
/// Method: Directly set mode n=2 amplitude to 35% of the bubble radius
/// (> 30% Plesset threshold), then call `update_shape_stability` with a
/// zero timestep so no further integration occurs.
/// # Panics
/// - Panics if assertion fails: `Amplitude 35%·R must set is_shape_unstable = true (threshold = 30%·R)`.
///
#[test]
fn test_update_shape_stability_detects_breakup() {
    let params = BubbleParameters::default();
    let mut model = KellerMiksisModel::new(params.clone());
    let mut state = BubbleState::new(&params);

    model.shape_modes.amplitude[0] = 0.35 * state.radius;

    model.update_shape_stability(&mut state, 0.0);

    assert!(
        state.is_shape_unstable,
        "Amplitude 35%·R must set is_shape_unstable = true (threshold = 30%·R)"
    );
}

/// Under large inertial acceleration (model of violent bubble collapse),
/// 100 timesteps must grow the n=2 mode from 1 Å seed, confirming coupling
/// of R̈ → shape mode growth via the Plesset-Prosperetti driving term G_n.
///
/// Reference: Brennen (1995) §3.2 — G₂ = (n-1)[R̈/R − …] > 0 during collapse.
/// # Panics
/// - Panics if assertion fails: `Mode n=2 must grow during violent collapse: a_final={:.3e} m, a0={:.3e} m`.
///
#[test]
fn test_shape_modes_grow_during_violent_collapse() {
    let params = BubbleParameters::default();
    let mut model = KellerMiksisModel::new(params.clone());
    let mut state = BubbleState::new(&params);

    state.radius = 1.0e-7;
    state.wall_velocity = -100.0;
    state.wall_acceleration = 1.0e12;

    let dt = 1.0e-12;
    let a0 = model.shape_modes.amplitude[0].abs();

    for _ in 0..100 {
        model.update_shape_stability(&mut state, dt);
    }

    let a_final = model.shape_modes.amplitude[0].abs();
    assert!(
        a_final > a0,
        "Mode n=2 must grow during violent collapse: a_final={:.3e} m, a0={:.3e} m",
        a_final,
        a0
    );
}

/// At rest (Ṙ=0, R̈=0), shape modes must remain bounded (capillary oscillation
/// only) over 1000 steps; this validates the inviscid stability of the
/// symplectic Euler integrator used in `advance_shape_modes`.
///
/// Tolerance: 5% growth (symplectic Euler has O(h²) energy drift).
///
/// Reference: Leimkuhler & Reich (2004) §VIII.2.
/// # Panics
/// - Panics if assertion fails: `Mode n=2 must remain bounded at rest: a_final/a0={:.4}`.
///
#[test]
fn test_shape_modes_bounded_at_rest() {
    let params = BubbleParameters::default();
    let mut model = KellerMiksisModel::new(params.clone());
    let mut state = BubbleState::new(&params);

    model.shape_modes.amplitude[0] = 1.0e-8 * state.radius;
    let a0 = model.shape_modes.amplitude[0].abs();

    let omega2 = (8.0 * params.sigma / (params.rho_liquid * state.radius.powi(3))).sqrt();
    let period = 2.0 * std::f64::consts::PI / omega2;
    let dt = period / 100.0;

    for _ in 0..1000 {
        model.update_shape_stability(&mut state, dt);
    }

    let a_final = model.shape_modes.amplitude[0].abs();
    assert!(
        a_final <= 1.05 * a0,
        "Mode n=2 must remain bounded at rest: a_final/a0={:.4}",
        a_final / a0
    );
}
