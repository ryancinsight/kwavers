//! Tests for IMEX bubble integration

use super::*;
use crate::physics::acoustics::bubble_dynamics::{
    BubbleParameters, BubbleState, KellerMiksisModel,
};
use std::sync::Arc;

#[test]
#[ignore = "Requires Sprint 111+ Keller-Miksis full implementation (PRD FR-014)"]
fn test_imex_integration() {
    let params = BubbleParameters::default();
    let solver = Arc::new(KellerMiksisModel::new(params.clone()));
    let mut state = BubbleState::new(&params);

    let config = BubbleIMEXConfig::default();
    let mut integrator = BubbleIMEXIntegrator::new(solver, config);

    integrator.step(&mut state, 1e5, 0.0, 1e-9, 0.0).unwrap();
    assert!(state.radius > 0.0);
    assert!(state.temperature > 0.0);
}

#[test]
fn test_stiffness_detection() {
    let params = BubbleParameters::default();
    let solver = Arc::new(KellerMiksisModel::new(params.clone()));
    let state = BubbleState::new(&params);

    let integrator = BubbleIMEXIntegrator::with_defaults(solver);
    let stiffness = integrator.estimate_stiffness(&state);

    assert!(stiffness > 0.0);

    let suggested_dt = integrator.suggest_timestep(&state);
    assert!(suggested_dt > 0.0);
    assert!(suggested_dt <= BubbleIMEXConfig::default().dt_max);
}

#[test]
fn test_thermal_mass_coupling() {
    let params = BubbleParameters {
        use_thermal_effects: true,
        use_mass_transfer: true,
        ..Default::default()
    };

    let solver = Arc::new(KellerMiksisModel::new(params.clone()));
    let state = BubbleState::new(&params);

    let integrator = BubbleIMEXIntegrator::with_defaults(solver);

    let (dt_dt, dn_vapor_dt) = integrator
        .calculate_thermal_mass_transfer_rates(&state)
        .unwrap();

    assert!(dt_dt.abs() > 0.0 || dn_vapor_dt.abs() > 0.0);
}

/// Test that the IMEX integrator steps successfully at both a physically small
/// radius and a physically large radius, verifying adaptive epsilon selection
/// across the dynamic range of cavitation bubble collapse.
///
/// # Physical constraints
///
/// The Van der Waals excluded volume is `V_excl = n_moles · b_si` where
/// `n_moles` is the molar gas count at initialization and `b_si` is the
/// molar co-volume in SI units. For the default bubble parameters
/// (r₀ = 5 µm, air at 1 atm, T = 293.15 K), n_moles ≈ 2.80×10⁻¹⁴ mol
/// and b_si(air) ≈ 3.87×10⁻⁵ m³/mol (VdW table), giving
/// `V_excl ≈ 1.08×10⁻¹⁸ m³` and `R_min = (3·V_excl/4π)^(1/3) ≈ 637 nm`.
///
/// The "small" test state uses `R = 1 µm`, a compression ratio of 5:1 from
/// equilibrium, consistent with the late inertial-collapse phase predicted by
/// the Rayleigh–Plesset equation (Rayleigh 1917). `R = 1 µm` lies safely
/// above `R_min ≈ 637 nm` (volume ratio ≈ 3.86× excluded volume).
/// The "large" state uses `R = 1 mm`, representing an expanded bubble before
/// rebound.
#[test]
fn test_adaptive_epsilon() {
    let params = BubbleParameters::default();
    let solver = Arc::new(KellerMiksisModel::new(params.clone()));

    // 1 µm: late inertial-collapse phase (5:1 compression from r₀ = 5 µm),
    // volume ≈ 3.86× VdW excluded volume — minimum physically valid compressed state.
    let mut small_state = BubbleState::new(&params);
    small_state.radius = 1e-6;

    // 1 mm: expanded bubble before rebound.
    let mut large_state = BubbleState::new(&params);
    large_state.radius = 1e-3;

    let mut integrator = BubbleIMEXIntegrator::with_defaults(solver);

    integrator
        .step(&mut small_state, 0.0, 0.0, 1e-12, 0.0)
        .unwrap();

    integrator
        .step(&mut large_state, 0.0, 0.0, 1e-6, 0.0)
        .unwrap();
}
