//! Tests for IMEX bubble integration

use super::*;
use crate::physics::acoustics::bubble_dynamics::{BubbleParameters, BubbleState, KellerMiksisModel};
use std::sync::Arc;

#[test]
#[ignore = "Requires Sprint 111+ Keller-Miksis full implementation (PRD FR-014)"]
fn test_imex_integration() {
    let params = BubbleParameters::default();
    let solver = Arc::new(KellerMiksisModel::new(params.clone()));
    let mut state = BubbleState::new(&params);

    let config = BubbleIMEXConfig::default();
    let mut integrator = BubbleIMEXIntegrator::new(solver, config);

    let result = integrator.step(
        &mut state, 1e5, 0.0, 1e-9, 0.0,
    );

    assert!(result.is_ok());
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

#[test]
#[ignore = "Requires Sprint 111+ Keller-Miksis full implementation (PRD FR-014)"]
fn test_adaptive_epsilon() {
    let params = BubbleParameters::default();
    let solver = Arc::new(KellerMiksisModel::new(params.clone()));

    let mut small_state = BubbleState::new(&params);
    small_state.radius = 1e-9;

    let mut large_state = BubbleState::new(&params);
    large_state.radius = 1e-3;

    let mut integrator = BubbleIMEXIntegrator::with_defaults(solver);

    let small_result = integrator.step(&mut small_state, 0.0, 0.0, 1e-12, 0.0);
    assert!(small_result.is_ok());

    let large_result = integrator.step(&mut large_state, 0.0, 0.0, 1e-6, 0.0);
    assert!(large_result.is_ok());
}
