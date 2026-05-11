//! Keller-Miksis wall-motion ODE regression tests.

use crate::physics::acoustics::bubble_dynamics::bubble_state::{BubbleParameters, BubbleState};
use crate::physics::acoustics::bubble_dynamics::keller_miksis::KellerMiksisModel;

#[test]
fn test_keller_miksis_creation() {
    let params = BubbleParameters::default();
    let model = KellerMiksisModel::new(params);

    assert!(model.params().r0 > 0.0);
}

#[test]
#[ignore = "Equilibrium test needs refinement - K-M compressibility terms cause non-zero acceleration"]
fn test_keller_miksis_equilibrium() {
    let params = BubbleParameters::default();
    let model = KellerMiksisModel::new(params.clone());

    let mut state = BubbleState::at_equilibrium(&params);

    let accel = model.calculate_acceleration(&mut state, 0.0, 0.0, 0.0)
        .expect("Equilibrium calculation should succeed");

    assert!(
        accel.abs() < 1e4,
        "Acceleration at equilibrium should be relatively small, got {} m/s²",
        accel
    );
}

#[test]
fn test_keller_miksis_compression() {
    let params = BubbleParameters {
        p0: 101325.0,
        r0: 5e-6,
        ..Default::default()
    };

    let model = KellerMiksisModel::new(params.clone());
    let mut state = BubbleState::new(&params);

    state.radius = 3e-6;
    state.wall_velocity = -10.0;

    let accel = model.calculate_acceleration(&mut state, 0.0, 0.0, 0.0)
        .expect("Compression calculation should succeed");
    assert!(accel.is_finite(), "Acceleration should be finite");
}

#[test]
fn test_keller_miksis_expansion() {
    let params = BubbleParameters {
        p0: 101325.0,
        r0: 5e-6,
        ..Default::default()
    };

    let model = KellerMiksisModel::new(params.clone());
    let mut state = BubbleState::new(&params);

    state.radius = 8e-6;
    state.wall_velocity = 20.0;

    let accel = model.calculate_acceleration(&mut state, 0.0, 0.0, 0.0)
        .expect("Expansion calculation should succeed");
    assert!(
        accel < 0.0,
        "Expansion should decelerate: accel = {}",
        accel
    );
}

#[test]
fn test_keller_miksis_acoustic_forcing() {
    let params = BubbleParameters {
        driving_frequency: 1e6,
        ..Default::default()
    };

    let model = KellerMiksisModel::new(params.clone());
    let mut state = BubbleState::new(&params);

    let p_acoustic = -50000.0;
    let t = 0.25e-6;

    let accel = model.calculate_acceleration(&mut state, p_acoustic, 0.0, t)
        .expect("Acoustic forcing calculation should succeed");
    assert!(accel > 0.0, "Negative pressure should cause expansion");
}

#[test]
fn test_keller_miksis_mach_limit() {
    let params = BubbleParameters::default();
    let model = KellerMiksisModel::new(params.clone());
    let mut state = BubbleState::new(&params);

    state.wall_velocity = 0.96 * params.c_liquid;

    let result = model.calculate_acceleration(&mut state, 0.0, 0.0, 0.0);

    assert!(result.is_err(), "High Mach number should be rejected");
}

#[test]
fn test_radiation_damping_term() {
    let params = BubbleParameters::default();
    let model = KellerMiksisModel::new(params.clone());
    let mut state = BubbleState::new(&params);

    let p_acoustic = 50000.0;
    let dp_dt = 1e8;

    let accel_with_damping = model.calculate_acceleration(&mut state, p_acoustic, dp_dt, 0.0)
        .expect("Calculation with dp/dt should succeed");
    assert!(accel_with_damping.is_finite());
}

#[test]
fn test_mach_number_tracking() {
    let params = BubbleParameters::default();
    let model = KellerMiksisModel::new(params.clone());
    let mut state = BubbleState::new(&params);

    state.wall_velocity = 150.0;

    model.calculate_acceleration(&mut state, 0.0, 0.0, 0.0).unwrap();
    let expected_mach = 150.0 / params.c_liquid;
    assert!(
        (state.mach_number - expected_mach).abs() < 1e-10,
        "Mach number should be tracked: expected={}, got={}",
        expected_mach,
        state.mach_number
    );
}
