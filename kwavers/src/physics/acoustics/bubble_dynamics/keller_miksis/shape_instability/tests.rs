use super::*;

#[test]
fn test_default_state_is_zero() {
    let state = ShapeModeState::default();

    for k in 0..N_MODES {
        assert_eq!(state.amplitude[k], 0.0);
        assert_eq!(state.rate[k], 0.0);
    }
}

#[test]
fn test_seed_mode2() {
    let mut state = ShapeModeState::default();
    state.seed(2, 1.0e-10);

    assert_eq!(state.amplitude[0], 1.0e-10);
    assert_eq!(state.amplitude[1], 0.0);
}

#[test]
fn test_is_unstable_triggered() {
    let mut state = ShapeModeState::default();
    let r = 1.0e-5;
    state.amplitude[0] = 0.31 * r;

    assert!(state.is_unstable(r));
    assert!(state.max_normalised_amplitude(r) > BREAKUP_FRACTION);
}

#[test]
fn test_is_unstable_false_for_small_perturbation() {
    let mut state = ShapeModeState::default();
    let r = 1.0e-5;
    state.amplitude[0] = 0.01 * r;

    assert!(!state.is_unstable(r));
    assert_eq!(state.max_normalised_amplitude(r), 0.01);
}

#[test]
fn test_capillary_oscillation_bounded() {
    let r0: f64 = 1.0e-4;
    let sigma = 0.072;
    let rho_l = 998.0;
    let nu = 0.0;
    let a0 = 1.0e-8 * r0;
    let mut modes = ShapeModeState::default();
    modes.seed(2, a0);

    let omega2 = (8.0 * sigma / (rho_l * r0.powi(3))).sqrt();
    let period = 2.0 * std::f64::consts::PI / omega2;
    let dt = period / 100.0;

    for _ in 0..1000 {
        advance_shape_modes(&mut modes, r0, 0.0, 0.0, sigma, rho_l, nu, dt);
    }

    let a_final = modes.amplitude[0].abs();
    assert!(
        a_final <= 1.05 * a0,
        "capillary oscillation amplitude ratio = {:.4}",
        a_final / a0
    );
}

#[test]
fn test_inertial_growth_during_collapse() {
    let r = 1.0e-6;
    let r_dot = -100.0;
    let r_ddot = 1.0e12;
    let sigma = 0.072;
    let rho_l = 998.0;
    let nu = 1.0e-6;
    let dt = 1.0e-12;
    let a0 = 1.0e-10;
    let mut modes = ShapeModeState::default();
    modes.seed(2, a0);

    for _ in 0..100 {
        advance_shape_modes(&mut modes, r, r_dot, r_ddot, sigma, rho_l, nu, dt);
    }

    assert!(modes.amplitude[0].abs() > a0);
}

#[test]
fn test_jet_speed_none_far_from_wall() {
    let speed = jet_speed(3.0, 101325.0, 2340.0, 998.0, 1500.0);
    assert_eq!(speed, None);
}

#[test]
fn test_jet_speed_some_near_wall() {
    let speed = jet_speed(1.0, 101325.0, 2340.0, 998.0, 1500.0)
        .expect("stand-off below critical should form a jet");

    assert!(speed > 0.0);
    assert!(speed <= 1500.0);
}

#[test]
fn test_jet_speed_increases_nearer_wall() {
    let v_far =
        jet_speed(1.5, 101325.0, 2340.0, 998.0, 1500.0).expect("stand-off 1.5 should form a jet");
    let v_near =
        jet_speed(0.8, 101325.0, 2340.0, 998.0, 1500.0).expect("stand-off 0.8 should form a jet");

    assert!(v_near > v_far);
}
