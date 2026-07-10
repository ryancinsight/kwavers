use super::*;
use leto::Array3;

#[test]
fn scattering_potential_matches_slowness_square_contract() {
    let slowness = Array3::from_shape_vec([2, 1, 1], vec![0.0005, 0.00025]).unwrap();
    let potential = real_scattering_potential(4.0, &slowness, 0.00025).unwrap();

    assert_eq!((potential.len()), 2);
    assert_eq!(
        potential[0],
        16.0 * (0.0005_f64.powi(2) - 0.00025_f64.powi(2))
    );
    assert_eq!(potential[1], 0.0);
}

#[test]
fn pstd_scattering_potential_uses_leapfrog_temporal_mass_symbol() {
    let slowness = Array3::from_shape_vec([2, 1, 1], vec![0.0005, 0.00025]).unwrap();
    let omega = 4.0;
    let time_step = 0.25;
    let temporal_mass = pstd_temporal_angular_frequency_squared(omega, time_step).unwrap();
    let potential = real_pstd_scattering_potential(omega, time_step, &slowness, 0.00025).unwrap();
    let continuous = real_scattering_potential(omega, &slowness, 0.00025).unwrap();

    assert_eq!(
        potential[0],
        temporal_mass * (0.0005_f64.powi(2) - 0.00025_f64.powi(2))
    );
    assert_eq!(potential[1], 0.0);
    assert_ne!(potential[0], continuous[0]);
}

#[test]
fn operator_scattering_derivative_factor_matches_selected_mass_symbol() {
    let omega = 4.0;
    let time_step = 0.25;
    let operator = GreenOperatorKind::SpectralPstdPeriodic {
        time_step_s: time_step,
        reference_sound_speed_m_s: SOUND_SPEED_WATER_SIM,
        temporal_transfer: None,
        absorbing_boundary: AbsorbingBoundary::disabled(),
    };
    let expected = 2.0 * pstd_temporal_angular_frequency_squared(omega, time_step).unwrap();
    let actual = scattering_slowness_derivative_factor_for_operator(omega, operator).unwrap();

    assert_eq!(actual, expected);
    assert_ne!(actual, 2.0 * omega * omega);
}

#[test]
fn convergence_epsilon_bounds_potential_norm() {
    let potential = [-3.0, 2.0, 0.25];
    let epsilon = convergence_epsilon(&potential).unwrap();

    assert_eq!(epsilon, 3.0);
    assert!(potential.iter().all(|value| value.abs() <= epsilon));
}

#[test]
fn shifted_potential_has_negative_imaginary_part() {
    let shifted = shifted_potential(&[2.0, -1.0], 3.0).unwrap();

    assert_eq!(shifted[0], Complex64::new(2.0, -3.0));
    assert_eq!(shifted[1], Complex64::new(-1.0, -3.0));
}

#[test]
fn pointwise_preconditioner_is_finite_for_valid_shift() {
    let shifted = shifted_potential(&[0.0, 2.0], 2.0).unwrap();
    let gamma = pointwise_preconditioner(&shifted, 2.0).unwrap();

    assert_eq!(gamma[0], Complex64::new(-1.0, 0.0));
    assert!(gamma[1].re.is_finite());
    assert!(gamma[1].im.is_finite());
}
