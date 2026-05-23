use super::core::ShiftingStrategy;
use super::shifter::PhaseShifter;
use approx::assert_relative_eq;
use crate::core::constants::numerical::MHZ_TO_HZ;
use ndarray::{arr2, Array2};
use std::f64::consts::PI;

fn linear_array() -> Array2<f64> {
    arr2(&[[-0.001, 0.0, 0.0], [0.0, 0.0, 0.0], [0.001, 0.0, 0.0]])
}

#[test]
fn focused_strategy_applies_spherical_phase_law() {
    let mut shifter = PhaseShifter::new(linear_array(), MHZ_TO_HZ);
    shifter.set_strategy(ShiftingStrategy::Focused);

    let phases = shifter.apply_phases(&[0.0, 0.0, 0.02]).unwrap();
    let k = 2.0 * PI / shifter.wavelength;
    let outer_distance = (0.001_f64.powi(2) + 0.02_f64.powi(2)).sqrt();
    let expected_outer = -k * (outer_distance - 0.02);

    assert_relative_eq!(phases[0], expected_outer, epsilon = 1e-12);
    assert_relative_eq!(phases[1], 0.0, epsilon = 1e-12);
    assert_relative_eq!(phases[2], expected_outer, epsilon = 1e-12);
    assert_eq!(shifter.get_phase_offsets(), &phases);
}

#[test]
fn multifocus_strategy_accepts_flat_focal_point_chunks() {
    let mut shifter = PhaseShifter::new(linear_array(), MHZ_TO_HZ);
    shifter.set_strategy(ShiftingStrategy::MultiFocus);

    let phases = shifter
        .apply_phases(&[0.0, 0.0, 0.02, 0.001, 0.0, 0.02])
        .unwrap();
    let k = 2.0 * PI / shifter.wavelength;
    let left_to_center_focus = (0.001_f64.powi(2) + 0.02_f64.powi(2)).sqrt();
    let left_to_right_focus = (0.002_f64.powi(2) + 0.02_f64.powi(2)).sqrt();
    let center_to_right_focus = (0.001_f64.powi(2) + 0.02_f64.powi(2)).sqrt();
    let reference = (0.001_f64.powi(2) + 0.02_f64.powi(2)).sqrt();

    let expected_left =
        (-k * (left_to_center_focus - 0.02) - k * (left_to_right_focus - reference)) / 2.0;
    let expected_center = (-k * (0.02 - 0.02) - k * (center_to_right_focus - reference)) / 2.0;
    let expected_right = (-k * (left_to_center_focus - 0.02) - k * (0.02 - reference)) / 2.0;

    assert_relative_eq!(phases[0], expected_left, epsilon = 1e-12);
    assert_relative_eq!(phases[1], expected_center, epsilon = 1e-12);
    assert_relative_eq!(phases[2], expected_right, epsilon = 1e-12);
}

#[test]
fn custom_strategy_sets_wrapped_phase_pattern() {
    let mut shifter = PhaseShifter::new(linear_array(), MHZ_TO_HZ);
    shifter.set_strategy(ShiftingStrategy::Custom);

    let phases = shifter.apply_phases(&[0.0, PI, 3.0 * PI]).unwrap();

    assert_relative_eq!(phases[0], 0.0, epsilon = 1e-12);
    assert_relative_eq!(phases[1], PI, epsilon = 1e-12);
    assert_relative_eq!(phases[2], PI, epsilon = 1e-12);
    assert_eq!(shifter.get_phase_offsets().len(), 3);
}

#[test]
fn linear_strategy_accepts_documented_sixty_degree_bound() {
    let mut shifter = PhaseShifter::new(linear_array(), MHZ_TO_HZ);

    shifter.set_strategy(ShiftingStrategy::Linear);
    let phases = shifter.apply_phases(&[60.0]).unwrap();
    let k = 2.0 * PI / shifter.wavelength;
    let expected_left = -k * -0.001 * 60.0_f64.to_radians().sin();

    assert_relative_eq!(phases[0], expected_left, epsilon = 1e-12);
    assert_relative_eq!(phases[1], 0.0, epsilon = 1e-12);
    assert_relative_eq!(phases[2], -expected_left, epsilon = 1e-12);
}

#[test]
fn phase_application_reuses_cached_buffer_for_equal_array_length() {
    let mut shifter = PhaseShifter::new(linear_array(), MHZ_TO_HZ);
    let phase_buffer = shifter.get_phase_offsets().as_ptr();

    let linear = shifter.apply_phases(&[15.0]).unwrap();
    assert_eq!(shifter.get_phase_offsets().as_ptr(), phase_buffer);
    assert_relative_eq!(linear[1], 0.0, epsilon = 1e-12);

    shifter.set_strategy(ShiftingStrategy::Focused);
    let focused = shifter.apply_phases(&[0.0, 0.0, 0.02]).unwrap();
    assert_eq!(shifter.get_phase_offsets().as_ptr(), phase_buffer);
    assert_relative_eq!(focused[1], 0.0, epsilon = 1e-12);

    shifter.set_strategy(ShiftingStrategy::MultiFocus);
    let multifocus = shifter
        .apply_phases(&[0.0, 0.0, 0.02, 0.001, 0.0, 0.02])
        .unwrap();
    assert_eq!(shifter.get_phase_offsets().as_ptr(), phase_buffer);
    assert_eq!(multifocus.len(), 3);

    shifter.set_strategy(ShiftingStrategy::Custom);
    let custom = shifter.apply_phases(&[0.0, PI, 3.0 * PI]).unwrap();
    assert_eq!(shifter.get_phase_offsets().as_ptr(), phase_buffer);
    assert_relative_eq!(custom[2], PI, epsilon = 1e-12);
}

#[test]
fn invalid_multifocus_input_preserves_last_valid_phase_offsets() {
    let mut shifter = PhaseShifter::new(linear_array(), MHZ_TO_HZ);
    shifter.set_strategy(ShiftingStrategy::Focused);
    let valid = shifter.apply_phases(&[0.0, 0.0, 0.02]).unwrap();

    shifter.set_strategy(ShiftingStrategy::MultiFocus);
    let err = shifter
        .apply_phases(&[0.0, 0.0, 0.02, 0.0, 0.0, 0.0005])
        .unwrap_err();

    assert!(format!("{err}").contains("Focal distance below minimum"));
    assert_eq!(shifter.get_phase_offsets(), &valid);
}

#[test]
fn strategy_input_shapes_and_domains_are_rejected_before_computation() {
    let mut shifter = PhaseShifter::new(linear_array(), MHZ_TO_HZ);

    shifter.set_strategy(ShiftingStrategy::Linear);
    let steering_error = shifter.apply_phases(&[61.0]).unwrap_err();
    assert!(format!("{steering_error}").contains("Steering angle exceeds maximum"));

    shifter.set_strategy(ShiftingStrategy::Focused);
    let focus_shape_error = shifter.apply_phases(&[0.0, 0.0]).unwrap_err();
    assert!(format!("{focus_shape_error}").contains("3D point"));
    let focus_distance_error = shifter.apply_phases(&[0.0, 0.0, 0.0005]).unwrap_err();
    assert!(format!("{focus_distance_error}").contains("1"));

    shifter.set_strategy(ShiftingStrategy::MultiFocus);
    let multifocus_error = shifter.apply_phases(&[0.0, 0.0]).unwrap_err();
    assert!(format!("{multifocus_error}").contains("3D focal points"));
    let multifocus_distance_error = shifter.apply_phases(&[0.0, 0.0, 0.0005]).unwrap_err();
    assert!(format!("{multifocus_distance_error}").contains("1"));

    shifter.set_strategy(ShiftingStrategy::Custom);
    let custom_error = shifter.apply_phases(&[0.0, 1.0]).unwrap_err();
    assert!(format!("{custom_error}").contains("requires 3 phases"));
}
