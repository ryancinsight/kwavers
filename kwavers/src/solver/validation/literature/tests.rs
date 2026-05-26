use crate::core::constants::fundamental::{ACOUSTIC_ABSORPTION_TISSUE, DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER_SIM};
use crate::core::constants::tissue_acoustics::SOFT_TISSUE_ABSORPTION_POWER_Y;
use crate::core::constants::numerical::MHZ_TO_HZ;
use super::types::{treeby_2010, pinton_2009, LiteratureValidationResult};
use super::validator::LiteratureValidator;

#[test]
fn test_relative_l2_error() {
    let computed = vec![1.0, 2.0, 3.0];
    let reference = vec![1.0, 2.0, 3.0];
    assert_eq!(
        LiteratureValidator::relative_l2_error(&computed, &reference),
        0.0
    );

    let computed = vec![1.1, 2.0, 3.0];
    let error = LiteratureValidator::relative_l2_error(&computed, &reference);
    assert!(error > 0.0 && error < 0.1);
}

#[test]
fn test_linear_regression() {
    let x = vec![1.0, 2.0, 3.0];
    let y = vec![2.0, 4.0, 6.0];
    let (slope, intercept) = LiteratureValidator::linear_regression(&x, &y);
    assert!((slope - 2.0).abs() < 1e-10);
    assert!(intercept.abs() < 1e-10);
}

#[test]
fn test_treeby_parameters() {
    assert_eq!(treeby_2010::SOUND_SPEED, SOUND_SPEED_WATER_SIM);
    assert_eq!(treeby_2010::DENSITY, DENSITY_WATER_NOMINAL);
    assert!(treeby_2010::MAX_PHASE_VELOCITY_ERROR <= 0.001);
}

#[test]
fn test_pinton_parameters() {
    assert!(pinton_2009::SHEAR_SPEED > 0.0);
    assert!(pinton_2009::COMPRESSIONAL_SPEED > pinton_2009::SHEAR_SPEED);
}

#[test]
fn test_convergence_rate_analysis() {
    let dx = vec![0.1, 0.05, 0.025];
    let errors: Vec<f64> = dx.iter().map(|&x: &f64| x.powi(2)).collect();

    let validator = LiteratureValidator::new();
    let result = validator.validate_convergence_rate(&dx, &errors, 2.0);

    assert!(result.passed, "Should detect 2nd order convergence");
    assert!(result.error_metrics["observed_order"] > 1.9);
}

#[test]
fn test_absorption_power_law() {
    let freqs = vec![MHZ_TO_HZ, 2.0 * MHZ_TO_HZ, 3.0 * MHZ_TO_HZ, 4.0 * MHZ_TO_HZ, 5.0 * MHZ_TO_HZ];
    let alpha: Vec<f64> = freqs
        .iter()
        .map(|&f: &f64| ACOUSTIC_ABSORPTION_TISSUE * (f / MHZ_TO_HZ).powf(SOFT_TISSUE_ABSORPTION_POWER_Y))
        .collect();

    let validator = LiteratureValidator::new();
    let result = validator.validate_treeby_absorption(&alpha, &freqs, 1.1);

    assert!(result.error_metrics["fitted_y"] > 1.05);
    assert!(result.error_metrics["fitted_y"] < 1.15);
}

#[test]
fn validation_result_builder() {
    let mut result = LiteratureValidationResult::new("Test");
    result.with_error(0.01, 0.001);

    assert_eq!(result.relative_error, 0.01);
    assert_eq!(result.absolute_error, 0.001);
}
