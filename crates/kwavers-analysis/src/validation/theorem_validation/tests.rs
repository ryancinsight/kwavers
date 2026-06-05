use super::*;
use kwavers_core::constants::fundamental::SOUND_SPEED_WATER_SIM;

#[test]
fn test_beer_lambert_validation() {
    let distances = vec![1.0, 2.0, 3.0];
    let intensities = vec![0.9048, 0.8187, 0.7408];
    let result = TheoremValidator::validate_beer_lambert_law(1.0, 0.1, &distances, &intensities);

    assert!(result.passed);
    assert!(result.measured_error < 0.01);
}

#[test]
fn test_cfl_validation() {
    let dt = 1e-8;
    let dx = 5e-5;
    let c = SOUND_SPEED_WATER_SIM;
    let dimensions = 3;

    let result = TheoremValidator::validate_cfl_condition(dt, dx, c, dimensions);

    assert!(
        result.passed,
        "CFL condition should pass for conservative timestep"
    );
    assert!(
        result.measured_error < 0.5,
        "CFL number should be < 0.5 for stability"
    );

    let unstable_dt = 1e-7;
    let result_unstable = TheoremValidator::validate_cfl_condition(unstable_dt, dx, c, dimensions);
    assert!(
        !result_unstable.passed,
        "Large timestep should violate CFL condition"
    );
}

#[test]
fn test_comprehensive_validation() {
    let validator = TheoremValidator;
    let results = validator.run_comprehensive_validation();

    assert!(!results.is_empty());
    assert!(results.len() >= 8);

    let pass_rate = results.iter().filter(|r| r.passed).count() as f64 / results.len() as f64;
    assert!(pass_rate >= 0.5);
}

#[test]
fn test_validation_report() {
    let validator = TheoremValidator;
    let validations = vec![TheoremValidation {
        theorem: "Test Theorem".to_string(),
        passed: true,
        error_bound: 1e-6,
        measured_error: 1e-7,
        confidence: 0.95,
        details: "Test validation".to_string(),
    }];

    let report = validator.generate_validation_report(&validations);
    assert!(report.contains("Theorem Validation Report"));
    assert!(report.contains("✅ PASS"));
    assert!(report.contains("95%"));
}
