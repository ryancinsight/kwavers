use super::*;
use kwavers_core::constants::numerical::{MHZ_TO_HZ, MPA_TO_PA};
use kwavers_core::constants::thermodynamic::BODY_TEMPERATURE_C;
use kwavers_core::error::{KwaversError, ValidationError};
use leto::Array3;

/// Assert the error is the safety ConstraintViolation variant and return its
/// message so rejection tests pin the *reason*, not merely the presence of an
/// error.
fn expect_constraint_violation(err: KwaversError) -> String {
    match err {
        KwaversError::Validation(ValidationError::ConstraintViolation { message }) => message,
        other => panic!("expected ConstraintViolation, got {other:?}"),
    }
}

#[test]
fn test_safety_monitor_creation() {
    let monitor = TranscranialSafetyMonitor::new((16, 16, 16), 0.01, 650e3);
    assert_eq!(monitor.temperature.shape(), [16, 16, 16]);
}

#[test]
fn test_safety_level_classification() {
    assert_eq!(
        TranscranialSafetyLevel::from_value(0.5, 1.0),
        TranscranialSafetyLevel::Safe
    );
    assert_eq!(
        TranscranialSafetyLevel::from_value(0.85, 1.0),
        TranscranialSafetyLevel::Monitor
    );
    assert_eq!(
        TranscranialSafetyLevel::from_value(0.95, 1.0),
        TranscranialSafetyLevel::Warning
    );
    assert_eq!(
        TranscranialSafetyLevel::from_value(1.1, 1.0),
        TranscranialSafetyLevel::Critical
    );
}

#[test]
fn test_mechanical_index_calculation() {
    let mut monitor = TranscranialSafetyMonitor::new((8, 8, 8), 0.01, MHZ_TO_HZ);
    let temperature = Array3::from_elem((8, 8, 8), BODY_TEMPERATURE_C);
    let mut pressure = Array3::zeros([8, 8, 8]);
    pressure[[4, 4, 4]] = MPA_TO_PA; // 1 MPa

    monitor.update_fields(&temperature, &pressure, 0.1).unwrap();

    // MI should be approximately 1.0 for 1 MPa at 1 MHz
    assert!((monitor.mechanical_index.current_mi - 1.0).abs() < 1e-12);
    assert!((monitor.mechanical_index.safety_margin - 1.9).abs() < 1e-12);
}

#[test]
fn test_mechanical_index_uses_pressure_magnitude() {
    let mut monitor = TranscranialSafetyMonitor::new((8, 8, 8), 0.01, MHZ_TO_HZ);
    let temperature = Array3::from_elem((8, 8, 8), BODY_TEMPERATURE_C);
    let mut pressure = Array3::zeros([8, 8, 8]);
    pressure[[4, 4, 4]] = -MPA_TO_PA;

    monitor.update_fields(&temperature, &pressure, 0.1).unwrap();

    assert!((monitor.mechanical_index.current_mi - 1.0).abs() < 1e-12);
    assert!((monitor.mechanical_index.peak_pressure - 1.0).abs() < 1e-12);
}

#[test]
fn test_mechanical_index_invalid_frequency_fails_closed() {
    let mut monitor = TranscranialSafetyMonitor::new((8, 8, 8), 0.01, 0.0);
    let temperature = Array3::from_elem((8, 8, 8), BODY_TEMPERATURE_C);
    let mut pressure = Array3::zeros([8, 8, 8]);
    pressure[[4, 4, 4]] = MPA_TO_PA;

    let message = expect_constraint_violation(
        monitor
            .update_fields(&temperature, &pressure, 0.1)
            .expect_err("invalid frequency must fail closed"),
    );
    assert!(
        message.contains("Mechanical index"),
        "rejection must come from the mechanical-index limit: {message}"
    );
    assert!(monitor.mechanical_index.current_mi.is_infinite());
    assert_eq!(monitor.mechanical_index.safety_margin, 0.0);
}

#[test]
fn test_mechanical_index_nonfinite_pressure_fails_closed() {
    let mut monitor = TranscranialSafetyMonitor::new((8, 8, 8), 0.01, MHZ_TO_HZ);
    let temperature = Array3::from_elem((8, 8, 8), BODY_TEMPERATURE_C);
    let mut pressure = Array3::zeros([8, 8, 8]);
    pressure[[4, 4, 4]] = f64::NAN;

    let message = expect_constraint_violation(
        monitor
            .update_fields(&temperature, &pressure, 0.1)
            .expect_err("non-finite pressure must fail closed"),
    );
    assert!(
        message.contains("Mechanical index"),
        "rejection must come from the mechanical-index limit: {message}"
    );
    assert!(monitor.mechanical_index.current_mi.is_infinite());
    assert!(monitor.mechanical_index.peak_pressure.is_infinite());
    assert_eq!(monitor.mechanical_index.safety_margin, 0.0);
}

#[test]
fn test_thermal_dose_accumulation() {
    let mut monitor = TranscranialSafetyMonitor::new((4, 4, 4), 0.01, 650e3);
    let mut temperature = Array3::from_elem((4, 4, 4), BODY_TEMPERATURE_C);
    temperature[[2, 2, 2]] = 42.0; // Hot spot below safety limit (43°C)
    let pressure = Array3::zeros([4, 4, 4]);

    monitor
        .update_fields(&temperature, &pressure, 1.0)
        .expect("Update should succeed with safe temperature");

    // Thermal dose should accumulate
    assert!(monitor.thermal_dose.current_dose[[2, 2, 2]] > 0.0);
}

#[test]
fn test_safety_limit_checking() {
    let mut monitor = TranscranialSafetyMonitor::new((4, 4, 4), 0.01, 650e3);
    let mut temperature = Array3::from_elem((4, 4, 4), BODY_TEMPERATURE_C);
    temperature[[2, 2, 2]] = 50.0; // Above limit
    let pressure = Array3::zeros([4, 4, 4]);

    let message = expect_constraint_violation(
        monitor
            .update_fields(&temperature, &pressure, 1.0)
            .expect_err("temperature above the safety limit must be rejected"),
    );
    assert!(
        message.contains("Temperature"),
        "rejection must name the temperature limit: {message}"
    );
}
