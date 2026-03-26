use super::*;
use ndarray::Array3;

#[test]
fn test_safety_monitor_creation() {
    let monitor = SafetyMonitor::new((16, 16, 16), 0.01, 650e3);
    assert_eq!(monitor.temperature.dim(), (16, 16, 16));
}

#[test]
fn test_safety_level_classification() {
    assert_eq!(SafetyLevel::from_value(0.5, 1.0), SafetyLevel::Safe);
    assert_eq!(SafetyLevel::from_value(0.85, 1.0), SafetyLevel::Monitor);
    assert_eq!(SafetyLevel::from_value(0.95, 1.0), SafetyLevel::Warning);
    assert_eq!(SafetyLevel::from_value(1.1, 1.0), SafetyLevel::Critical);
}

#[test]
fn test_mechanical_index_calculation() {
    let mut monitor = SafetyMonitor::new((8, 8, 8), 0.01, 1e6);
    let temperature = Array3::from_elem((8, 8, 8), 37.0);
    let mut pressure = Array3::zeros((8, 8, 8));
    pressure[[4, 4, 4]] = 1e6; // 1 MPa

    let result = monitor.update_fields(&temperature, &pressure, 0.1);
    assert!(result.is_ok());

    // MI should be approximately 1.0 for 1 MPa at 1 MHz
    assert!(monitor.mechanical_index.current_mi > 0.0);
}

#[test]
fn test_thermal_dose_accumulation() {
    let mut monitor = SafetyMonitor::new((4, 4, 4), 0.01, 650e3);
    let mut temperature = Array3::from_elem((4, 4, 4), 37.0);
    temperature[[2, 2, 2]] = 42.0; // Hot spot below safety limit (43°C)
    let pressure = Array3::zeros((4, 4, 4));

    let result = monitor.update_fields(&temperature, &pressure, 1.0);
    assert!(
        result.is_ok(),
        "Update should succeed with safe temperature"
    );

    // Thermal dose should accumulate
    assert!(monitor.thermal_dose.current_dose[[2, 2, 2]] > 0.0);
}

#[test]
fn test_safety_limit_checking() {
    let mut monitor = SafetyMonitor::new((4, 4, 4), 0.01, 650e3);
    let mut temperature = Array3::from_elem((4, 4, 4), 37.0);
    temperature[[2, 2, 2]] = 50.0; // Above limit
    let pressure = Array3::zeros((4, 4, 4));

    let result = monitor.update_fields(&temperature, &pressure, 1.0);
    assert!(result.is_err()); // Should fail safety check
}
