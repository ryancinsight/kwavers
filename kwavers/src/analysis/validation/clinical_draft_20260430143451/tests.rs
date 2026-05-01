//! Value-semantic regression tests for clinical validation.

use super::{
    ClinicalValidator, DopplerValidationThresholds, ImageQualityMetrics, MeasurementAccuracy,
    SafetyIndices,
};

#[test]
fn test_clinical_validator_creation() {
    let validator = ClinicalValidator::new();
    assert!(!validator.requirements.is_empty());
}

#[test]
fn test_bmode_validation_pass() {
    let validator = ClinicalValidator::new();

    let quality = ImageQualityMetrics {
        contrast_resolution: 35.0, // > 30 dB required
        axial_resolution: 0.3,     // < 0.5 mm required
        lateral_resolution: 0.8,   // < 1.0 mm required
        dynamic_range: 70.0,       // > 60 dB required
        snr: 25.0,                 // > 20 dB required
        cnr: 15.0,
    };

    let accuracy = MeasurementAccuracy {
        distance_error_percent: 3.0, // < 5% required
        area_error_percent: 8.0,     // < 10% required
        volume_error_percent: 12.0,
        velocity_error_percent: 8.0,
        angle_error_degrees: 3.0,
    };

    let safety = SafetyIndices {
        mechanical_index: 1.5, // < 1.9 required
        thermal_index_bone: 0.5,
        thermal_index_soft: 2.0, // < 6.0 required
        thermal_index_cranial: 0.3,
        spta_intensity: 500.0,
        sppa_intensity: 100.0,
    };

    let result = validator
        .validate_bmode(&quality, &accuracy, &safety)
        .unwrap();
    assert!(result.passed);
    assert!(result.clinical_score > 80.0);
    assert!(result.regulatory_compliant);
}

#[test]
fn test_bmode_validation_fail() {
    let validator = ClinicalValidator::new();

    let quality = ImageQualityMetrics {
        contrast_resolution: 20.0, // < 30 dB required - FAIL
        axial_resolution: 1.0,     // > 0.5 mm required - FAIL
        lateral_resolution: 1.5,   // > 1.0 mm required - FAIL
        dynamic_range: 40.0,       // < 60 dB required - FAIL
        snr: 15.0,                 // < 20 dB required - FAIL
        cnr: 10.0,
    };

    let accuracy = MeasurementAccuracy {
        distance_error_percent: 8.0, // > 5% required - FAIL
        area_error_percent: 15.0,    // > 10% required - FAIL
        volume_error_percent: 12.0,
        velocity_error_percent: 8.0,
        angle_error_degrees: 3.0,
    };

    let safety = SafetyIndices {
        mechanical_index: 1.5,
        thermal_index_bone: 0.5,
        thermal_index_soft: 2.0,
        thermal_index_cranial: 0.3,
        spta_intensity: 500.0,
        sppa_intensity: 100.0,
    };

    let result = validator
        .validate_bmode(&quality, &accuracy, &safety)
        .unwrap();
    assert!(!result.passed);
    assert!(result.clinical_score < 80.0);
    assert!(!result.issues.is_empty());
    assert!(!result.recommendations.is_empty());
}

#[test]
fn test_safety_validation() {
    let validator = ClinicalValidator::new();

    let safety = SafetyIndices {
        mechanical_index: 1.5,
        thermal_index_bone: 0.8,
        thermal_index_soft: 4.0,
        thermal_index_cranial: 0.7,
        spta_intensity: 600.0,
        sppa_intensity: 150.0,
    };

    let result = validator.validate_safety(&safety).unwrap();
    assert!(result.passed);
    assert!(result.regulatory_compliant);
    assert_eq!(result.clinical_score, 100.0);
}

#[test]
fn test_safety_validation_fail() {
    let validator = ClinicalValidator::new();

    let safety = SafetyIndices {
        mechanical_index: 2.5,   // Exceeds 1.9 limit - FAIL
        thermal_index_bone: 1.5, // Exceeds 1.0 limit - FAIL
        thermal_index_soft: 8.0, // Exceeds 6.0 limit - FAIL
        thermal_index_cranial: 0.7,
        spta_intensity: 600.0,
        sppa_intensity: 150.0,
    };

    let result = validator.validate_safety(&safety).unwrap();
    assert!(!result.passed);
    assert!(!result.regulatory_compliant);
    assert!(!result.issues.is_empty());
    assert!(!result.recommendations.is_empty());
}

#[test]
fn test_doppler_validation_pass() {
    let validator = ClinicalValidator::new();
    let safety = SafetyIndices {
        mechanical_index: 1.2,
        thermal_index_bone: 0.5,
        thermal_index_soft: 2.0,
        thermal_index_cranial: 0.3,
        spta_intensity: 400.0,
        sppa_intensity: 100.0,
    };

    let result = validator.validate_doppler(6.0, 3.0, 6.5, &safety).unwrap();

    assert!(result.passed);
    assert!(result.clinical_score > 50.0);
}

#[test]
fn test_doppler_validation_custom_thresholds() {
    let validator = ClinicalValidator::new();
    let safety = SafetyIndices {
        mechanical_index: 1.2,
        thermal_index_bone: 0.5,
        thermal_index_soft: 2.0,
        thermal_index_cranial: 0.3,
        spta_intensity: 400.0,
        sppa_intensity: 100.0,
    };

    let thresholds = DopplerValidationThresholds {
        min_sensitivity_cm_s: 10.0,
        max_velocity_error_percent: 5.0,
        max_angle_error_degrees: 2.0,
    };

    let result = validator
        .validate_doppler_with_thresholds(6.0, 3.0, 6.5, &safety, &thresholds)
        .unwrap();

    assert!(!result.passed);
    assert!(!result.issues.is_empty());
}
