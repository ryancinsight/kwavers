use super::calculator::MechanicalIndexCalculator;
use super::types::{MechanicalIndexResult, MechanicalIndexSafetyStatus, MechanicalIndexTissueType};
use kwavers_core::constants::numerical::MPA_TO_PA;
use leto::Array3;

#[test]
fn test_mi_calculation_soft_tissue() {
    let mi_calc = MechanicalIndexCalculator::new(5.0, 0.5, MechanicalIndexTissueType::SoftTissue);

    // Create pressure field with 1 MPa peak negative pressure
    let mut pressure = Array3::zeros((10, 10, 10));
    pressure[[5, 5, 5]] = -MPA_TO_PA; // -1 MPa in Pascals

    let result = mi_calc.calculate(&pressure, 5.0).unwrap();

    let expected_mi = (1.0 * 10.0_f64.powf(-(0.5 * 5.0 * 5.0) / 20.0)) / 5.0_f64.sqrt();
    assert!((result.mi - expected_mi).abs() < 1e-6);
    assert_eq!(result.center_frequency_mhz, 5.0);
    assert!(result.is_safe()); // Well below 1.9 limit
}

#[test]
fn test_mi_safety_limits() {
    assert_eq!(MechanicalIndexTissueType::SoftTissue.safety_limit(), 1.9);
    assert_eq!(MechanicalIndexTissueType::Ophthalmic.safety_limit(), 0.23);
    assert_eq!(MechanicalIndexTissueType::Lung.safety_limit(), 0.7);
}

#[test]
fn test_mi_safety_status() {
    let mi_calc = MechanicalIndexCalculator::new(1.0, 0.3, MechanicalIndexTissueType::Ophthalmic);

    // Create field that exceeds ophthalmic limit (0.23)
    let mut pressure = Array3::zeros((10, 10, 10));
    pressure[[5, 5, 5]] = -0.5 * MPA_TO_PA; // -0.5 MPa

    let result = mi_calc.calculate(&pressure, 3.0).unwrap();

    // MI = 0.5 × 10^(-0.3*1*3/20) / sqrt(1.0), exceeds 0.23 limit
    assert!(result.mi > 0.23);
    assert_eq!(result.safety_status, MechanicalIndexSafetyStatus::Unsafe);
}

#[test]
fn test_mi_depth_profile() {
    let mi_calc = MechanicalIndexCalculator::new(3.0, 0.5, MechanicalIndexTissueType::Brain);

    let mut pressure = Array3::zeros((10, 10, 10));
    pressure[[5, 5, 5]] = -0.8 * MPA_TO_PA; // -0.8 MPa

    let depths = vec![1.0, 3.0, 5.0, 7.0];
    let results = mi_calc.calculate_depth_profile(&pressure, &depths).unwrap();

    assert_eq!(results.len(), 4);
    // MI should decrease with depth due to attenuation
    assert!(results[3].mi < results[0].mi);
}

#[test]
fn test_cavitation_threshold() {
    assert!(MechanicalIndexTissueType::SoftTissue.cavitation_threshold() > 0.4);
    assert!(MechanicalIndexTissueType::Lung.cavitation_threshold() < 0.5); // Lower for gas-body tissue
}

#[test]
fn test_mi_report_format() {
    let result = MechanicalIndexResult {
        mi: 0.8,
        peak_rarefactional_pressure_mpa: 1.5,
        center_frequency_mhz: 3.5,
        safety_status: MechanicalIndexSafetyStatus::Safe,
        focal_distance_cm: 5.0,
        safety_limit: 1.9,
    };

    let report = result.format_report();
    assert!(report.contains("MI Value: 0.800"));
    assert!(report.contains("Safety Status: Safe"));
}

#[test]
fn test_calculate_max_mi_selects_shallowest_depth_under_attenuation() {
    let mi_calc = MechanicalIndexCalculator::new(4.0, 0.5, MechanicalIndexTissueType::SoftTissue);
    let mut pressure = Array3::zeros((10, 10, 10));
    pressure[[5, 5, 5]] = -MPA_TO_PA;

    let result = mi_calc.calculate_max_mi(&pressure, 6.0, 4).unwrap();

    assert_eq!(result.focal_distance_cm, 0.0);
    assert_eq!(result.safety_status, MechanicalIndexSafetyStatus::Safe);
    assert!((result.peak_rarefactional_pressure_mpa - 1.0).abs() < 1e-12);
    assert!((result.mi - 0.5).abs() < 1e-12);
}

#[test]
fn test_calculate_max_mi_rejects_single_depth_sample() {
    let mi_calc = MechanicalIndexCalculator::new(3.0, 0.5, MechanicalIndexTissueType::SoftTissue);
    let mut pressure = Array3::zeros((10, 10, 10));
    pressure[[5, 5, 5]] = -MPA_TO_PA;

    let error = mi_calc.calculate_max_mi(&pressure, 5.0, 1).unwrap_err();

    assert!(error.to_string().contains("num_points"));
}

#[test]
fn test_mi_rejects_nonpositive_frequency() {
    let mi_calc = MechanicalIndexCalculator::new(0.0, 0.5, MechanicalIndexTissueType::SoftTissue);
    let mut pressure = Array3::zeros((10, 10, 10));
    pressure[[5, 5, 5]] = -MPA_TO_PA;

    let error = mi_calc.calculate(&pressure, 1.0).unwrap_err();

    assert!(error.to_string().contains("center_frequency_mhz"));
}

#[test]
fn test_mi_rejects_negative_focal_distance() {
    let mi_calc = MechanicalIndexCalculator::new(3.0, 0.5, MechanicalIndexTissueType::SoftTissue);
    let mut pressure = Array3::zeros((10, 10, 10));
    pressure[[5, 5, 5]] = -MPA_TO_PA;

    let error = mi_calc.calculate(&pressure, -1.0).unwrap_err();

    assert!(error.to_string().contains("focal_distance_cm"));
}
