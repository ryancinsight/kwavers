use super::calculator::ThermalIndexCalculator;
use super::types::{ThermalIndexModel, ThermalIndexStatus};

#[test]
fn test_thermal_index_power_ratio_without_attenuation() {
    let calc = ThermalIndexCalculator::new(ThermalIndexModel::SoftTissue, 5.0, 0.0, 0.25, 6.0);

    let result = calc.calculate(0.5, 0.0).unwrap();

    assert_eq!(result.model, ThermalIndexModel::SoftTissue);
    assert_eq!(result.model.label(), "TIS");
    assert_eq!(result.safety_status, ThermalIndexStatus::Safe);
    assert!((result.thermal_index - 2.0).abs() < 1e-12);
    assert!((result.derated_acoustic_power_w - 0.5).abs() < 1e-12);
    assert!(result.is_safe());
}

#[test]
fn test_thermal_index_uses_power_derating_not_amplitude_derating() {
    let calc = ThermalIndexCalculator::new(ThermalIndexModel::Bone, 2.0, 0.5, 0.1, 6.0);

    let result = calc.calculate(0.4, 3.0).unwrap();

    let expected_power = 0.4 * 10.0_f64.powf(-(0.5 * 2.0 * 3.0) / 10.0);
    let expected_ti = expected_power / 0.1;
    assert_eq!(result.model.label(), "TIB");
    assert!((result.derated_acoustic_power_w - expected_power).abs() < 1e-12);
    assert!((result.thermal_index - expected_ti).abs() < 1e-12);
}

#[test]
fn test_thermal_index_status_thresholds() {
    let calc = ThermalIndexCalculator::new(ThermalIndexModel::CranialBone, 3.0, 0.0, 1.0, 6.0);

    let caution = calc.calculate(4.9, 0.0).unwrap();
    let unsafe_result = calc.calculate(6.1, 0.0).unwrap();

    assert_eq!(caution.model.label(), "TIC");
    assert_eq!(caution.safety_status, ThermalIndexStatus::Caution);
    assert_eq!(unsafe_result.safety_status, ThermalIndexStatus::Unsafe);
}

#[test]
fn test_thermal_index_rejects_invalid_domains() {
    let invalid_frequency =
        ThermalIndexCalculator::new(ThermalIndexModel::SoftTissue, 0.0, 0.5, 0.25, 6.0);
    let error = invalid_frequency.calculate(0.5, 1.0).unwrap_err();
    assert!(error.to_string().contains("center_frequency_mhz"));

    let invalid_reference =
        ThermalIndexCalculator::new(ThermalIndexModel::SoftTissue, 5.0, 0.5, 0.0, 6.0);
    let error = invalid_reference.calculate(0.5, 1.0).unwrap_err();
    assert!(error.to_string().contains("reference_power_w"));

    let invalid_depth =
        ThermalIndexCalculator::new(ThermalIndexModel::SoftTissue, 5.0, 0.5, 0.25, 6.0);
    let error = invalid_depth.calculate(0.5, -1.0).unwrap_err();
    assert!(error.to_string().contains("depth_cm"));
}
