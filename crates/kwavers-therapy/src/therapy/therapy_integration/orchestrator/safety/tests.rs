use super::*;
use kwavers_core::constants::medical::TI_LIMIT_SOFT_TISSUE;
use kwavers_core::constants::numerical::{MHZ_TO_HZ, MPA_TO_PA};
use ndarray::Array3;

#[test]
fn test_thermal_index_calculation() {
    let acoustic_field = AcousticField {
        pressure: Array3::from_elem((10, 10, 10), MPA_TO_PA), // 1 MPa uniform pressure
        velocity_x: Array3::zeros((10, 10, 10)),
        velocity_y: Array3::zeros((10, 10, 10)),
        velocity_z: Array3::zeros((10, 10, 10)),
    };

    let acoustic_params = AcousticTherapyParams {
        frequency: MHZ_TO_HZ,  // 1 MHz
        pnp: MPA_TO_PA,        // 1 MPa
        prf: 100.0,            // 100 Hz
        duty_cycle: 0.1,       // 10%
        focal_depth: 0.05,     // 5 cm
        treatment_volume: 1.0, // 1 cm³
    };

    let mut safety_metrics = SafetyMetrics {
        thermal_index: 0.0,
        mechanical_index: 0.0,
        cavitation_dose: 0.0,
        temperature_rise: Array3::zeros((10, 10, 10)),
    };

    update_safety_metrics(
        &mut safety_metrics,
        &acoustic_field,
        &acoustic_params,
        0.01,
        None,
    )
    .unwrap();

    // TI = P_rms * sqrt(f) / 1e6
    // P_rms = 1e6 Pa, sqrt(1e6) = 1000
    // TI = 1e6 * 1000 / 1e6 = 1000
    assert!(safety_metrics.thermal_index > 0.0);
    assert!(safety_metrics.thermal_index < 2000.0); // Reasonable upper bound
}

#[test]
fn test_mechanical_index_calculation() {
    let acoustic_field = AcousticField {
        pressure: Array3::from_elem((8, 8, 8), 0.5 * MPA_TO_PA), // 0.5 MPa
        velocity_x: Array3::zeros((8, 8, 8)),
        velocity_y: Array3::zeros((8, 8, 8)),
        velocity_z: Array3::zeros((8, 8, 8)),
    };

    let acoustic_params = AcousticTherapyParams {
        frequency: MHZ_TO_HZ, // 1 MHz
        pnp: 0.5 * MPA_TO_PA, // 0.5 MPa
        prf: 100.0,
        duty_cycle: 0.1,
        focal_depth: 0.03,
        treatment_volume: 0.5,
    };

    let mut safety_metrics = SafetyMetrics {
        thermal_index: 0.0,
        mechanical_index: 0.0,
        cavitation_dose: 0.0,
        temperature_rise: Array3::zeros((8, 8, 8)),
    };

    update_safety_metrics(
        &mut safety_metrics,
        &acoustic_field,
        &acoustic_params,
        0.01,
        None,
    )
    .unwrap();

    // MI (FDA 510(k) / IEC 62359): MI = pnp_Pa / (1e3 × sqrt(f_Hz))
    // MI = 0.5e6 / (1e3 × sqrt(1e6)) = 0.5e6 / (1e3 × 1e3) = 0.5e6 / 1e6 = 0.5
    assert!((safety_metrics.mechanical_index - 0.5).abs() < 0.001);
}

#[test]
fn test_cavitation_dose_accumulation() {
    let acoustic_field = AcousticField {
        pressure: Array3::from_elem((5, 5, 5), MPA_TO_PA),
        velocity_x: Array3::zeros((5, 5, 5)),
        velocity_y: Array3::zeros((5, 5, 5)),
        velocity_z: Array3::zeros((5, 5, 5)),
    };

    let acoustic_params = AcousticTherapyParams {
        frequency: MHZ_TO_HZ,
        pnp: MPA_TO_PA,
        prf: 100.0,
        duty_cycle: 0.1,
        focal_depth: 0.05,
        treatment_volume: 1.0,
    };

    let mut safety_metrics = SafetyMetrics {
        thermal_index: 0.0,
        mechanical_index: 0.0,
        cavitation_dose: 0.0,
        temperature_rise: Array3::zeros((5, 5, 5)),
    };

    let cavitation_activity = Array3::from_elem((5, 5, 5), 0.5); // 50% activity
    let dt = 0.01; // 10 ms

    // First update
    update_safety_metrics(
        &mut safety_metrics,
        &acoustic_field,
        &acoustic_params,
        dt,
        Some(&cavitation_activity),
    )
    .unwrap();

    let dose_after_first = safety_metrics.cavitation_dose;
    assert!(dose_after_first > 0.0);

    // Second update - dose should accumulate
    update_safety_metrics(
        &mut safety_metrics,
        &acoustic_field,
        &acoustic_params,
        dt,
        Some(&cavitation_activity),
    )
    .unwrap();

    assert!(safety_metrics.cavitation_dose > dose_after_first);
    assert!((safety_metrics.cavitation_dose - 2.0 * dose_after_first).abs() < 1e-10);
}

#[test]
fn test_safety_limit_checking_all_safe() {
    let safety_metrics = SafetyMetrics {
        thermal_index: 3.0,
        mechanical_index: 1.0,
        cavitation_dose: 500.0,
        temperature_rise: Array3::zeros((5, 5, 5)),
    };

    let safety_limits = TherapyIntegrationSafetyLimits {
        thermal_index_max: TI_LIMIT_SOFT_TISSUE,
        mechanical_index_max: 1.9,
        cavitation_dose_max: 1000.0,
        max_treatment_time: 300.0,
    };

    let status = check_safety_limits(&safety_metrics, &safety_limits, 60.0);
    assert_eq!(status, TherapyIntegrationSafetyStatus::Safe);
}

#[test]
fn test_safety_limit_thermal_exceeded() {
    let safety_metrics = SafetyMetrics {
        thermal_index: 7.0, // Exceeds limit
        mechanical_index: 1.0,
        cavitation_dose: 500.0,
        temperature_rise: Array3::zeros((5, 5, 5)),
    };

    let safety_limits = TherapyIntegrationSafetyLimits {
        thermal_index_max: TI_LIMIT_SOFT_TISSUE,
        mechanical_index_max: 1.9,
        cavitation_dose_max: 1000.0,
        max_treatment_time: 300.0,
    };

    let status = check_safety_limits(&safety_metrics, &safety_limits, 60.0);
    assert_eq!(status, TherapyIntegrationSafetyStatus::ThermalLimitExceeded);
}

#[test]
fn test_safety_limit_mechanical_exceeded() {
    let safety_metrics = SafetyMetrics {
        thermal_index: 3.0,
        mechanical_index: 2.0, // Exceeds limit
        cavitation_dose: 500.0,
        temperature_rise: Array3::zeros((5, 5, 5)),
    };

    let safety_limits = TherapyIntegrationSafetyLimits {
        thermal_index_max: TI_LIMIT_SOFT_TISSUE,
        mechanical_index_max: 1.9,
        cavitation_dose_max: 1000.0,
        max_treatment_time: 300.0,
    };

    let status = check_safety_limits(&safety_metrics, &safety_limits, 60.0);
    assert_eq!(
        status,
        TherapyIntegrationSafetyStatus::MechanicalLimitExceeded
    );
}

#[test]
fn test_safety_limit_time_exceeded() {
    let safety_metrics = SafetyMetrics {
        thermal_index: 3.0,
        mechanical_index: 1.0,
        cavitation_dose: 500.0,
        temperature_rise: Array3::zeros((5, 5, 5)),
    };

    let safety_limits = TherapyIntegrationSafetyLimits {
        thermal_index_max: TI_LIMIT_SOFT_TISSUE,
        mechanical_index_max: 1.9,
        cavitation_dose_max: 1000.0,
        max_treatment_time: 300.0,
    };

    let status = check_safety_limits(&safety_metrics, &safety_limits, 301.0); // Over time limit
    assert_eq!(status, TherapyIntegrationSafetyStatus::TimeLimitExceeded);
}
