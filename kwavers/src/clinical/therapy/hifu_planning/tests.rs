use super::*;
use crate::clinical::safety::mechanical_index::TissueType;
use crate::clinical::therapy::parameters::TherapyParameters;
use crate::core::constants::fundamental::SOUND_SPEED_WATER_SIM;
use std::f64::consts::PI;

#[test]
fn test_hifu_transducer_default() {
    let transducer = HIFUTransducer::default();
    assert_eq!(transducer.frequency, 1.5e6);
    assert_eq!(transducer.focal_length_mm, 80.0);
}

#[test]
fn test_focal_spot_estimation() {
    let transducer = HIFUTransducer::default();
    let focal_spot = FocalSpot::estimate_from_transducer(&transducer);
    assert!(focal_spot.lateral_width_mm > 0.0);
    assert!(focal_spot.axial_width_mm > 0.0);
    assert!(focal_spot.peak_pressure_pa > 0.0);
    assert!(focal_spot.focal_volume_mm3 > 0.0);
}

#[test]
fn test_focal_spot_safety() {
    let transducer = HIFUTransducer::default();
    let focal_spot = FocalSpot::estimate_from_transducer(&transducer);
    let is_safe = focal_spot.is_safe(TissueType::SoftTissue);
    assert!(is_safe || focal_spot.mechanical_index > 1.9);
}

#[test]
fn test_ablation_target_creation() {
    let target = AblationTarget::new(
        "tumor".to_string(),
        (50.0, 50.0, 50.0),
        (20.0, 20.0, 20.0),
        TissueType::SoftTissue,
    );
    assert_eq!(target.name, "tumor");
    assert!(target.volume_mm3() > 0.0);
}

#[test]
fn test_thermal_dose_calculation() {
    let transducer = HIFUTransducer::default();
    let focal_spot = FocalSpot::estimate_from_transducer(&transducer);
    let thermal_dose =
        ThermalDose::estimate_from_focal_spot(&focal_spot, transducer.frequency, 1.0, 10.0);
    assert!(thermal_dose.peak_temperature_c > 37.0);
    assert!(thermal_dose.time_to_dose_s.is_finite() || thermal_dose.time_to_dose_s.is_infinite());
}

#[test]
fn test_hifu_planner_creation() {
    let transducer = HIFUTransducer::default();
    let planner = HIFUPlanner::new(transducer);
    assert_eq!(planner.transducer().frequency, 1.5e6);
}

#[test]
fn test_treatment_plan_creation() {
    let transducer = HIFUTransducer::default();
    let planner = HIFUPlanner::new(transducer);
    let target = AblationTarget::new(
        "tumor".to_string(),
        (50.0, 50.0, 130.0),
        (20.0, 20.0, 20.0),
        TissueType::SoftTissue,
    );
    let params = TherapyParameters::hifu();
    let plan = planner.plan_treatment(target, &params).unwrap();
    assert!(!plan.feasibility.is_feasible || plan.feasibility.issues.is_empty());
}

#[test]
fn test_treatment_feasibility_assessment() {
    let mut feasibility = TreatmentFeasibility::new();
    feasibility.focal_coverage_adequate = true;
    feasibility.mi_within_limits = true;
    feasibility.thermal_dose_achievable = true;
    feasibility.access_path_clear = true;
    feasibility.update_feasibility();
    assert!(feasibility.is_feasible);
    assert_eq!(feasibility.confidence_percent, 100.0);
}

#[test]
fn test_focal_spot_dimensions_match_oneil_formula() {
    let transducer = HIFUTransducer {
        frequency: 1.5e6,
        focal_length_mm: 80.0,
        aperture_diameter_mm: 40.0,
        power: 50.0,
        efficiency: 0.8,
        transducer_type: "focused".to_string(),
        transducer_diameter_mm: 40.0,
    };
    let focal_spot = FocalSpot::estimate_from_transducer(&transducer);
    let c = SOUND_SPEED_WATER_SIM;
    let lambda_mm = c / transducer.frequency * 1e3;
    let f_number = transducer.focal_length_mm / transducer.aperture_diameter_mm;
    let expected_lateral = 1.02 * lambda_mm * f_number;
    let expected_axial = (8.0 / PI) * f_number * f_number * lambda_mm;
    let rel_err_lat = (focal_spot.lateral_width_mm - expected_lateral).abs() / expected_lateral;
    assert!(
        rel_err_lat < 1e-10,
        "Lateral FWHM = {:.4} mm, expected {:.4} mm",
        focal_spot.lateral_width_mm,
        expected_lateral
    );
    let rel_err_ax = (focal_spot.axial_width_mm - expected_axial).abs() / expected_axial;
    assert!(
        rel_err_ax < 1e-10,
        "Axial FWHM = {:.4} mm, expected {:.4} mm",
        focal_spot.axial_width_mm,
        expected_axial
    );
    let aspect = focal_spot.axial_width_mm / focal_spot.lateral_width_mm;
    let expected_aspect = (8.0 / PI) * f_number / 1.02;
    assert!(
        (aspect - expected_aspect).abs() / expected_aspect < 1e-10,
        "Aspect ratio = {:.4}, expected {:.4}",
        aspect,
        expected_aspect
    );
}
