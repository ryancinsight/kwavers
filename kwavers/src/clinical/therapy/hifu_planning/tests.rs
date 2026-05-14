use super::*;
use crate::clinical::safety::mechanical_index::TissueType;
use crate::clinical::therapy::parameters::TherapyParameters;
use crate::core::constants::fundamental::SOUND_SPEED_WATER_SIM;
use crate::core::error::KwaversError;
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
    let schedule = planner.plan_sonication_schedule(&target, &params).unwrap();
    let plan = planner.plan_treatment(target, &params).unwrap();
    assert!(!plan.feasibility.is_feasible || plan.feasibility.issues.is_empty());
    assert!(schedule.subspot_count() > 1);
    assert!(schedule.coverage_guaranteed);
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

#[test]
fn test_sonication_schedule_pitch_proves_target_coverage() {
    let focal_spot = FocalSpot {
        location_mm: (0.0, 0.0, 80.0),
        lateral_width_mm: 6.0,
        axial_width_mm: 12.0,
        peak_pressure_pa: 4.0e6,
        mechanical_index: 1.0,
        focal_volume_mm3: 100.0,
        volume_minus6db_mm3: 70.0,
    };
    let target = AblationTarget::new(
        "box".to_owned(),
        (10.0, 20.0, 30.0),
        (10.0, 10.0, 10.0),
        TissueType::SoftTissue,
    )
    .with_safety_margin(1.0);
    let params = TherapyParameters {
        treatment_duration: 75.0,
        duty_cycle: 1.0,
        frequency: 1.5e6,
        ..TherapyParameters::hifu()
    };

    let schedule = SonicationSchedule::plan(&target, &focal_spot, &params, params.frequency)
        .expect("valid schedule");

    let expected_lateral_pitch = 6.0 / 3.0_f64.sqrt();
    let expected_axial_pitch = 12.0 / 3.0_f64.sqrt();
    assert!((schedule.pitch_mm.0 - expected_lateral_pitch).abs() < 1e-12);
    assert!((schedule.pitch_mm.1 - expected_lateral_pitch).abs() < 1e-12);
    assert!((schedule.pitch_mm.2 - expected_axial_pitch).abs() < 1e-12);
    assert_eq!(schedule.expanded_target_dimensions_mm, (12.0, 12.0, 12.0));
    assert_eq!(schedule.subspot_count(), 75);
    assert!(schedule.coverage_guaranteed);
    assert!((schedule.per_spot_dwell_time_s - 1.0).abs() < 1e-12);
    assert_eq!(schedule.subspots[0].index, 0);
    assert_eq!(schedule.subspots[0].location_mm, (4.0, 14.0, 24.0));
    assert_eq!(schedule.subspots[74].location_mm, (16.0, 26.0, 36.0));
}

#[test]
fn test_hifu_plan_uses_subspot_dose_for_feasibility() {
    let transducer = HIFUTransducer {
        frequency: 1.5e6,
        focal_length_mm: 80.0,
        aperture_diameter_mm: 40.0,
        power: 50.0,
        efficiency: 0.8,
        transducer_type: "focused".to_string(),
        transducer_diameter_mm: 40.0,
    };
    let planner = HIFUPlanner::new(transducer);
    let target = AblationTarget::new(
        "large_target".to_owned(),
        (0.0, 0.0, 80.0),
        (30.0, 30.0, 30.0),
        TissueType::SoftTissue,
    )
    .with_safety_margin(2.0);
    let params = TherapyParameters {
        treatment_duration: 5.0,
        duty_cycle: 0.5,
        frequency: 1.5e6,
        ..TherapyParameters::hifu()
    };

    let schedule = planner.plan_sonication_schedule(&target, &params).unwrap();
    let plan = planner.plan_treatment(target, &params).unwrap();

    assert!(schedule.subspot_count() > 1);
    assert!(plan.thermal_dose.cem43 > schedule.minimum_subspot_cem43);
    assert_eq!(
        plan.feasibility.thermal_dose_achievable,
        schedule.minimum_subspot_cem43 >= 240.0
    );
    if !plan.feasibility.thermal_dose_achievable {
        assert!(plan
            .feasibility
            .issues
            .iter()
            .any(|issue| issue.contains("Minimum subspot thermal dose")));
    }
}

#[test]
fn test_sonication_schedule_rejects_nonpositive_target_dimension() {
    let focal_spot = FocalSpot::estimate_from_transducer(&HIFUTransducer::default());
    let target = AblationTarget::new(
        "invalid".to_owned(),
        (0.0, 0.0, 80.0),
        (0.0, 10.0, 10.0),
        TissueType::SoftTissue,
    );
    let params = TherapyParameters::hifu();

    let err = SonicationSchedule::plan(&target, &focal_spot, &params, params.frequency)
        .expect_err("zero target dimension must be rejected");

    let KwaversError::InvalidInput(message) = err else {
        panic!("expected InvalidInput for zero dimension");
    };
    assert!(message.contains("target.dimensions_mm.0"));
    assert!(message.contains("finite and positive"));
}
