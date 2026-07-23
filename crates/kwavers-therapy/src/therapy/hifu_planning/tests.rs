use super::*;
use crate::safety::mechanical_index::MechanicalIndexTissueType;
use crate::therapy::domain_types::ClinicalTherapyParameters;
use kwavers_core::constants::fundamental::{DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER_SIM};
use kwavers_core::constants::numerical::{MHZ_TO_HZ, MPA_TO_PA};
use kwavers_core::constants::thermodynamic::BODY_TEMPERATURE_C;
use kwavers_core::constants::thermodynamic::KELVIN_OFFSET_C;
use kwavers_core::error::KwaversError;
use std::f64::consts::PI;

#[test]
fn test_hifu_transducer_default() {
    let transducer = ClinicalHIFUTransducer::default();
    assert_eq!(transducer.frequency, 1.5 * MHZ_TO_HZ);
    assert_eq!(transducer.focal_length_mm, 80.0);
}

#[test]
fn test_focal_spot_estimation() {
    let transducer = ClinicalHIFUTransducer::default();
    let focal_spot = FocalSpot::estimate_from_transducer(&transducer).expect("valid transducer");
    assert!(focal_spot.lateral_width_mm > 0.0);
    assert!(focal_spot.axial_width_mm > 0.0);
    assert!(focal_spot.peak_pressure_pa > 0.0);
    assert!(focal_spot.focal_volume_mm3 > 0.0);
}

#[test]
fn test_focal_spot_safety() {
    let transducer = ClinicalHIFUTransducer::default();
    let focal_spot = FocalSpot::estimate_from_transducer(&transducer).expect("valid transducer");
    let is_safe = focal_spot.is_safe(MechanicalIndexTissueType::SoftTissue);
    assert!(is_safe || focal_spot.mechanical_index > 1.9);
}

#[test]
fn test_ablation_target_creation() {
    let target = AblationTarget::new(
        "tumor".to_string(),
        (50.0, 50.0, 50.0),
        (20.0, 20.0, 20.0),
        MechanicalIndexTissueType::SoftTissue,
    );
    assert_eq!(target.name, "tumor");
    assert!(target.volume_mm3() > 0.0);
}

#[test]
fn test_thermal_dose_calculation() {
    let transducer = ClinicalHIFUTransducer::default();
    let focal_spot = FocalSpot::estimate_from_transducer(&transducer).expect("valid transducer");
    let thermal_dose = FocalSpotDoseEstimate::estimate_from_focal_spot(
        &focal_spot,
        transducer.frequency,
        1.0,
        10.0,
    )
    .expect("valid focal dose");
    assert!(thermal_dose.peak_temperature.into_base() - KELVIN_OFFSET_C > BODY_TEMPERATURE_C);
    assert!(thermal_dose.time_to_dose.is_some());
}

#[test]
fn test_hifu_planner_creation() {
    let transducer = ClinicalHIFUTransducer::default();
    let planner = HIFUPlanner::new(transducer);
    assert_eq!(planner.transducer().frequency, 1.5 * MHZ_TO_HZ);
}

#[test]
fn test_treatment_plan_creation() {
    let transducer = ClinicalHIFUTransducer::default();
    let planner = HIFUPlanner::new(transducer);
    let target = AblationTarget::new(
        "tumor".to_string(),
        (50.0, 50.0, 130.0),
        (20.0, 20.0, 20.0),
        MechanicalIndexTissueType::SoftTissue,
    );
    let params = ClinicalTherapyParameters::hifu();
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
    let transducer = ClinicalHIFUTransducer {
        frequency: 1.5 * MHZ_TO_HZ,
        focal_length_mm: 80.0,
        aperture_diameter_mm: 40.0,
        power: 50.0,
        efficiency: 0.8,
        transducer_type: "focused".to_string(),
        transducer_diameter_mm: 40.0,
    };
    let focal_spot = FocalSpot::estimate_from_transducer(&transducer).expect("valid transducer");
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
fn focal_spot_pressure_uses_acoustic_power_without_empirical_ceiling() {
    let transducer = ClinicalHIFUTransducer {
        frequency: MHZ_TO_HZ,
        focal_length_mm: 80.0,
        aperture_diameter_mm: 40.0,
        power: 10_000.0,
        efficiency: 0.5,
        transducer_type: "focused".to_string(),
        transducer_diameter_mm: 40.0,
    };

    let focal_spot = FocalSpot::estimate_from_transducer(&transducer).expect("valid transducer");
    let wavelength_mm = SOUND_SPEED_WATER_SIM / transducer.frequency * 1.0e3;
    let f_number = transducer.focal_length_mm / transducer.aperture_diameter_mm;
    let lateral_radius_mm = 0.5 * 1.02 * wavelength_mm * f_number;
    let focal_area_m2 = PI * lateral_radius_mm.powi(2) * 1.0e-6;
    let intensity = transducer.power * transducer.efficiency / focal_area_m2;
    let expected_pressure =
        (2.0 * DENSITY_WATER_NOMINAL * SOUND_SPEED_WATER_SIM * intensity).sqrt();

    // Magnitude sanity: a 10 kW HIFU transducer at 50% efficiency focused
    // through a 1 mm beam at 1 MHz produces sqrt(2·ρ·c·I) ≈ 45 MPa peak
    // pressure (well into the cavitation regime). The lower bound of 40
    // MPa keeps the check meaningful without being numerically brittle to
    // the exact resolution coefficient (1.02 here per Rayleigh).
    assert!(expected_pressure > 40.0 * MPA_TO_PA);
    assert!((focal_spot.peak_pressure_pa - expected_pressure).abs() / expected_pressure < 1.0e-12);
}

#[test]
fn focal_spot_rejects_invalid_transducer_domain() {
    let transducer = ClinicalHIFUTransducer {
        frequency: 0.0,
        ..ClinicalHIFUTransducer::default()
    };
    let err = FocalSpot::estimate_from_transducer(&transducer)
        .expect_err("zero frequency must be rejected");
    assert!(err.to_string().contains("transducer.frequency"));

    let transducer = ClinicalHIFUTransducer {
        efficiency: 1.2,
        ..ClinicalHIFUTransducer::default()
    };
    let err = FocalSpot::estimate_from_transducer(&transducer)
        .expect_err("efficiency above unity must be rejected");
    assert!(err.to_string().contains("transducer.efficiency"));
}

#[test]
fn focal_dose_uses_cem43_equivalent_minutes() {
    let focal_spot = FocalSpot {
        location_mm: (0.0, 0.0, 80.0),
        lateral_width_mm: 6.0,
        axial_width_mm: 12.0,
        peak_pressure_pa: 0.0,
        mechanical_index: 0.0,
        focal_volume_mm3: 100.0,
        volume_minus6db_mm3: 70.0,
    };

    let dose = FocalSpotDoseEstimate::estimate_from_focal_spot(&focal_spot, MHZ_TO_HZ, 1.0, 60.0)
        .expect("zero pressure remains a valid no-heating dose");

    let expected = 0.25_f64.powf(43.0 - BODY_TEMPERATURE_C);
    assert!((dose.cem43.as_minutes() - expected).abs() < 1.0e-15);
    assert_eq!(
        dose.peak_temperature.into_base() - KELVIN_OFFSET_C,
        BODY_TEMPERATURE_C
    );
    assert!(dose.time_to_dose.is_none());
}

#[test]
fn focal_dose_rejects_invalid_treatment_domain() {
    let focal_spot = FocalSpot {
        location_mm: (0.0, 0.0, 80.0),
        lateral_width_mm: 6.0,
        axial_width_mm: 12.0,
        peak_pressure_pa: MPA_TO_PA,
        mechanical_index: 1.0,
        focal_volume_mm3: 100.0,
        volume_minus6db_mm3: 70.0,
    };

    let err = FocalSpotDoseEstimate::estimate_from_focal_spot(&focal_spot, 0.0, 1.0, 10.0)
        .expect_err("zero frequency must be rejected");
    assert!(err.to_string().contains("frequency_hz"));

    let err = FocalSpotDoseEstimate::estimate_from_focal_spot(&focal_spot, MHZ_TO_HZ, 1.1, 10.0)
        .expect_err("duty cycle above unity must be rejected");
    assert!(err.to_string().contains("duty_cycle"));
}

#[test]
fn test_sonication_schedule_pitch_proves_target_coverage() {
    let focal_spot = FocalSpot {
        location_mm: (0.0, 0.0, 80.0),
        lateral_width_mm: 6.0,
        axial_width_mm: 12.0,
        peak_pressure_pa: 4.0 * MPA_TO_PA,
        mechanical_index: 1.0,
        focal_volume_mm3: 100.0,
        volume_minus6db_mm3: 70.0,
    };
    let target = AblationTarget::new(
        "box".to_owned(),
        (10.0, 20.0, 30.0),
        (10.0, 10.0, 10.0),
        MechanicalIndexTissueType::SoftTissue,
    )
    .with_safety_margin(1.0);
    let params = ClinicalTherapyParameters {
        treatment_duration: 75.0,
        duty_cycle: 1.0,
        frequency: 1.5 * MHZ_TO_HZ,
        ..ClinicalTherapyParameters::hifu()
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
    assert!((schedule.per_spot_dwell.into_base() - 1.0).abs() < 1e-12);
    assert_eq!(schedule.subspots[0].index, 0);
    assert_eq!(schedule.subspots[0].location_mm, (4.0, 14.0, 24.0));
    assert_eq!(schedule.subspots[74].location_mm, (16.0, 26.0, 36.0));
}

#[test]
fn test_hifu_plan_uses_subspot_dose_for_feasibility() {
    let transducer = ClinicalHIFUTransducer {
        frequency: 1.5 * MHZ_TO_HZ,
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
        MechanicalIndexTissueType::SoftTissue,
    )
    .with_safety_margin(2.0);
    let params = ClinicalTherapyParameters {
        treatment_duration: 5.0,
        duty_cycle: 0.5,
        frequency: 1.5 * MHZ_TO_HZ,
        ..ClinicalTherapyParameters::hifu()
    };

    let schedule = planner.plan_sonication_schedule(&target, &params).unwrap();
    let plan = planner.plan_treatment(target, &params).unwrap();

    assert!(schedule.subspot_count() > 1);
    assert!(plan.thermal_dose.cem43 > schedule.minimum_subspot_cem43);
    assert_eq!(
        plan.feasibility.thermal_dose_achievable,
        schedule.all_subspots_reach_ablation()
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
    let focal_spot = FocalSpot::estimate_from_transducer(&ClinicalHIFUTransducer::default())
        .expect("valid transducer");
    let target = AblationTarget::new(
        "invalid".to_owned(),
        (0.0, 0.0, 80.0),
        (0.0, 10.0, 10.0),
        MechanicalIndexTissueType::SoftTissue,
    );
    let params = ClinicalTherapyParameters::hifu();

    let err = SonicationSchedule::plan(&target, &focal_spot, &params, params.frequency)
        .expect_err("zero target dimension must be rejected");

    let KwaversError::InvalidInput(message) = err else {
        panic!("expected InvalidInput for zero dimension");
    };
    assert!(message.contains("target.dimensions_mm.0"));
    assert!(message.contains("finite and positive"));
}
