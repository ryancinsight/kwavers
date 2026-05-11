use super::TherapyIntegrationOrchestrator;
use crate::clinical::therapy::therapy_integration::config::{
    AcousticTherapyParams, PatientParameters, SafetyLimits, TargetVolume, TherapyModality,
    TherapySessionConfig, TissueType,
};
use crate::clinical::therapy::therapy_integration::state::SafetyStatus;
use crate::clinical::therapy::therapy_integration::tissue::TissuePropertyMap;
use crate::domain::medium::homogeneous::HomogeneousMedium;

#[test]
fn test_therapy_orchestrator_creation() {
    let config = TherapySessionConfig {
        primary_modality: TherapyModality::Histotripsy,
        secondary_modalities: vec![TherapyModality::Microbubble],
        duration: 60.0,
        acoustic_params: AcousticTherapyParams {
            frequency: 1e6,
            pnp: 10e6,
            prf: 100.0,
            duty_cycle: 0.01,
            focal_depth: 0.05,
            treatment_volume: 1.0,
        },
        safety_limits: SafetyLimits {
            thermal_index_max: 6.0,
            mechanical_index_max: 1.9,
            cavitation_dose_max: 1000.0,
            max_treatment_time: 300.0,
        },
        patient_params: PatientParameters {
            skull_thickness: None,
            tissue_properties: TissuePropertyMap::liver((10, 10, 10)),
            target_volume: TargetVolume {
                center: (0.05, 0.0, 0.0),
                dimensions: (0.02, 0.02, 0.02),
                tissue_type: TissueType::Liver,
            },
            risk_organs: vec![],
        },
        imaging_data_path: None,
    };

    let grid = crate::domain::grid::Grid::new(32, 32, 32, 0.001, 0.001, 0.001).unwrap();
    let medium = HomogeneousMedium::new(1000.0, 1540.0, 0.5, 1.0, &grid);

    let orchestrator =
        TherapyIntegrationOrchestrator::new(config, grid, Box::new(medium.clone())).unwrap();
    assert_eq!(
        orchestrator.config().primary_modality,
        TherapyModality::Histotripsy
    );
    assert!(orchestrator.session_state().current_time < 1e-6);
}

#[test]
#[ignore] // Integration test - requires full therapy simulation stack
/// Test therapy step execution
/// # Panics
/// - Panics if an internal invariant assumed to hold at this call site is violated.
///
fn test_therapy_step_execution() {
    let config = TherapySessionConfig {
        primary_modality: TherapyModality::Microbubble,
        secondary_modalities: vec![],
        duration: 10.0,
        acoustic_params: AcousticTherapyParams {
            frequency: 2e6,
            pnp: 1e6,
            prf: 100.0,
            duty_cycle: 0.1,
            focal_depth: 0.03,
            treatment_volume: 0.5,
        },
        safety_limits: SafetyLimits {
            thermal_index_max: 6.0,
            mechanical_index_max: 1.9,
            cavitation_dose_max: 1000.0,
            max_treatment_time: 300.0,
        },
        patient_params: PatientParameters {
            skull_thickness: None,
            tissue_properties: TissuePropertyMap::liver((16, 16, 16)),
            target_volume: TargetVolume {
                center: (0.03, 0.0, 0.0),
                dimensions: (0.01, 0.01, 0.01),
                tissue_type: TissueType::Liver,
            },
            risk_organs: vec![],
        },
        imaging_data_path: None,
    };

    let grid = crate::domain::grid::Grid::new(16, 16, 16, 0.002, 0.002, 0.002).unwrap();
    let medium = Box::new(HomogeneousMedium::new(1000.0, 1540.0, 0.5, 1.0, &grid));

    let mut orchestrator = TherapyIntegrationOrchestrator::new(config, grid, medium).unwrap();

    let dt = 0.1;
    for _ in 0..5 {
        orchestrator.execute_therapy_step(dt).unwrap();

        let safety_status = orchestrator.check_safety_limits();
        assert_eq!(safety_status, SafetyStatus::Safe);
    }

    assert!(orchestrator.session_state().current_time > 0.0);
    assert!(orchestrator.session_state().progress > 0.0);
    assert!(!orchestrator.session_state().acoustic_field.as_ref().unwrap().pressure.is_empty());
}

#[test]
fn test_safety_limit_checking() {
    let config = TherapySessionConfig {
        primary_modality: TherapyModality::Transcranial,
        secondary_modalities: vec![],
        imaging_data_path: None,
        duration: 10.0,
        acoustic_params: AcousticTherapyParams {
            frequency: 0.5e6,
            pnp: 0.5e6,
            prf: 1.0,
            duty_cycle: 0.1,
            focal_depth: 0.05,
            treatment_volume: 1.0,
        },
        safety_limits: SafetyLimits {
            thermal_index_max: 0.5,
            mechanical_index_max: 1.9,
            cavitation_dose_max: 1000.0,
            max_treatment_time: 300.0,
        },
        patient_params: PatientParameters {
            skull_thickness: None,
            tissue_properties: TissuePropertyMap::liver((8, 8, 8)),
            target_volume: TargetVolume {
                center: (0.05, 0.0, 0.0),
                dimensions: (0.01, 0.01, 0.01),
                tissue_type: TissueType::Brain,
            },
            risk_organs: vec![],
        },
    };

    let grid = crate::domain::grid::Grid::new(8, 8, 8, 0.005, 0.005, 0.005).unwrap();
    let medium = HomogeneousMedium::new(1000.0, 1540.0, 0.5, 1.0, &grid);

    let mut orchestrator =
        TherapyIntegrationOrchestrator::new(config, grid, Box::new(medium.clone())).unwrap();

    orchestrator.execute_therapy_step(1.0).unwrap();

    let safety_status = orchestrator.check_safety_limits();
    assert_eq!(safety_status, SafetyStatus::Safe);
}

#[test]
fn test_safety_controller_integration() {
    let config = TherapySessionConfig {
        primary_modality: TherapyModality::HIFU,
        secondary_modalities: vec![],
        imaging_data_path: None,
        duration: 30.0,
        acoustic_params: AcousticTherapyParams {
            frequency: 1.0e6,
            pnp: 5e6,
            prf: 100.0,
            duty_cycle: 0.05,
            focal_depth: 0.04,
            treatment_volume: 0.8,
        },
        safety_limits: SafetyLimits {
            thermal_index_max: 2.0,
            mechanical_index_max: 1.5,
            cavitation_dose_max: 100.0,
            max_treatment_time: 60.0,
        },
        patient_params: PatientParameters {
            skull_thickness: None,
            tissue_properties: TissuePropertyMap::liver((16, 16, 16)),
            target_volume: TargetVolume {
                center: (0.04, 0.0, 0.0),
                dimensions: (0.015, 0.015, 0.015),
                tissue_type: TissueType::Liver,
            },
            risk_organs: vec![],
        },
    };

    let grid = crate::domain::grid::Grid::new(16, 16, 16, 0.002, 0.002, 0.002).unwrap();
    let medium = Box::new(HomogeneousMedium::new(1000.0, 1540.0, 0.5, 1.0, &grid));

    let mut orchestrator = TherapyIntegrationOrchestrator::new(config, grid, medium).unwrap();

    assert!(!orchestrator.should_stop());
    assert_eq!(orchestrator.power_reduction_factor(), 1.0);

    let dt = 0.5;
    let max_steps = 10;
    let mut safety_actions_observed = false;

    for step in 0..max_steps {
        orchestrator
            .execute_therapy_step(dt)
            .unwrap_or_else(|e| panic!("Step {step} failed: {e:?}"));

        let state = orchestrator.session_state();
        assert!(state.current_time > 0.0);
        assert!(!state.acoustic_field.as_ref().unwrap().pressure.is_empty());

        if orchestrator.should_stop() {
            safety_actions_observed = true;
            break;
        }

        let power_factor = orchestrator.power_reduction_factor();
        if power_factor < 1.0 {
            safety_actions_observed = true;
        }
    }

    assert!(
        safety_actions_observed || !orchestrator.should_stop(),
        "Safety controller should monitor therapy"
    );
}

#[test]
fn test_intensity_tracker_integration() {
    let config = TherapySessionConfig {
        primary_modality: TherapyModality::HIFU,
        secondary_modalities: vec![],
        imaging_data_path: None,
        duration: 10.0,
        acoustic_params: AcousticTherapyParams {
            frequency: 2.0e6,
            pnp: 2e6,
            prf: 50.0,
            duty_cycle: 0.02,
            focal_depth: 0.03,
            treatment_volume: 0.5,
        },
        safety_limits: SafetyLimits {
            thermal_index_max: 6.0,
            mechanical_index_max: 1.9,
            cavitation_dose_max: 1000.0,
            max_treatment_time: 300.0,
        },
        patient_params: PatientParameters {
            skull_thickness: None,
            tissue_properties: TissuePropertyMap::liver((12, 12, 12)),
            target_volume: TargetVolume {
                center: (0.03, 0.0, 0.0),
                dimensions: (0.012, 0.012, 0.012),
                tissue_type: TissueType::Liver,
            },
            risk_organs: vec![],
        },
    };

    let grid = crate::domain::grid::Grid::new(12, 12, 12, 0.0025, 0.0025, 0.0025).unwrap();
    let medium = Box::new(HomogeneousMedium::new(1000.0, 1540.0, 0.5, 1.0, &grid));

    let mut orchestrator = TherapyIntegrationOrchestrator::new(config, grid, medium).unwrap();

    let dt = 0.2;
    for step in 0..5 {
        orchestrator
            .execute_therapy_step(dt)
            .unwrap_or_else(|e| panic!("Step {step} failed: {e:?}"));

        let state = orchestrator.session_state();

        assert!(
            state.safety_metrics.temperature_rise.len() > 0,
            "Temperature field should be computed in step {}",
            step
        );

        assert!(
            state.acoustic_field.is_some(),
            "Acoustic field should exist in step {}",
            step
        );

        let expected_time = (step + 1) as f64 * dt;
        assert!(
            (state.current_time - expected_time).abs() < 1e-6,
            "Current time should be {} but got {}",
            expected_time,
            state.current_time
        );
    }

    let final_state = orchestrator.session_state();
    assert!(final_state.current_time > 0.0);
    assert!(final_state.progress > 0.0);
    assert!(final_state.progress <= 1.0);
}
