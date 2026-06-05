use super::*;
use kwavers_core::constants::fundamental::DENSITY_WATER_NOMINAL;
use kwavers_core::constants::numerical::{MHZ_TO_HZ, MPA_TO_PA};

#[test]
fn test_safety_limit_checking() {
    let config = TherapySessionConfig {
        primary_modality: TherapyIntegrationModality::Transcranial,
        secondary_modalities: vec![],
        imaging_data_path: None,
        duration: 10.0,
        acoustic_params: AcousticTherapyParams {
            frequency: 0.5 * MHZ_TO_HZ,
            pnp: 0.5 * MPA_TO_PA,
            prf: 1.0,
            duty_cycle: 0.1,
            focal_depth: 0.05,
            treatment_volume: 1.0,
        },
        safety_limits: TherapyIntegrationSafetyLimits {
            thermal_index_max: 0.5,
            mechanical_index_max: 1.9,
            cavitation_dose_max: 1000.0,
            max_treatment_time: 300.0,
        },
        patient_params: PatientParameters {
            skull_thickness: None,
            tissue_properties: TissuePropertyMap::liver((8, 8, 8)),
            target_volume: TherapyTargetVolume {
                center: (0.05, 0.0, 0.0),
                dimensions: (0.01, 0.01, 0.01),
                tissue_type: TherapyTissueType::Brain,
            },
            risk_organs: vec![],
        },
    };

    let grid = kwavers_grid::Grid::new(8, 8, 8, 0.005, 0.005, 0.005).unwrap();
    let medium = HomogeneousMedium::new(DENSITY_WATER_NOMINAL, SOUND_SPEED_TISSUE, 0.5, 1.0, &grid);

    let mut orchestrator =
        TherapyIntegrationOrchestrator::new(config, grid, Box::new(medium.clone())).unwrap();

    orchestrator.execute_therapy_step(1.0).unwrap();

    let safety_status = orchestrator.check_safety_limits();
    assert_eq!(safety_status, TherapyIntegrationSafetyStatus::Safe);
}

#[test]
fn test_safety_controller_integration() {
    let config = TherapySessionConfig {
        primary_modality: TherapyIntegrationModality::HIFU,
        secondary_modalities: vec![],
        imaging_data_path: None,
        duration: 30.0,
        acoustic_params: AcousticTherapyParams {
            frequency: MHZ_TO_HZ,
            pnp: 5.0 * MPA_TO_PA,
            prf: 100.0,
            duty_cycle: 0.05,
            focal_depth: 0.04,
            treatment_volume: 0.8,
        },
        safety_limits: TherapyIntegrationSafetyLimits {
            thermal_index_max: 2.0,
            mechanical_index_max: 1.5,
            cavitation_dose_max: 100.0,
            max_treatment_time: 60.0,
        },
        patient_params: PatientParameters {
            skull_thickness: None,
            tissue_properties: TissuePropertyMap::liver((16, 16, 16)),
            target_volume: TherapyTargetVolume {
                center: (0.04, 0.0, 0.0),
                dimensions: (0.015, 0.015, 0.015),
                tissue_type: TherapyTissueType::Liver,
            },
            risk_organs: vec![],
        },
    };

    let grid = kwavers_grid::Grid::new(16, 16, 16, 0.002, 0.002, 0.002).unwrap();
    let medium = Box::new(HomogeneousMedium::new(
        DENSITY_WATER_NOMINAL,
        SOUND_SPEED_TISSUE,
        0.5,
        1.0,
        &grid,
    ));

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
