use super::*;
use crate::core::constants::medical::TI_LIMIT_SOFT_TISSUE;
use crate::core::constants::numerical::{MHZ_TO_HZ, MPA_TO_PA};

#[test]
fn test_therapy_orchestrator_creation() {
    let config = TherapySessionConfig {
        primary_modality: TherapyIntegrationModality::Histotripsy,
        secondary_modalities: vec![TherapyIntegrationModality::Microbubble],
        duration: 60.0,
        acoustic_params: AcousticTherapyParams {
            frequency: MHZ_TO_HZ,
            pnp: 10.0 * MPA_TO_PA,
            prf: 100.0,
            duty_cycle: 0.01,
            focal_depth: 0.05,
            treatment_volume: 1.0,
        },
        safety_limits: TherapyIntegrationSafetyLimits {
            thermal_index_max: TI_LIMIT_SOFT_TISSUE,
            mechanical_index_max: 1.9,
            cavitation_dose_max: 1000.0,
            max_treatment_time: 300.0,
        },
        patient_params: PatientParameters {
            skull_thickness: None,
            tissue_properties: TissuePropertyMap::liver((10, 10, 10)),
            target_volume: TherapyTargetVolume {
                center: (0.05, 0.0, 0.0),
                dimensions: (0.02, 0.02, 0.02),
                tissue_type: TherapyTissueType::Liver,
            },
            risk_organs: vec![],
        },
        imaging_data_path: None,
    };

    let grid = crate::domain::grid::Grid::new(32, 32, 32, 0.001, 0.001, 0.001).unwrap();
    let medium = HomogeneousMedium::new(1000.0, SOUND_SPEED_TISSUE, 0.5, 1.0, &grid);

    let orchestrator =
        TherapyIntegrationOrchestrator::new(config, grid, Box::new(medium.clone())).unwrap();
    assert_eq!(
        orchestrator.config().primary_modality,
        TherapyIntegrationModality::Histotripsy
    );
    assert!(orchestrator.session_state().current_time < 1e-6);
}
