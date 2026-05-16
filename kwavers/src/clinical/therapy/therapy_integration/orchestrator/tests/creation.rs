use super::*;

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
