use super::*;
use crate::core::constants::fundamental::DENSITY_WATER_NOMINAL;
use crate::core::constants::medical::TI_LIMIT_SOFT_TISSUE;
use crate::core::constants::numerical::{MHZ_TO_HZ, MPA_TO_PA};
use crate::core::constants::thermodynamic::BODY_TEMPERATURE_C;
use crate::core::constants::tissue_thermal::SPECIFIC_HEAT_TISSUE;

#[test]
fn test_intensity_tracker_integration() {
    let config = TherapySessionConfig {
        primary_modality: TherapyIntegrationModality::HIFU,
        secondary_modalities: vec![],
        imaging_data_path: None,
        duration: 10.0,
        acoustic_params: AcousticTherapyParams {
            frequency: 2.0 * MHZ_TO_HZ,
            pnp: 2.0 * MPA_TO_PA,
            prf: 50.0,
            duty_cycle: 0.02,
            focal_depth: 0.03,
            treatment_volume: 0.5,
        },
        safety_limits: TherapyIntegrationSafetyLimits {
            thermal_index_max: TI_LIMIT_SOFT_TISSUE,
            mechanical_index_max: 1.9,
            cavitation_dose_max: 1000.0,
            max_treatment_time: 300.0,
        },
        patient_params: PatientParameters {
            skull_thickness: None,
            tissue_properties: TissuePropertyMap::liver((12, 12, 12)),
            target_volume: TherapyTargetVolume {
                center: (0.03, 0.0, 0.0),
                dimensions: (0.012, 0.012, 0.012),
                tissue_type: TherapyTissueType::Liver,
            },
            risk_organs: vec![],
        },
    };

    let grid = crate::domain::grid::Grid::new(12, 12, 12, 0.0025, 0.0025, 0.0025).unwrap();
    let medium = Box::new(HomogeneousMedium::new(
        DENSITY_WATER_NOMINAL,
        SOUND_SPEED_TISSUE,
        0.5,
        1.0,
        &grid,
    ));

    let mut orchestrator = TherapyIntegrationOrchestrator::new(config, grid, medium).unwrap();

    // Grid 12³, dx=0.0025 m, PNP=2 MPa, focal_depth=0.03 m.
    // Peak voxel i=11: x = 11·0.0025 − 0.03 = −0.0025 m.
    // Gaussian: P_peak = 2e6 · exp(−r²/w²) = 2e6 · exp(−0.25)
    const DT: f64 = 0.2;
    const PNP: f64 = 2.0 * MPA_TO_PA;
    const BEAM_W_SQ: f64 = 0.005 * 0.005; // (5 mm)²
    const L_FOCAL: f64 = 0.01;
    const ALPHA: f64 = 0.5;
    const RHO: f64 = crate::core::constants::fundamental::DENSITY_WATER_NOMINAL;
    const C0: f64 = crate::core::constants::fundamental::SOUND_SPEED_TISSUE;
    let cp = SPECIFIC_HEAT_TISSUE;
    let r_sq = (11.0 * 0.0025 - 0.03_f64).powi(2);
    let p_peak = PNP * (-r_sq / BEAM_W_SQ).exp();
    let r = r_sq.sqrt();
    let heating_scale = ALPHA * DT / (RHO * RHO * C0 * cp);
    let dist_factor = (-r / L_FOCAL).exp();
    let expected_delta_t = heating_scale * p_peak * p_peak * dist_factor;
    let expected_mi = PNP / (1e3 * 2e6_f64.sqrt());
    let duration = 10.0_f64;

    for step in 0..5usize {
        orchestrator
            .execute_therapy_step(DT)
            .unwrap_or_else(|e| panic!("Step {step} failed: {e:?}"));

        let state = orchestrator.session_state();

        let t_field = &state.safety_metrics.temperature_rise;
        assert_eq!(
            t_field.dim(),
            (12, 12, 12),
            "step {step}: temperature_rise shape mismatch"
        );
        let t_min = t_field.iter().cloned().fold(f64::INFINITY, f64::min);
        assert!(
            t_min >= BODY_TEMPERATURE_C,
            "step {step}: temperature_rise min={t_min:.4} must be ≥ 37.0 °C"
        );

        let ti = state.safety_metrics.thermal_index;
        assert!(
            (ti - expected_delta_t).abs() < 1e-5,
            "step {step}: TI={ti:.6e} expected≈{expected_delta_t:.6e}"
        );

        let mi = state.safety_metrics.mechanical_index;
        assert!(
            (mi - expected_mi).abs() < 1e-6,
            "step {step}: MI={mi:.6e} expected≈{expected_mi:.6e}"
        );

        let p_max = state
            .acoustic_field
            .as_ref()
            .expect("acoustic field must be set")
            .pressure
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        assert!(
            (p_max - p_peak).abs() < 1.0,
            "step {step}: P_peak={p_max:.3e} expected≈{p_peak:.3e}"
        );

        let expected_time = (step + 1) as f64 * DT;
        assert!(
            (state.current_time - expected_time).abs() < 1e-10,
            "step {step}: current_time={} expected={}",
            state.current_time,
            expected_time
        );
    }

    let final_state = orchestrator.session_state();
    let expected_final_time = 5.0 * DT;
    assert!(
        (final_state.current_time - expected_final_time).abs() < 1e-10,
        "current_time={} expected={expected_final_time}",
        final_state.current_time
    );
    assert!(
        (final_state.progress - expected_final_time / duration).abs() < 1e-10,
        "progress={} expected={}",
        final_state.progress,
        expected_final_time / duration
    );
}
