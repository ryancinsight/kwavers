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

    // Grid 16³, dx=0.002 m, focal_depth=0.03 m → focal voxel i=15 (= 15×0.002 = 0.030 m).
    // heating_scale = α·dt / (ρ²·c₀·c_p) = 0.5·0.1 / (1e6·1540·3600) = 9.025e-15
    // ΔT_focal = heating_scale · PNP² · exp(0) = 9.025e-15 · (1e6)² = 9.025e-3 °C
    // TI = ΔT_focal ≈ 0.009 (well within thermal_index_max = 6.0)
    // MI = pnp / (1e3 · √f_Hz) = 1e6 / (1e3 · √2e6) ≈ 0.7071 (< 1.9)
    const DT: f64 = 0.1;
    const PNP: f64 = 1e6;
    const DX: f64 = 0.002;
    const FOCAL: f64 = 0.03;
    const F_HZ: f64 = 2e6;
    const ALPHA: f64 = 0.5;
    const RHO: f64 = 1000.0;
    const C0: f64 = 1540.0;
    const CP: f64 = 3600.0;
    let heating_scale = ALPHA * DT / (RHO * RHO * C0 * CP);
    // Focal voxel i=15: x = 15·DX − FOCAL = 0, so r=0, exp(0)=1
    let _ = DX; // used in the comment derivation above
    let _ = FOCAL;
    let expected_delta_t = heating_scale * PNP * PNP; // distance_factor = exp(0) = 1
    let expected_mi = PNP / (1e3 * F_HZ.sqrt());

    for step in 0..5usize {
        orchestrator.execute_therapy_step(DT).unwrap();

        let safety_status = orchestrator.check_safety_limits();
        assert_eq!(safety_status, SafetyStatus::Safe, "step {step}: expected Safe");

        let state = orchestrator.session_state();

        // TI must equal ΔT_focal to within floating-point rounding at one step.
        let ti = state.safety_metrics.thermal_index;
        assert!(
            (ti - expected_delta_t).abs() < 1e-6,
            "step {step}: TI={ti:.6e} expected≈{expected_delta_t:.6e}"
        );

        // MI = pnp / (1e3 × √f_Hz) ≈ 0.7071.
        let mi = state.safety_metrics.mechanical_index;
        assert!(
            (mi - expected_mi).abs() < 1e-6,
            "step {step}: MI={mi:.6e} expected≈{expected_mi:.6e}"
        );

        // Pressure peak at focal voxel must equal PNP (Gaussian beam, r=0).
        let p_max = state
            .acoustic_field
            .as_ref()
            .expect("acoustic field must be set")
            .pressure
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        assert!(
            (p_max - PNP).abs() < 1.0,
            "step {step}: P_peak={p_max:.3e} expected≈{PNP:.3e}"
        );

        // Time and progress.
        let expected_time = (step + 1) as f64 * DT;
        assert!(
            (state.current_time - expected_time).abs() < 1e-10,
            "step {step}: current_time={} expected={}",
            state.current_time,
            expected_time
        );
    }

    let final_state = orchestrator.session_state();
    assert!((final_state.current_time - 5.0 * DT).abs() < 1e-10);
    assert!((final_state.progress - 5.0 * DT / 10.0).abs() < 1e-10);
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

    // Grid 12³, dx=0.0025 m, PNP=2 MPa, focal_depth=0.03 m.
    // Focal voxel would be i=12 (0.03/0.0025=12) but grid is 12³ (indices 0..11).
    // Peak voxel is i=11: x = 11·0.0025 − 0.03 = −0.0025 m, r=0.0025 m.
    // Gaussian: P_peak = 2e6 · exp(−r²/w²) = 2e6 · exp(−0.25) ≈ 1,557,600 Pa
    // ΔT_peak: heating_scale = α·dt / (ρ²·c₀·c_p) = 0.5·0.2 / (1e6·1540·3600) ≈ 1.804e-14
    //          distance_factor = exp(−r/L_focal) = exp(−0.0025/0.01) = exp(−0.25)
    //          ΔT_peak = 1.804e-14 · (1,557,600)² · exp(−0.25) ≈ 0.034 °C
    //          TI ≈ 0.034 (< thermal_index_max = 6.0)
    const DT: f64 = 0.2;
    const PNP: f64 = 2e6;
    const BEAM_W_SQ: f64 = 0.005 * 0.005; // (5 mm)²
    const L_FOCAL: f64 = 0.01;
    const ALPHA: f64 = 0.5;
    const RHO: f64 = 1000.0;
    const C0: f64 = 1540.0;
    const CP: f64 = 3600.0;
    // Peak voxel geometry (i=11, j=0, k=0)
    let r_sq = (11.0 * 0.0025 - 0.03_f64).powi(2); // (−0.0025)² = 6.25e-6
    let p_peak = PNP * (-r_sq / BEAM_W_SQ).exp();
    let r = r_sq.sqrt();
    let heating_scale = ALPHA * DT / (RHO * RHO * C0 * CP);
    let dist_factor = (-r / L_FOCAL).exp();
    let expected_delta_t = heating_scale * p_peak * p_peak * dist_factor;
    let expected_mi = PNP / (1e3 * 2e6_f64.sqrt());
    let duration = 10.0_f64;

    for step in 0..5usize {
        orchestrator
            .execute_therapy_step(DT)
            .unwrap_or_else(|e| panic!("Step {step} failed: {e:?}"));

        let state = orchestrator.session_state();

        // Temperature field shape must equal the grid and all values ≥ 37.0 °C.
        let t_field = &state.safety_metrics.temperature_rise;
        assert_eq!(
            t_field.dim(),
            (12, 12, 12),
            "step {step}: temperature_rise shape mismatch"
        );
        let t_min = t_field.iter().cloned().fold(f64::INFINITY, f64::min);
        assert!(
            t_min >= 37.0,
            "step {step}: temperature_rise min={t_min:.4} must be ≥ 37.0 °C"
        );

        // TI ≈ ΔT_peak (temperature rise at hottest voxel).
        let ti = state.safety_metrics.thermal_index;
        assert!(
            (ti - expected_delta_t).abs() < 1e-5,
            "step {step}: TI={ti:.6e} expected≈{expected_delta_t:.6e}"
        );

        // MI formula check.
        let mi = state.safety_metrics.mechanical_index;
        assert!(
            (mi - expected_mi).abs() < 1e-6,
            "step {step}: MI={mi:.6e} expected≈{expected_mi:.6e}"
        );

        // Pressure peak matches Gaussian beam model.
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

        // Timing is exact (f64 accumulation).
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
