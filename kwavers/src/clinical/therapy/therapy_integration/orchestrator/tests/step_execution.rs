use super::*;

#[test]
fn test_therapy_step_execution() {
    let config = TherapySessionConfig {
        primary_modality: TherapyIntegrationModality::Microbubble,
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
        safety_limits: TherapyIntegrationSafetyLimits {
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
                tissue_type: TherapyTissueType::Liver,
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
    const RHO: f64 = crate::core::constants::fundamental::DENSITY_WATER_NOMINAL;
    const C0: f64 = crate::core::constants::fundamental::SOUND_SPEED_TISSUE;
    const CP: f64 = 3600.0;
    let heating_scale = ALPHA * DT / (RHO * RHO * C0 * CP);
    let _ = DX;
    let _ = FOCAL;
    let expected_delta_t = heating_scale * PNP * PNP;
    let expected_mi = PNP / (1e3 * F_HZ.sqrt());

    for step in 0..5usize {
        orchestrator.execute_therapy_step(DT).unwrap();

        let safety_status = orchestrator.check_safety_limits();
        assert_eq!(
            safety_status,
            TherapyIntegrationSafetyStatus::Safe,
            "step {step}: expected Safe"
        );

        let state = orchestrator.session_state();

        let ti = state.safety_metrics.thermal_index;
        assert!(
            (ti - expected_delta_t).abs() < 1e-6,
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
            (p_max - PNP).abs() < 1.0,
            "step {step}: P_peak={p_max:.3e} expected≈{PNP:.3e}"
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
    assert!((final_state.current_time - 5.0 * DT).abs() < 1e-10);
    assert!((final_state.progress - 5.0 * DT / 10.0).abs() < 1e-10);
}
