use crate::core::constants::fundamental::{DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER_SIM};
use super::grid::GridShape;
use super::metrics::diagnostics_for_prediction;
use super::predict::{point_source_observation_cube, source_kappa_filtered_source_weights};
use super::*;
use crate::physics::acoustics::imaging::modalities::ultrasound::frequency_domain_fwi::MultiRowRingArray;
use ndarray::Array3;
use num_complex::Complex64;
use std::f64::consts::PI;

#[test]
fn point_source_prediction_matches_outgoing_green_formula() {
    let array = MultiRowRingArray::new(4, 1, 0.006, 0.0).expect("array");
    let frequency_hz = 1.0 / (2.0 * PI);

    let cube =
        point_source_observation_cube(&array, &[frequency_hz], 1.0, 0.001).expect("prediction");

    let self_distance: f64 = 0.5e-3;
    let expected_self =
        Complex64::new(self_distance.cos(), self_distance.sin()) / (4.0 * PI * self_distance);
    let remote_distance = 2.0_f64.sqrt() * 0.003;
    let expected_remote =
        Complex64::new(remote_distance.cos(), remote_distance.sin()) / (4.0 * PI * remote_distance);
    assert_eq!(cube.dim(), (1, 4, 4));
    assert!((cube[[0, 0, 0]] - expected_self).norm() <= 1.0e-12);
    assert!((cube[[0, 0, 1]] - expected_remote).norm() <= 1.0e-12);
}

#[test]
fn source_kappa_weights_match_two_cell_symbol() {
    let spacing_m = 1.0e-3;
    let sound_speed_m_s = SOUND_SPEED_WATER_SIM;
    let time_step_s = 1.0e-7;

    let weights = source_kappa_filtered_source_weights(
        GridShape::new((2, 1, 1)).expect("shape"),
        spacing_m,
        sound_speed_m_s,
        time_step_s,
        &[(0, 0, 0)],
    )
    .expect("weights");

    let q = (0.5 * sound_speed_m_s * time_step_s * PI / spacing_m).cos();
    assert!((weights[[0, 0, 0]].re - 0.5 * (1.0 + q)).abs() <= 1.0e-15);
    assert!((weights[[1, 0, 0]].re - 0.5 * (1.0 - q)).abs() <= 1.0e-15);
    assert!((weights.iter().map(|value| value.re).sum::<f64>() - 1.0).abs() <= 1.0e-15);
}

#[test]
fn direct_field_metrics_recover_exact_scaled_prediction() {
    let array = MultiRowRingArray::new(4, 1, 0.006, 0.0).expect("array");
    let frequencies_hz = [100.0];
    let predicted = Array3::from_shape_fn((1, 4, 4), |(_, transmit, receiver)| {
        Complex64::new(1.0 + transmit as f64, 0.5 + receiver as f64)
    });
    let scale = Complex64::new(0.25, -0.75);
    let observed = predicted.mapv(|value| scale * value);
    let config = BreastUstPstdDatasetConfig {
        spacing_m: 1.0e-3,
        time_step_s: 0.001,
        cycles_per_frequency: 4,
        frequency_bin_cycles: 2,
        source_amplitude_pa: 4.0,
        density_kg_m3: DENSITY_WATER_NOMINAL,
        cpml_thickness_cells: 0,
    };

    let diagnostics = diagnostics_for_prediction(
        &predicted,
        &observed,
        &frequencies_hz,
        config,
        &[40],
        &[20],
        &array,
    )
    .expect("diagnostics");

    assert!(diagnostics.normalized_l2_residual <= 1.0e-14);
    assert!(diagnostics.active_only_normalized_l2_residual <= 1.0e-14);
    assert!(diagnostics.passive_only_normalized_l2_residual <= 1.0e-14);
    assert!(diagnostics.active_self_channel_phase_error_rms_rad <= 1.0e-14);
    assert!(diagnostics.active_self_channel_log_amplitude_error_rms <= 1.0e-14);
    assert_eq!(diagnostics.active_pair_count, 4);
    assert!(diagnostics.passive_phase_error_rms_rad <= 1.0e-14);
    assert!(diagnostics.passive_log_amplitude_error_rms <= 1.0e-14);
    assert_eq!(diagnostics.passive_pair_count, 12);
}

#[test]
fn direct_field_metrics_separate_active_and_passive_receiver_classes() {
    let array = MultiRowRingArray::new(4, 2, 0.006, 0.001).expect("array");
    let frequencies_hz = [100.0];
    let predicted = Array3::from_elem((1, 4, 8), Complex64::new(1.0, 0.0));
    let mut observed = predicted.clone();
    for transmit in 0..4 {
        observed[[0, transmit, transmit]] = Complex64::new(0.0, 2.0);
    }
    let config = BreastUstPstdDatasetConfig {
        spacing_m: 1.0e-3,
        time_step_s: 0.001,
        cycles_per_frequency: 4,
        frequency_bin_cycles: 2,
        source_amplitude_pa: 4.0,
        density_kg_m3: DENSITY_WATER_NOMINAL,
        cpml_thickness_cells: 0,
    };

    let diagnostics = diagnostics_for_prediction(
        &predicted,
        &observed,
        &frequencies_hz,
        config,
        &[40],
        &[20],
        &array,
    )
    .expect("diagnostics");

    assert!((diagnostics.active_only_normalized_l2_residual - 0.5_f64.sqrt()).abs() <= 1.0e-14);
    assert!(diagnostics.passive_only_normalized_l2_residual <= 1.0e-14);
    assert_eq!(diagnostics.active_pair_count, 8);
    assert_eq!(diagnostics.passive_pair_count, 24);
    assert!(diagnostics.active_self_channel_phase_error_rms_rad > 0.0);
    assert!(diagnostics.active_self_channel_log_amplitude_error_rms > 0.0);
}

#[test]
fn homogeneous_diagnostic_rejects_nonuniform_reference_medium() {
    let array = MultiRowRingArray::new(4, 1, 0.006, 0.0).expect("array");
    let config = BreastUstPstdDatasetConfig {
        spacing_m: 1.0e-3,
        time_step_s: 1.0e-7,
        cycles_per_frequency: 1,
        frequency_bin_cycles: 1,
        source_amplitude_pa: 1.0e3,
        density_kg_m3: DENSITY_WATER_NOMINAL,
        cpml_thickness_cells: 0,
    };
    let mut model = Array3::from_elem((12, 12, 3), SOUND_SPEED_WATER_SIM);
    model[[6, 6, 1]] = 1510.0;

    let err = diagnose_breast_ust_homogeneous_direct_field(&model, &array, &[200_000.0], config)
        .expect_err("nonuniform model must reject");

    assert!(err.to_string().contains("constant sound speed"));
}
