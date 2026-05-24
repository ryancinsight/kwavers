use super::grid::GridShape;
use super::metrics::diagnostics_for_prediction;
use super::predict::{
    point_source_observation_cube, pstd_periodic_observation_cube,
    source_kappa_filtered_source_weights,
};
use super::*;
use crate::core::constants::fundamental::{DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER_SIM};
use crate::clinical::imaging::reconstruction::breast_ust_fwi::{
    generate_breast_ust_pstd_frequency_dataset, snap_multi_row_ring_array_to_grid,
};
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

#[test]
fn homogeneous_diagnostic_reports_passive_residual_deltas() {
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
    let model = Array3::from_elem((12, 12, 3), SOUND_SPEED_WATER_SIM);

    let diagnostics =
        diagnose_breast_ust_homogeneous_direct_field(&model, &array, &[200_000.0], config)
            .expect("homogeneous diagnostics");

    let source_kappa_passive_delta = diagnostics
        .source_kappa_filtered
        .passive_only_normalized_l2_residual
        - diagnostics.point_source.passive_only_normalized_l2_residual;
    let pstd_passive_delta = diagnostics
        .pstd_periodic
        .passive_only_normalized_l2_residual
        - diagnostics.point_source.passive_only_normalized_l2_residual;
    assert!(diagnostics
        .source_kappa_filtered_passive_residual_delta
        .is_finite());
    assert!(diagnostics.pstd_periodic_passive_residual_delta.is_finite());
    assert!(
        (diagnostics.source_kappa_filtered_passive_residual_delta - source_kappa_passive_delta)
            .abs()
            <= 1.0e-15
    );
    assert!(
        (diagnostics.pstd_periodic_passive_residual_delta - pstd_passive_delta).abs() <= 1.0e-15
    );
}

#[test]
fn finite_grid_pstd_prediction_matches_homogeneous_dataset() {
    let model = Array3::from_elem((4, 4, 3), SOUND_SPEED_WATER_SIM);
    let config = BreastUstPstdDatasetConfig {
        spacing_m: 3.2e-3,
        time_step_s: 1.0e-7,
        cycles_per_frequency: 4,
        frequency_bin_cycles: 1,
        source_amplitude_pa: 1.0e3,
        density_kg_m3: DENSITY_WATER_NOMINAL,
        cpml_thickness_cells: 0,
    };
    let unsnapped = MultiRowRingArray::new(4, 1, 0.00768, 0.0).expect("array");
    let array = snap_multi_row_ring_array_to_grid(&unsnapped, model.dim(), config.spacing_m)
        .expect("snapped array");
    let frequencies_hz = [200_000.0, 300_000.0];

    let dataset =
        generate_breast_ust_pstd_frequency_dataset(&model, &array, &frequencies_hz, config)
            .expect("homogeneous PSTD dataset");
    let predicted = pstd_periodic_observation_cube(
        &array,
        &frequencies_hz,
        SOUND_SPEED_WATER_SIM,
        config.spacing_m,
        model.dim(),
        config.time_step_s,
        &dataset.time_steps_per_frequency,
        &dataset.frequency_bin_start_steps_per_frequency,
        config.source_amplitude_pa,
    )
    .expect("finite-grid PSTD prediction");

    let mut max_abs_error = 0.0_f64;
    let mut max_error_index = (0, 0, 0);
    let mut max_error_observed = Complex64::new(0.0, 0.0);
    let mut max_error_expected = Complex64::new(0.0, 0.0);
    let mut max_reference = 0.0_f64;
    let mut scale_numerator = Complex64::new(0.0, 0.0);
    let mut scale_denominator = 0.0_f64;
    for ((frequency, transmit, receiver), &observed) in dataset.observed_pressure.indexed_iter() {
        let expected = predicted[[frequency, transmit, receiver]];
        let abs_error = (observed - expected).norm();
        if abs_error > max_abs_error {
            max_abs_error = abs_error;
            max_error_index = (frequency, transmit, receiver);
            max_error_observed = observed;
            max_error_expected = expected;
        }
        scale_numerator += observed * expected.conj();
        scale_denominator += expected.norm_sqr();
        max_reference = max_reference.max(observed.norm().max(expected.norm()));
    }
    let global_scale = if scale_denominator > 0.0 {
        scale_numerator / scale_denominator
    } else {
        Complex64::new(0.0, 0.0)
    };
    let scaled_error = dataset
        .observed_pressure
        .iter()
        .zip(predicted.iter())
        .map(|(&observed, &expected)| (observed - global_scale * expected).norm_sqr())
        .sum::<f64>()
        .sqrt();
    let observed_norm = dataset
        .observed_pressure
        .iter()
        .map(|value| value.norm_sqr())
        .sum::<f64>()
        .sqrt();
    let scaled_normalized_error = scaled_error / observed_norm.max(f64::MIN_POSITIVE);

    let tolerance = 1.0e-10 * max_reference.max(1.0);
    assert!(
        max_abs_error <= tolerance,
        "homogeneous PSTD modal predictor must match generated dataset: \
         max_abs_error={max_abs_error:e}, tolerance={tolerance:e}, \
         max_reference={max_reference:e}, max_error_index={max_error_index:?}, \
         observed_at_max={max_error_observed:?}, expected_at_max={max_error_expected:?}, \
         global_scale={global_scale:?}, scaled_normalized_error={scaled_normalized_error:e}"
    );
}
