use super::*;
use crate::core::constants::fundamental::SOUND_SPEED_WATER_SIM;
use ndarray::Array3;
use num_complex::Complex64;

#[test]
fn residual_metrics_recover_row_source_scale() {
    let predicted = Array3::from_shape_vec(
        (1, 2, 2),
        vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 2.0),
            Complex64::new(2.0, -1.0),
            Complex64::new(0.5, 0.25),
        ],
    )
    .expect("shape");
    let scale = Complex64::new(3.0, -2.0);
    let observed = predicted.mapv(|value| scale * value);

    let metrics =
        scaled_observation_residual_metrics(&predicted, &observed, None).expect("metrics");

    assert_eq!(metrics.row_count, 2);
    assert!(metrics.normalized_l2_residual <= 1.0e-14);
    assert!((metrics.source_scale_magnitude_min - scale.norm()).abs() <= 1.0e-14);
    assert!((metrics.source_scale_magnitude_max - scale.norm()).abs() <= 1.0e-14);
}

#[test]
fn source_channel_diagnostics_isolate_passive_mask() {
    let predicted = Array3::from_shape_vec(
        (1, 2, 4),
        vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, -1.0),
            Complex64::new(3.0, 0.5),
            Complex64::new(4.0, 1.5),
            Complex64::new(2.0, 0.5),
            Complex64::new(5.0, -0.5),
            Complex64::new(7.0, 0.25),
            Complex64::new(11.0, -1.0),
        ],
    )
    .expect("shape");
    let scale = Complex64::new(2.0, 0.5);
    let mut observed = predicted.mapv(|value| scale * value);
    let active = source_receiver_mask(predicted.dim(), 2, 2).expect("mask");
    for (value, &is_active) in observed.iter_mut().zip(active.iter()) {
        if is_active {
            *value += Complex64::new(10.0, -3.0);
        }
    }

    let diagnostics =
        source_channel_residual_diagnostics(&predicted, &observed, 2, 2).expect("diagnostics");

    assert!(diagnostics.all_channel_normalized_l2_residual > 0.1);
    assert!(diagnostics.passive_only_normalized_l2_residual <= 1.0e-14);
    assert_eq!(diagnostics.active_receiver_count_per_row, 2);
    assert_eq!(diagnostics.passive_receiver_count_per_row, 2);
}

#[test]
fn source_excitation_detects_transmit_dispersion() {
    let predicted = Array3::from_shape_vec(
        (1, 2, 2),
        vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 0.5),
            Complex64::new(3.0, -1.0),
            Complex64::new(4.0, 0.25),
        ],
    )
    .expect("shape");
    let coefficient = sine_frequency_bin_coefficient(100.0, 0.001, 40, 20).expect("coefficient");
    let mut observed = predicted.clone();
    for receiver in 0..2 {
        observed[[0, 0, receiver]] *= 4.0 * coefficient;
        observed[[0, 1, receiver]] *= 8.0 * Complex64::new(0.0, 1.0) * coefficient;
    }

    let diagnostics =
        source_excitation_diagnostics(&predicted, &observed, &[100.0], 4.0, 0.001, &[40], &[20])
            .expect("diagnostics");

    assert!(diagnostics.max_source_scale_magnitude_coefficient_of_variation > 0.3);
    assert!(diagnostics.max_source_scale_phase_span_rad > 1.0);
}

#[test]
fn reconstruction_metrics_and_table1_parity_match_contracts() {
    let reference = Array3::from_shape_vec((1, 1, 3), vec![1450.0, SOUND_SPEED_WATER_SIM, 1550.0])
        .expect("shape");
    let shifted = reference.mapv(|value| value + 5.0);

    let metrics = reconstruction_metrics(&reference, &shifted).expect("metrics");
    let parity =
        table1_parity(metrics.rmse_m_s, metrics.pearson_correlation, 1, 2.0, 0.95).expect("parity");

    assert!((metrics.rmse_m_s - 5.0).abs() <= 1.0e-12);
    assert!((metrics.pearson_correlation - 1.0).abs() <= 1.0e-12);
    assert_eq!(parity.table1_3d_rmse_m_s, 15.5);
    assert_eq!(parity.table1_3d_pearson_correlation, 0.8848);
}

#[test]
fn identifiability_reports_rank_upper_bound() {
    let report = acquisition_identifiability(
        (8, 8, 4),
        &[200_000.0],
        4,
        4,
        BreastUstSourceScalingPolicy::Estimated,
    )
    .expect("report");

    assert_eq!(report.unknown_voxels, 256);
    assert_eq!(report.complex_observations, 16);
    assert_eq!(report.real_observation_dof, 32);
    assert_eq!(report.estimated_source_scale_real_dof, 8);
    assert_eq!(report.informative_real_dof_upper_bound, 24);
    assert!(report.underdetermined_by_rank_upper_bound);
}

#[test]
fn forward_operator_equivalence_selects_lowest_residual_model() {
    let accurate = Array3::from_shape_vec(
        (1, 2, 2),
        vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 0.5),
            Complex64::new(3.0, -1.0),
            Complex64::new(4.0, 0.25),
        ],
    )
    .expect("shape");
    let mut distorted = accurate.clone();
    distorted[[0, 0, 1]] *= Complex64::new(0.25, 0.0);
    distorted[[0, 1, 1]] *= Complex64::new(-0.5, 1.0);
    let coefficient = sine_frequency_bin_coefficient(100.0, 0.001, 40, 20).expect("coefficient");
    let observed = accurate.mapv(|value| 4.0 * coefficient * value);
    let predictions = [
        BreastUstForwardOperatorPrediction {
            model: "distorted",
            pressure: &distorted,
        },
        BreastUstForwardOperatorPrediction {
            model: "accurate",
            pressure: &accurate,
        },
    ];

    let diagnostics = forward_operator_equivalence_diagnostics(
        &predictions,
        &observed,
        &[100.0],
        4.0,
        0.001,
        &[40],
        &[20],
    )
    .expect("diagnostics");

    assert_eq!(diagnostics.model_count, 2);
    assert_eq!(
        diagnostics.receiver_channel_policy,
        BreastUstReceiverChannelPolicy::All
    );
    assert_eq!(diagnostics.best_model, "accurate");
    assert!(diagnostics.best_normalized_l2_residual <= 1.0e-14);
    assert_eq!(diagnostics.worst_model, "distorted");
    assert!(diagnostics.residual_spread > 0.0);
}

#[test]
fn forward_operator_equivalence_respects_receiver_channel_policy() {
    let observed = Array3::from_elem((1, 2, 4), Complex64::new(1.0, 0.0));
    let mut active_distorted = observed.clone();
    let mut passive_distorted = observed.clone();
    for transmit in 0..2 {
        for receiver in 0..4 {
            let row_dependent_distortion = if receiver / 2 == 0 {
                Complex64::new(0.0, 2.0)
            } else {
                Complex64::new(3.0, 0.0)
            };
            if receiver % 2 == transmit {
                active_distorted[[0, transmit, receiver]] = row_dependent_distortion;
            } else {
                passive_distorted[[0, transmit, receiver]] = row_dependent_distortion;
            }
        }
    }
    let predictions = [
        BreastUstForwardOperatorPrediction {
            model: "active_distorted",
            pressure: &active_distorted,
        },
        BreastUstForwardOperatorPrediction {
            model: "passive_distorted",
            pressure: &passive_distorted,
        },
    ];

    let passive = forward_operator_equivalence_diagnostics_with_receiver_policy(
        &predictions,
        &observed,
        &[100.0],
        1.0,
        0.001,
        &[40],
        &[20],
        BreastUstReceiverChannelPolicy::PassiveOnly,
    )
    .expect("passive diagnostics");
    let active = forward_operator_equivalence_diagnostics_with_receiver_policy(
        &predictions,
        &observed,
        &[100.0],
        1.0,
        0.001,
        &[40],
        &[20],
        BreastUstReceiverChannelPolicy::ActiveOnly,
    )
    .expect("active diagnostics");

    assert_eq!(
        passive.receiver_channel_policy,
        BreastUstReceiverChannelPolicy::PassiveOnly
    );
    assert_eq!(passive.best_model, "active_distorted");
    assert!(passive.best_normalized_l2_residual <= 1.0e-14);
    assert_eq!(
        active.receiver_channel_policy,
        BreastUstReceiverChannelPolicy::ActiveOnly
    );
    assert_eq!(active.best_model, "passive_distorted");
    assert!(active.best_normalized_l2_residual <= 1.0e-14);
}

#[test]
fn scattering_increment_diagnostics_identify_exact_increment_model() {
    let baseline = Array3::from_shape_vec(
        (1, 2, 4),
        vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(2.0, 0.0),
        ],
    )
    .expect("shape");
    let increment = Array3::from_shape_vec(
        (1, 2, 4),
        vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(-1.0, 0.0),
            Complex64::new(0.0, 2.0),
            Complex64::new(0.0, -2.0),
            Complex64::new(0.5, 0.0),
            Complex64::new(-0.5, 0.0),
            Complex64::new(0.0, 1.0),
            Complex64::new(0.0, -1.0),
        ],
    )
    .expect("shape");
    let scale = Complex64::new(2.0, -0.5);
    let observed = baseline.mapv(|value| scale * value) + &increment;
    let exact = &baseline + &increment.mapv(|value| value / scale);
    let half = &baseline + &increment.mapv(|value| value / (2.0 * scale));
    let predictions = [
        BreastUstForwardOperatorPrediction {
            model: "baseline",
            pressure: &baseline,
        },
        BreastUstForwardOperatorPrediction {
            model: "half_increment",
            pressure: &half,
        },
        BreastUstForwardOperatorPrediction {
            model: "exact_increment",
            pressure: &exact,
        },
    ];

    let diagnostics = scattering_increment_diagnostics(
        &baseline,
        &predictions,
        &observed,
        BreastUstReceiverChannelPolicy::All,
    )
    .expect("diagnostics");

    assert_eq!(diagnostics.model_count, 3);
    assert_eq!(diagnostics.best_model, "exact_increment");
    assert!(diagnostics.best_normalized_increment_residual <= 1.0e-14);
    assert!(diagnostics.observed_increment_l2_norm > 0.0);
    let baseline_row = scattering_model(&diagnostics, "baseline");
    let half_row = scattering_model(&diagnostics, "half_increment");
    let exact_row = scattering_model(&diagnostics, "exact_increment");
    assert!((baseline_row.normalized_increment_residual - 1.0).abs() <= 1.0e-14);
    assert!((half_row.normalized_increment_residual - 0.5).abs() <= 1.0e-14);
    assert!(exact_row.baseline_scaled_full_field_normalized_residual <= 1.0e-14);
    assert!(exact_row.model_scaled_full_field_normalized_residual <= 1.0e-14);
    assert!(exact_row.source_scale_relative_drift_mean <= 1.0e-14);
    assert!(exact_row.source_scale_phase_drift_max_abs_rad <= 1.0e-14);
}

#[test]
fn scattering_increment_diagnostics_respect_passive_receiver_policy() {
    let baseline = Array3::from_elem((1, 2, 4), Complex64::new(1.0, 0.0));
    let passive_increment = Array3::from_shape_vec(
        (1, 2, 4),
        vec![
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(-1.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(-2.0, 0.0),
            Complex64::new(0.0, 0.0),
        ],
    )
    .expect("shape");
    let active_increment = Array3::from_shape_vec(
        (1, 2, 4),
        vec![
            Complex64::new(4.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(-4.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(3.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(-3.0, 0.0),
        ],
    )
    .expect("shape");
    let scale = Complex64::new(3.0, 0.0);
    let observed = baseline.mapv(|value| scale * value) + &passive_increment + &active_increment;
    let passive_exact = &baseline + &passive_increment.mapv(|value| value / scale);
    let active_only = &baseline + &active_increment.mapv(|value| value / scale);
    let predictions = [
        BreastUstForwardOperatorPrediction {
            model: "active_only",
            pressure: &active_only,
        },
        BreastUstForwardOperatorPrediction {
            model: "passive_exact",
            pressure: &passive_exact,
        },
    ];

    let diagnostics = scattering_increment_diagnostics(
        &baseline,
        &predictions,
        &observed,
        BreastUstReceiverChannelPolicy::PassiveOnly,
    )
    .expect("diagnostics");

    assert_eq!(
        diagnostics.receiver_channel_policy,
        BreastUstReceiverChannelPolicy::PassiveOnly
    );
    assert_eq!(diagnostics.best_model, "passive_exact");
    assert!(diagnostics.best_normalized_increment_residual <= 1.0e-14);
    assert!(scattering_model(&diagnostics, "active_only").normalized_increment_residual > 0.99);
}

#[test]
fn scattering_increment_diagnostics_reject_zero_observed_increment() {
    let baseline = Array3::from_elem((1, 1, 2), Complex64::new(1.0, 0.0));
    let observed = baseline.mapv(|value| Complex64::new(2.0, 0.0) * value);
    let predictions = [BreastUstForwardOperatorPrediction {
        model: "baseline",
        pressure: &baseline,
    }];

    let error = scattering_increment_diagnostics(
        &baseline,
        &predictions,
        &observed,
        BreastUstReceiverChannelPolicy::All,
    )
    .unwrap_err();

    assert!(error.to_string().contains("zero energy"));
}

fn scattering_model<'a>(
    diagnostics: &'a BreastUstScatteringIncrementDiagnostics,
    model: &str,
) -> &'a BreastUstScatteringIncrementModelDiagnostics {
    diagnostics
        .per_model
        .iter()
        .find(|row| row.model == model)
        .expect("model diagnostics")
}
