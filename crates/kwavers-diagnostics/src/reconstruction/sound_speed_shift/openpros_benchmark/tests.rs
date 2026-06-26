use super::super::{
    ShiftPrior, ShiftSampling, ShiftSensitivity, FINITE_FREQUENCY_SOUND_SPEED_SHIFT_MODEL,
};
use super::{
    openpros_shift_benchmark_case, run_openpros_shift_benchmark, OpenProsShiftBenchmarkConfig,
    OPENPROS_PAPER_ID,
};
use kwavers_core::constants::numerical::MHZ_TO_HZ;

#[test]
fn openpros_case_matches_limited_view_waveform_structure() {
    let config = OpenProsShiftBenchmarkConfig::default();
    let case = openpros_shift_benchmark_case(&config).unwrap();

    assert_eq!(case.waveform.paper_id, OPENPROS_PAPER_ID);
    assert_eq!(case.waveform.source_channels, 40);
    assert_eq!(case.waveform.receivers_per_channel, 17);
    assert_eq!(case.waveform.time_steps, 1_000);
    assert_eq!(case.waveform.peak_frequency_hz, MHZ_TO_HZ);
    assert_eq!(case.waveform.absorbing_boundary_points, 120);
    assert_eq!(case.waveform.sos_shape, (41, 17));
    assert_eq!(case.samples.len(), 40 * 17);
    assert_eq!(case.frame_time_shifts_s.len(), case.samples.len());
    assert!(case.truth_shift_m_s.iter().any(|value| *value > 100.0));
    assert!(case.truth_shift_m_s.iter().any(|value| *value < -20.0));
    assert_eq!(case.dense_config.prior, ShiftPrior::Dense);
    assert_eq!(case.dense_config.sampling, ShiftSampling::Dense);
    assert_eq!(case.sparse_config.prior, ShiftPrior::Sparse);
    assert_eq!(
        case.sparse_config.sampling,
        ShiftSampling::Sparse {
            stride: config.sparse_stride,
            offset: 0
        }
    );
    assert!(matches!(
        case.dense_config.sensitivity,
        ShiftSensitivity::FiniteFrequency { .. }
    ));
}

#[test]
fn openpros_benchmark_runs_dense_and_sparse_reconstructions() {
    let config = OpenProsShiftBenchmarkConfig {
        spatial_decimation: 20,
        source_count_per_probe: 4,
        receiver_count_per_probe: 9,
        dense_iterations: 16,
        sparse_iterations: 24,
        sparse_stride: 3,
        ..OpenProsShiftBenchmarkConfig::default()
    };

    let result = run_openpros_shift_benchmark(&config).unwrap();

    assert_eq!(
        result.dense_reconstruction.model_family,
        FINITE_FREQUENCY_SOUND_SPEED_SHIFT_MODEL
    );
    assert_eq!(
        result.sparse_reconstruction.model_family,
        FINITE_FREQUENCY_SOUND_SPEED_SHIFT_MODEL
    );
    assert_eq!(result.dense_metrics.rows_available, 4 * 4 * 9);
    assert_eq!(
        result.dense_metrics.rows_used,
        result.dense_metrics.rows_available
    );
    assert!(result.sparse_metrics.rows_used < result.dense_metrics.rows_used);
    assert_eq!(
        result.dense_metrics.active_voxels,
        result.sparse_metrics.active_voxels
    );
    assert!(result.dense_metrics.stored_weight_count > result.dense_metrics.rows_used);
    assert!(result.sparse_metrics.stored_weight_count > result.sparse_metrics.rows_used);
    // Value-semantic recovery quality (measured: dense pearson≈0.74, nrmse≈0.61,
    // obj-reduction≈0.9999; sparse pearson≈0.58, nrmse≈0.74). Thresholds bound the
    // recovered shift well above a broken reconstruction (pearson≈0, nrmse≈1) yet
    // below the achieved values, so a regression that degraded the inversion fails.
    assert!(
        result.dense_metrics.objective_reduction_fraction > 0.9,
        "dense inversion must strongly fit the data, got obj-reduction {}",
        result.dense_metrics.objective_reduction_fraction
    );
    assert!(
        result.dense_metrics.pearson_correlation > 0.6,
        "dense reconstruction must correlate with the truth shift, got r {}",
        result.dense_metrics.pearson_correlation
    );
    assert!(
        result.dense_metrics.normalized_root_mean_square_error < 0.7,
        "dense NRMSE must be bounded, got {}",
        result.dense_metrics.normalized_root_mean_square_error
    );
    assert!(
        result.sparse_metrics.pearson_correlation > 0.4,
        "sparse reconstruction must still correlate with the truth shift, got r {}",
        result.sparse_metrics.pearson_correlation
    );
    // Dense uses every available row, so it must be at least as accurate as sparse.
    assert!(
        result.dense_metrics.normalized_root_mean_square_error
            <= result.sparse_metrics.normalized_root_mean_square_error,
        "dense NRMSE {} must not exceed sparse {}",
        result.dense_metrics.normalized_root_mean_square_error,
        result.sparse_metrics.normalized_root_mean_square_error
    );
}

#[test]
fn openpros_benchmark_rejects_degenerate_sparse_sampling() {
    let config = OpenProsShiftBenchmarkConfig {
        sparse_stride: 1,
        ..OpenProsShiftBenchmarkConfig::default()
    };

    let err = openpros_shift_benchmark_case(&config).unwrap_err();

    assert!(err.to_string().contains("sparse_stride"));
}
