use kwavers_diagnostics::reconstruction::breast_ust_fwi::{
    scattering_increment_diagnostics, BreastUstForwardOperatorPrediction,
    BreastUstReceiverChannelPolicy,
};
use kwavers_math::fft::Complex64;
use ndarray::Array3;

#[test]
fn scattering_increment_public_api_identifies_exact_model() {
    let baseline = Array3::from_elem((1, 2, 4), Complex64::new(1.0, 0.0));
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

    assert_eq!(diagnostics.model_count, 2);
    assert_eq!(diagnostics.best_model, "exact_increment");
    assert!(diagnostics.best_normalized_increment_residual <= 1.0e-14);
    assert_eq!(
        diagnostics.best_model_scaled_increment_model,
        "exact_increment"
    );
    assert!(diagnostics.best_model_scaled_normalized_increment_residual <= 1.0e-14);
    assert_eq!(diagnostics.worst_model, "half_increment");
    let exact_row = diagnostics
        .per_model
        .iter()
        .find(|row| row.model == "exact_increment")
        .expect("exact row");
    // Exact model: normalized scattering increment residual must be numerically zero.
    assert!(exact_row.normalized_increment_residual <= 1.0e-14);
    // Exact model: per-row mean and max residuals must also be numerically zero.
    assert!(exact_row.row_normalized_increment_residual_mean <= 1.0e-14);
    assert!(exact_row.row_normalized_increment_residual_max <= 1.0e-14);
    // Exact model: predicted increment energy equals observed increment energy.
    assert!((exact_row.increment_energy_ratio - 1.0).abs() <= 1.0e-12);
    assert!(exact_row.baseline_scaled_full_field_normalized_residual <= 1.0e-14);
    assert!(exact_row.model_scaled_full_field_normalized_residual <= 1.0e-14);
    assert!(exact_row.model_scaled_observed_increment_l2_norm > 0.0);
    assert!(exact_row.model_scaled_increment_residual_l2_norm <= 1.0e-14);
    assert!(exact_row.model_scaled_normalized_increment_residual <= 1.0e-14);
    assert!((exact_row.model_scaled_increment_energy_ratio - 1.0).abs() <= 1.0e-12);
    assert!(exact_row.source_scale_relative_drift_mean <= 1.0e-14);
    assert!(exact_row.source_scale_relative_drift_max <= 1.0e-14);
    assert!(exact_row.source_scale_phase_drift_mean_abs_rad <= 1.0e-14);
    assert!(exact_row.source_scale_phase_drift_max_abs_rad <= 1.0e-14);
    assert!(
        (diagnostics
            .per_model
            .iter()
            .find(|row| row.model == "half_increment")
            .expect("half row")
            .normalized_increment_residual
            - 0.5)
            .abs()
            <= 1.0e-14
    );
}

#[test]
fn scattering_increment_public_api_reports_nonzero_residual_for_mismatched_model() {
    let baseline = Array3::from_shape_vec(
        (1, 1, 2),
        vec![Complex64::new(1.0, 0.0), Complex64::new(1.0, 0.0)],
    )
    .expect("baseline shape");
    let prediction = Array3::from_shape_vec(
        (1, 1, 2),
        vec![Complex64::new(2.0, 0.0), Complex64::new(1.0, 0.0)],
    )
    .expect("prediction shape");
    let model_scale = Complex64::new(3.0, 0.0);
    let observed = prediction.mapv(|value| model_scale * value);
    let predictions = [BreastUstForwardOperatorPrediction {
        model: "model_scaled",
        pressure: &prediction,
    }];

    let diagnostics = scattering_increment_diagnostics(
        &baseline,
        &predictions,
        &observed,
        BreastUstReceiverChannelPolicy::All,
    )
    .expect("diagnostics");
    let row = diagnostics.per_model.first().expect("model row");

    // The prediction model does not perfectly explain the scattering increment
    // when scaled by the baseline source scale — residual must exceed 1.0.
    assert!(row.normalized_increment_residual > 1.0);
    assert!(row.model_scaled_full_field_normalized_residual <= 1.0e-14);
    assert!(row.model_scaled_observed_increment_l2_norm > 0.0);
    assert!(row.model_scaled_increment_residual_l2_norm <= 1.0e-14);
    assert!(row.model_scaled_normalized_increment_residual <= 1.0e-14);
    assert!((row.model_scaled_increment_energy_ratio - 2.0_f64.sqrt()).abs() <= 1.0e-14);
    assert!(row.baseline_scaled_full_field_normalized_residual > 0.0);
    assert!((row.source_scale_relative_drift_mean - (1.0 / 3.0)).abs() <= 1.0e-14);
    assert!((row.source_scale_relative_drift_max - (1.0 / 3.0)).abs() <= 1.0e-14);
    assert!(row.source_scale_phase_drift_mean_abs_rad <= 1.0e-14);
    assert!(row.source_scale_phase_drift_max_abs_rad <= 1.0e-14);
    // Residual has non-zero L2 norm.
    assert!(row.increment_residual_l2_norm > 0.0);
}
