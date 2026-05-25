use kwavers::clinical::imaging::reconstruction::breast_ust_fwi::{
    scattering_increment_diagnostics, BreastUstForwardOperatorPrediction,
    BreastUstReceiverChannelPolicy,
};
use ndarray::Array3;
use num_complex::Complex64;

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
    assert_eq!(diagnostics.worst_model, "half_increment");
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
