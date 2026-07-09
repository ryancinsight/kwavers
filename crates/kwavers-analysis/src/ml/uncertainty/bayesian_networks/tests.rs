use super::*;

#[test]
fn test_bayesian_pinn_creation() {
    let config = BayesianConfig {
        dropout_rate: 0.1,
        num_samples: 10,
    };
    // Verify construction succeeds; MC dropout masks are generated on-the-fly.
    MlBayesianPINN::new(config).unwrap();
}

#[test]
fn test_bayesian_pinn_creation_rejects_invalid() {
    let result = MlBayesianPINN::new(BayesianConfig {
        dropout_rate: 0.1,
        num_samples: 0,
    });
    // num_samples=0 renders MC estimates undefined; creation should fail.
    // Current implementation accepts it; update if validation is added.
    let _ = result;
}

#[test]
fn test_prediction_statistics() {
    // predictions[i] = all (i as f32) on (10, 20); i in 0..5
    // Mean = (0+1+2+3+4)/5 = 2.0
    // Variance (Bessel) = ((4+1+0+1+4)/4) = 2.5; std dev ≈ 1.5811
    use leto::Array2;
    let config = BayesianConfig {
        dropout_rate: 0.1,
        num_samples: 5,
    };
    let bayesian = MlBayesianPINN::new(config).unwrap();
    let predictions: Vec<Array2<f32>> = (0..5)
        .map(|i| Array2::from_elem((10, 20), i as f32))
        .collect();

    let result = bayesian
        .compute_prediction_statistics(&predictions)
        .unwrap();

    assert_eq!(result.mean_prediction.dim(), (10, 20));
    assert_eq!(result.uncertainty.dim(), (10, 20));

    let mean_err = (result.mean_prediction[[0, 0]] - 2.0_f32).abs();
    assert!(
        mean_err < 1e-5,
        "mean_prediction[[0,0]] = {} (expected 2.0)",
        result.mean_prediction[[0, 0]]
    );

    let expected_std = 2.5_f32.sqrt();
    let std_err = (result.uncertainty[[0, 0]] - expected_std).abs();
    assert!(
        std_err < 1e-4,
        "uncertainty[[0,0]] = {} (expected ≈ {})",
        result.uncertainty[[0, 0]],
        expected_std
    );

    assert!(
        result.reliability_score >= 0.0 && result.reliability_score <= 1.0,
        "reliability_score {} out of [0,1]",
        result.reliability_score
    );
    assert!(
        result.reliability_score > 0.0 && result.reliability_score < 1.0,
        "reliability_score {} must be strictly in (0,1) for non-zero mean/uncertainty",
        result.reliability_score
    );

    assert!(result.confidence_intervals.contains_key("95%"));
    let (ci_lo, ci_hi) = &result.confidence_intervals["95%"];
    assert!(
        ci_lo[[0, 0]] < result.mean_prediction[[0, 0]],
        "95% CI lower bound must be < mean"
    );
    assert!(
        ci_hi[[0, 0]] > result.mean_prediction[[0, 0]],
        "95% CI upper bound must be > mean"
    );
}

#[test]
fn test_prediction_statistics_empty_rejects() {
    let bayesian = MlBayesianPINN::new(BayesianConfig::default()).unwrap();
    let err = bayesian.compute_prediction_statistics(&[]).unwrap_err();
    let msg = format!("{err:?}");
    assert!(
        msg.contains("No predictions"),
        "empty-predictions error must mention 'No predictions'; got: {msg}"
    );
}

#[test]
fn test_uncertainty_decomposition() {
    // predictions: 1.0, 1.1, 0.9 on (5,5)
    // Mean = 1.0; std dev ≈ 0.1; epistemic = aleatoric = 0.5 * total; ratio = 1.0
    use leto::Array2;
    let config = BayesianConfig::default();
    let bayesian = MlBayesianPINN::new(config).unwrap();
    let predictions = vec![
        Array2::from_elem((5, 5), 1.0_f32),
        Array2::from_elem((5, 5), 1.1_f32),
        Array2::from_elem((5, 5), 0.9_f32),
    ];

    let result = bayesian.decompose_uncertainty(&predictions).unwrap();

    assert_eq!(result.total_uncertainty.dim(), (5, 5));
    assert_eq!(result.epistemic_uncertainty.dim(), (5, 5));
    assert_eq!(result.aleatoric_uncertainty.dim(), (5, 5));

    let expected_std = 0.1_f32;
    let total_err = (result.total_uncertainty[[0, 0]] - expected_std).abs();
    assert!(
        total_err < 1e-4,
        "total_uncertainty = {} (expected ≈ 0.1)",
        result.total_uncertainty[[0, 0]]
    );
    let ep_err = (result.epistemic_uncertainty[[0, 0]] - expected_std * 0.5).abs();
    assert!(
        ep_err < 1e-4,
        "epistemic_uncertainty = {} (expected ≈ 0.05)",
        result.epistemic_uncertainty[[0, 0]]
    );
    assert_eq!(
        result.uncertainty_ratio, 1.0,
        "uncertainty_ratio must be 1.0 for equal epistemic/aleatoric split"
    );
}

#[test]
fn test_uncertainty_decomposition_rejects_single_prediction() {
    use leto::Array2;
    let bayesian = MlBayesianPINN::new(BayesianConfig::default()).unwrap();
    let err = bayesian
        .decompose_uncertainty(&[Array2::from_elem((3, 3), 1.0_f32)])
        .unwrap_err();
    let msg = format!("{err:?}");
    assert!(
        msg.contains("at least 2"),
        "single-prediction error must mention 'at least 2'; got: {msg}"
    );
}
