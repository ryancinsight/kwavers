use super::*;
use ndarray::Array2;

#[test]
fn test_ensemble_quantifier_creation() {
    let config = EnsembleConfig {
        ensemble_size: 5,
        num_samples: 10,
    };
    let q = EnsembleQuantifier::new(config).unwrap();
    assert_eq!(
        q.ensemble_models.len(),
        5,
        "ensemble must have exactly 5 models"
    );
}

#[test]
fn test_bootstrap_sampling() {
    let quantifier = EnsembleQuantifier::new(EnsembleConfig::default()).unwrap();
    let indices = quantifier.bootstrap_sample(100);
    assert_eq!(
        indices.len(),
        100,
        "bootstrap must return exactly 100 indices"
    );
    for &idx in &indices {
        assert!(idx < 100, "bootstrap index {idx} out of range [0,100)");
    }
}

#[test]
fn test_ensemble_statistics() {
    // predictions: 1.0, 1.1, 0.9 with equal weights on (5,5)
    // mean = (1.0+1.1+0.9)/3 = 1.0 per element
    // weighted variance = ((0² + 0.1² + 0.1²)·1.0) / 3.0 = 0.02/3 ≈ 0.006667
    // uncertainty (std) = sqrt(0.006667) ≈ 0.08165
    let quantifier = EnsembleQuantifier::new(EnsembleConfig::default()).unwrap();
    let predictions = vec![
        Array2::from_elem((5, 5), 1.0_f32),
        Array2::from_elem((5, 5), 1.1_f32),
        Array2::from_elem((5, 5), 0.9_f32),
    ];
    let weights = vec![1.0_f64, 1.0, 1.0];

    let result = quantifier
        .compute_ensemble_statistics(&predictions, &weights)
        .unwrap();

    assert_eq!(result.mean_prediction.dim(), (5, 5));
    assert_eq!(result.uncertainty.dim(), (5, 5));

    let mean_err = (result.mean_prediction[[0, 0]] - 1.0_f32).abs();
    assert!(
        mean_err < 1e-5,
        "weighted mean = {} (expected 1.0)",
        result.mean_prediction[[0, 0]]
    );

    let expected_std = (0.02_f32 / 3.0).sqrt();
    let std_err = (result.uncertainty[[0, 0]] - expected_std).abs();
    assert!(
        std_err < 1e-4,
        "uncertainty = {} (expected ≈ {expected_std})",
        result.uncertainty[[0, 0]]
    );

    assert!(
        result.reliability_score > 0.0 && result.reliability_score <= 1.0,
        "reliability_score {} out of (0,1]",
        result.reliability_score
    );

    let (lo, hi) = &result.confidence_intervals["95%"];
    assert!(
        lo[[0, 0]] < result.mean_prediction[[0, 0]],
        "95% CI lower bound must be < mean"
    );
    assert!(
        hi[[0, 0]] > result.mean_prediction[[0, 0]],
        "95% CI upper bound must be > mean"
    );
}

#[test]
fn test_ensemble_diversity() {
    // predictions: 1.0, 1.5, 2.0 on (3,3)
    // pairwise L2 distances: ||1.0-1.5||=sqrt(9·0.25)=1.5, ||1.0-2.0||=3.0, ||1.5-2.0||=1.5
    // diversity = (1.5+3.0+1.5)/3 = 2.0
    let quantifier = EnsembleQuantifier::new(EnsembleConfig::default()).unwrap();
    let predictions = vec![
        Array2::from_elem((3, 3), 1.0_f32),
        Array2::from_elem((3, 3), 1.5_f32),
        Array2::from_elem((3, 3), 2.0_f32),
    ];

    let diversity = quantifier.compute_ensemble_diversity(&predictions);
    let diversity_err = (diversity - 2.0).abs();
    assert!(
        diversity_err < 1e-5,
        "diversity = {diversity} (expected 2.0)"
    );
    assert!(
        diversity > 0.0,
        "diverse ensemble must have positive diversity"
    );
}

#[test]
fn test_ensemble_statistics_empty_rejects() {
    let quantifier = EnsembleQuantifier::new(EnsembleConfig::default()).unwrap();
    let err = quantifier
        .compute_ensemble_statistics(&[], &[])
        .unwrap_err();
    let msg = format!("{err:?}");
    assert!(
        msg.contains("No predictions"),
        "empty-input error must mention 'No predictions'; got: {msg}"
    );
}
