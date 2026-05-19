//! Tests for uncertainty quantification types and configuration.

use super::types::{
    PinnUncertaintyConfig, PinnUncertaintyMethod, PinnPredictionWithUncertainty, UncertaintyStats,
};

#[test]
fn test_uncertainty_config() {
    let config = PinnUncertaintyConfig {
        mc_samples: 50,
        dropout_prob: 0.1,
        ensemble_size: 5,
        conformal_alpha: 0.05,
        variance_threshold: 0.1,
    };

    assert_eq!(config.mc_samples, 50);
    assert_eq!(config.ensemble_size, 5);
}

#[test]
fn test_uncertainty_methods() {
    let methods = vec![
        PinnUncertaintyMethod::MCDropout,
        PinnUncertaintyMethod::DeepEnsemble,
        PinnUncertaintyMethod::Conformal,
        PinnUncertaintyMethod::Hybrid,
    ];

    for method in methods {
        // Methods should be debug-printable
        let _ = format!("{:?}", method);
    }
}

#[test]
fn test_prediction_with_uncertainty() {
    let prediction = PinnPredictionWithUncertainty {
        mean: vec![1.0, 0.5],
        std: vec![0.1, 0.05],
        confidence_interval: (vec![0.8, 0.4], vec![1.2, 0.6]),
        entropy: 0.5,
        reliability: 0.9,
        method: PinnUncertaintyMethod::DeepEnsemble,
    };

    assert_eq!(prediction.mean.len(), 2);
    assert_eq!(prediction.std.len(), 2);
    assert!(prediction.reliability >= 0.0 && prediction.reliability <= 1.0);
}

#[test]
fn test_conformal_predictor() {
    let config = PinnUncertaintyConfig {
        mc_samples: 0,
        dropout_prob: 0.0,
        ensemble_size: 1,
        conformal_alpha: 0.05,
        variance_threshold: 0.1,
    };

    assert!(config.conformal_alpha > 0.0 && config.conformal_alpha < 1.0);
}

#[test]
fn test_uncertainty_stats_default() {
    let stats = UncertaintyStats::default();
    assert_eq!(stats.total_predictions, 0);
    assert_eq!(stats.average_uncertainty, 0.0);
    assert_eq!(stats.calibration_error, 0.0);
    assert_eq!(stats.coverage_probability, 0.0);
    assert_eq!(stats.reliability_score, 0.0);
}
