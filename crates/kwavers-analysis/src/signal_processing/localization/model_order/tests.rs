use super::estimator::ModelOrderEstimator;
use super::types::{ModelOrderConfig, ModelOrderCriterion, ModelOrderResult};

#[test]
fn test_config_creation() {
    let config = ModelOrderConfig::new(4, 100).unwrap();
    assert_eq!(config.num_sensors, 4);
    assert_eq!(config.num_samples, 100);
    assert_eq!(config.max_sources, Some(3));
}

#[test]
fn test_config_validation_too_few_sensors() {
    let result = ModelOrderConfig::new(1, 100);
    assert!(result.is_err());
}

#[test]
fn test_config_validation_too_few_samples() {
    let result = ModelOrderConfig::new(10, 5);
    assert!(result.is_err());
}

#[test]
fn test_estimator_creation() {
    let config = ModelOrderConfig::new(4, 100).unwrap();
    let _estimator = ModelOrderEstimator::new(config).unwrap();
}

#[test]
fn test_single_source_clear_gap() {
    // Eigenvalues with clear signal/noise separation
    // 1 large eigenvalue (signal), 3 small eigenvalues (noise)
    let eigenvalues = vec![10.0, 1.0, 1.0, 1.0];

    let config = ModelOrderConfig::new(4, 100).unwrap();
    let estimator = ModelOrderEstimator::new(config).unwrap();

    let result = estimator.estimate(&eigenvalues).unwrap();

    // Should detect 1 source
    assert_eq!(result.num_sources, 1);
    assert_eq!(result.signal_indices, vec![0]);
    assert_eq!(result.noise_indices, vec![1, 2, 3]);
}

#[test]
fn test_two_sources_clear_gap() {
    // 2 large eigenvalues (signal), 2 small eigenvalues (noise)
    let eigenvalues = vec![15.0, 10.0, 1.0, 1.0];

    let config = ModelOrderConfig::new(4, 100).unwrap();
    let estimator = ModelOrderEstimator::new(config).unwrap();

    let result = estimator.estimate(&eigenvalues).unwrap();

    // Should detect 2 sources
    assert_eq!(result.num_sources, 2);
    assert_eq!(result.signal_indices, vec![0, 1]);
    assert_eq!(result.noise_indices, vec![2, 3]);
}

#[test]
fn test_no_sources_all_noise() {
    // All eigenvalues approximately equal (pure noise)
    let eigenvalues = vec![1.01, 1.0, 0.99, 1.0];

    let config = ModelOrderConfig::new(4, 100).unwrap();
    let estimator = ModelOrderEstimator::new(config).unwrap();

    let result = estimator.estimate(&eigenvalues).unwrap();

    // Should detect 0 sources (all noise)
    assert_eq!(result.num_sources, 0);
    assert_eq!(result.signal_indices.len(), 0);
    assert_eq!(result.noise_indices.len(), 4);
}

#[test]
fn test_aic_vs_mdl() {
    let eigenvalues = vec![10.0, 5.0, 1.0, 1.0];

    // Test with AIC
    let config_aic = ModelOrderConfig::new(4, 100)
        .unwrap()
        .with_criterion(ModelOrderCriterion::AIC);
    let estimator_aic = ModelOrderEstimator::new(config_aic).unwrap();
    let result_aic = estimator_aic.estimate(&eigenvalues).unwrap();

    // Test with MDL
    let config_mdl = ModelOrderConfig::new(4, 100)
        .unwrap()
        .with_criterion(ModelOrderCriterion::MDL);
    let estimator_mdl = ModelOrderEstimator::new(config_mdl).unwrap();
    let result_mdl = estimator_mdl.estimate(&eigenvalues).unwrap();

    // Both should work (may give same or different answers)
    assert!(result_aic.num_sources <= 3);
    assert!(result_mdl.num_sources <= 3);

    // MDL typically more conservative (same or fewer sources than AIC)
    assert!(result_mdl.num_sources <= result_aic.num_sources);
}

#[test]
fn test_noise_variance_estimation() {
    let eigenvalues = vec![20.0, 15.0, 2.0, 2.0, 2.0];

    let config = ModelOrderConfig::new(5, 100).unwrap();
    let estimator = ModelOrderEstimator::new(config).unwrap();

    let result = estimator.estimate(&eigenvalues).unwrap();

    // Noise variance should be close to 2.0
    let noise_var = result.noise_variance();
    assert!((noise_var - 2.0).abs() < 0.5);
}

#[test]
fn test_eigenvalue_threshold_filtering() {
    // Include a very small eigenvalue that should be filtered
    let eigenvalues = vec![10.0, 1.0, 1.0, 1e-12];

    let config = ModelOrderConfig::new(4, 100)
        .unwrap()
        .with_eigenvalue_threshold(1e-10);
    let estimator = ModelOrderEstimator::new(config).unwrap();

    let result = estimator.estimate(&eigenvalues).unwrap();

    // Should still work despite near-zero eigenvalue
    assert!(result.num_sources <= 3);
}

#[test]
fn test_max_sources_constraint() {
    let eigenvalues = vec![10.0, 9.0, 8.0, 7.0, 6.0];

    // Limit to max 2 sources
    let config = ModelOrderConfig::new(5, 100).unwrap().with_max_sources(2);
    let estimator = ModelOrderEstimator::new(config).unwrap();

    let result = estimator.estimate(&eigenvalues).unwrap();

    // Should not exceed max_sources
    assert!(result.num_sources <= 2);
}

#[test]
fn test_criterion_values_length() {
    let eigenvalues = vec![10.0, 5.0, 2.0, 1.0];

    let config = ModelOrderConfig::new(4, 100).unwrap();
    let estimator = ModelOrderEstimator::new(config).unwrap();

    let result = estimator.estimate(&eigenvalues).unwrap();

    // Should have criterion values for k = 0, 1, 2, 3 (max_sources = 3)
    assert!(!result.criterion_values.is_empty());
    assert!(result.criterion_values.len() <= 4);
}

#[test]
fn test_subspace_eigenvalues() {
    let eigenvalues = vec![20.0, 15.0, 2.0, 2.0];

    let config = ModelOrderConfig::new(4, 100).unwrap();
    let estimator = ModelOrderEstimator::new(config).unwrap();

    let result = estimator.estimate(&eigenvalues).unwrap();

    let signal_eigs = result.signal_eigenvalues();
    let noise_eigs = result.noise_eigenvalues();

    // Signal eigenvalues should be larger than noise eigenvalues
    if !signal_eigs.is_empty() && !noise_eigs.is_empty() {
        let min_signal = signal_eigs.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_noise = noise_eigs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert!(min_signal >= max_noise);
    }

    // Total should equal original eigenvalues
    assert_eq!(signal_eigs.len() + noise_eigs.len(), eigenvalues.len());
}

// Suppress unused import warning for ModelOrderResult in scope
const _: fn() = || {
    let _: ModelOrderResult;
};
