use super::*;
use crate::inverse::pinn::geometry::CollocationSamplingStrategy;
use kwavers_core::constants::fundamental::DENSITY_WATER_NOMINAL;

#[test]
fn test_default_config_valid() {
    let config = Config::default();
    config.validate().expect("default configuration is valid");
    assert_eq!(
        config.sampling_strategy,
        CollocationSamplingStrategy::LatinHypercube
    );
    assert_eq!(config.n_collocation_interior, 10_000);
}

#[test]
fn test_forward_problem_config() {
    let config = Config::forward_problem(1e9, 5e8, DENSITY_WATER_NOMINAL);
    assert!(config.validate().is_ok());
    assert!(!config.optimize_lambda);
    assert!(!config.optimize_mu);
    assert!(!config.optimize_rho);
    assert_eq!(config.lambda_init, Some(1e9));
    assert_eq!(config.loss_weights.data, 0.0);
}

#[test]
fn test_inverse_problem_config() {
    let config = Config::inverse_problem(1e9, 5e8, DENSITY_WATER_NOMINAL);
    assert!(config.validate().is_ok());
    assert!(config.optimize_lambda);
    assert!(config.optimize_mu);
    assert!(config.optimize_rho);
    assert!(config.loss_weights.data > config.loss_weights.pde);
}

#[test]
fn test_invalid_empty_layers() {
    let config = Config {
        hidden_layers: vec![],
        ..Default::default()
    };
    assert!(config.validate().is_err());
}

#[test]
fn test_invalid_zero_neurons() {
    let config = Config {
        hidden_layers: vec![100, 0, 100],
        ..Default::default()
    };
    assert!(config.validate().is_err());
}

#[test]
fn test_invalid_learning_rate() {
    assert!(Config {
        learning_rate: 0.0,
        ..Default::default()
    }
    .validate()
    .is_err());
    assert!(Config {
        learning_rate: 1.5,
        ..Default::default()
    }
    .validate()
    .is_err());
}

#[test]
fn test_optimize_without_init() {
    let config = Config {
        optimize_lambda: true,
        lambda_init: None,
        ..Default::default()
    };
    assert!(config.validate().is_err());
}

#[test]
fn test_loss_weights_default() {
    let weights = LossWeights::default();
    assert_eq!(weights.pde, 1.0);
    assert_eq!(weights.boundary, 10.0);
    assert_eq!(weights.initial, 10.0);
    assert_eq!(weights.data, 1.0);
    assert_eq!(weights.interface, 10.0);
}

#[test]
fn canonical_sampling_strategy_round_trips_with_elastic_config() {
    let config = Config {
        sampling_strategy: CollocationSamplingStrategy::Sobol,
        ..Config::default()
    };
    let encoded = serde_json::to_string(&config).expect("configuration serializes");
    let decoded: Config = serde_json::from_str(&encoded).expect("configuration deserializes");
    assert_eq!(
        decoded.sampling_strategy,
        CollocationSamplingStrategy::Sobol
    );
}

#[test]
fn test_activation_function_equality() {
    assert_eq!(
        ElasticPinnActivationFunction::Tanh,
        ElasticPinnActivationFunction::Tanh
    );
    assert_ne!(
        ElasticPinnActivationFunction::Tanh,
        ElasticPinnActivationFunction::Sin
    );
}
