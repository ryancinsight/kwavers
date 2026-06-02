use super::*;

#[test]
fn test_default_config() {
    let config = BurnPINNConfig::default();
    assert_eq!(config.hidden_layers, vec![50, 50, 50, 50]);
    assert_eq!(config.learning_rate, 1e-3);
    assert_eq!(config.num_collocation_points, 10_000);
    assert!(config.validate().is_ok());
}

#[test]
fn test_gpu_config() {
    let config = BurnPINNConfig::for_gpu();
    assert_eq!(config.hidden_layers, vec![100, 100, 100, 100, 100]);
    assert_eq!(config.learning_rate, 5e-4);
    assert_eq!(config.num_collocation_points, 50_000);
    assert!(config.validate().is_ok());
}

#[test]
fn test_prototyping_config() {
    let config = BurnPINNConfig::for_prototyping();
    assert_eq!(config.hidden_layers, vec![20, 20, 20]);
    assert_eq!(config.learning_rate, 1e-3);
    assert_eq!(config.num_collocation_points, 1_000);
    assert!(config.validate().is_ok());
}

#[test]
fn test_config_validation_empty_layers() {
    let config = BurnPINNConfig {
        hidden_layers: vec![],
        ..Default::default()
    };
    assert!(config.validate().is_err());
}

#[test]
fn test_config_validation_zero_layer_size() {
    let config = BurnPINNConfig {
        hidden_layers: vec![50, 0, 50],
        ..Default::default()
    };
    assert!(config.validate().is_err());
}

#[test]
fn test_config_validation_negative_learning_rate() {
    let config = BurnPINNConfig {
        learning_rate: -0.001,
        ..Default::default()
    };
    assert!(config.validate().is_err());
}

#[test]
fn test_config_validation_insufficient_collocation_points() {
    let config = BurnPINNConfig {
        num_collocation_points: 50,
        ..Default::default()
    };
    assert!(config.validate().is_err());
}

#[test]
fn test_num_parameters_default() {
    let config = BurnPINNConfig::default();
    let params = config.num_parameters();
    // Input: 2*50+50=150; Hidden×3: 3×(50*50+50)=7650; Output: 50+1=51 → 7851
    assert_eq!(params, 7851);
}

#[test]
fn test_num_parameters_simple() {
    let config = BurnPINNConfig {
        hidden_layers: vec![10, 10],
        ..Default::default()
    };
    let params = config.num_parameters();
    // Input: 2*10+10=30; Hidden: 10*10+10=110; Output: 10+1=11 → 151
    assert_eq!(params, 151);
}

#[test]
fn test_default_loss_weights() {
    let weights = BurnLossWeights::default();
    assert_eq!(weights.data, 1.0);
    assert_eq!(weights.pde, 1.0);
    assert_eq!(weights.boundary, 10.0);
    assert!(weights.validate().is_ok());
}

#[test]
fn test_data_driven_weights() {
    let weights = BurnLossWeights::data_driven();
    assert_eq!(weights.data, 10.0);
    assert_eq!(weights.pde, 1.0);
    assert_eq!(weights.boundary, 5.0);
    assert!(weights.validate().is_ok());
}

#[test]
fn test_physics_driven_weights() {
    let weights = BurnLossWeights::physics_driven();
    assert_eq!(weights.data, 0.1);
    assert_eq!(weights.pde, 10.0);
    assert_eq!(weights.boundary, 10.0);
    assert!(weights.validate().is_ok());
}

#[test]
fn test_balanced_weights() {
    let weights = BurnLossWeights::balanced();
    assert_eq!(weights.data, 1.0);
    assert_eq!(weights.pde, 1.0);
    assert_eq!(weights.boundary, 1.0);
    assert!(weights.validate().is_ok());
}

#[test]
fn test_loss_weights_validation_negative() {
    assert!(BurnLossWeights {
        data: -1.0,
        ..Default::default()
    }
    .validate()
    .is_err());
    assert!(BurnLossWeights {
        pde: -1.0,
        ..Default::default()
    }
    .validate()
    .is_err());
    assert!(BurnLossWeights {
        boundary: -1.0,
        ..Default::default()
    }
    .validate()
    .is_err());
}

#[test]
fn test_loss_weights_validation_infinite() {
    let weights = BurnLossWeights {
        data: f64::INFINITY,
        ..Default::default()
    };
    assert!(weights.validate().is_err());
}
