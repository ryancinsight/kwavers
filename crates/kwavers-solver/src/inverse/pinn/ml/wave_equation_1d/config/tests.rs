use super::*;

#[test]
fn test_default_config() {
    let config = PinnConfig::default();
    assert_eq!(config.hidden_layers, vec![50, 50, 50, 50]);
    assert_eq!(config.learning_rate, 1e-3);
    assert_eq!(config.num_collocation_points, 10_000);
    assert!(config.validate().is_ok());
}

#[test]
fn test_gpu_config() {
    let config = PinnConfig::for_gpu();
    assert_eq!(config.hidden_layers, vec![100, 100, 100, 100, 100]);
    assert_eq!(config.learning_rate, 5e-4);
    assert_eq!(config.num_collocation_points, 50_000);
    assert!(config.validate().is_ok());
}

#[test]
fn test_prototyping_config() {
    let config = PinnConfig::for_prototyping();
    assert_eq!(config.hidden_layers, vec![20, 20, 20]);
    assert_eq!(config.learning_rate, 1e-3);
    assert_eq!(config.num_collocation_points, 1_000);
    assert!(config.validate().is_ok());
}

#[test]
fn test_config_validation_empty_layers() {
    let config = PinnConfig {
        hidden_layers: vec![],
        ..Default::default()
    };
    assert!(config.validate().is_err());
}

#[test]
fn test_config_validation_zero_layer_size() {
    let config = PinnConfig {
        hidden_layers: vec![50, 0, 50],
        ..Default::default()
    };
    assert!(config.validate().is_err());
}

#[test]
fn test_config_validation_negative_learning_rate() {
    let config = PinnConfig {
        learning_rate: -0.001,
        ..Default::default()
    };
    assert!(config.validate().is_err());
}

#[test]
fn test_config_validation_insufficient_collocation_points() {
    let config = PinnConfig {
        num_collocation_points: 50,
        ..Default::default()
    };
    assert!(config.validate().is_err());
}

#[test]
fn test_num_parameters_default() {
    let config = PinnConfig::default();
    let params = config.num_parameters();
    // Input: 2*50+50=150; Hidden×3: 3×(50*50+50)=7650; Output: 50+1=51 → 7851
    assert_eq!(params, 7851);
}

#[test]
fn test_num_parameters_simple() {
    let config = PinnConfig {
        hidden_layers: vec![10, 10],
        ..Default::default()
    };
    let params = config.num_parameters();
    // Input: 2*10+10=30; Hidden: 10*10+10=110; Output: 10+1=11 → 151
    assert_eq!(params, 151);
}

#[test]
fn test_default_loss_weights() {
    let weights = LossWeights::default();
    assert_eq!(weights.data, 1.0);
    assert_eq!(weights.pde, 1.0);
    assert_eq!(weights.boundary, 10.0);
    assert!(weights.validate().is_ok());
}

#[test]
fn test_data_driven_weights() {
    let weights = LossWeights::data_driven();
    assert_eq!(weights.data, 10.0);
    assert_eq!(weights.pde, 1.0);
    assert_eq!(weights.boundary, 5.0);
    assert!(weights.validate().is_ok());
}

#[test]
fn test_physics_driven_weights() {
    let weights = LossWeights::physics_driven();
    assert_eq!(weights.data, 0.1);
    assert_eq!(weights.pde, 10.0);
    assert_eq!(weights.boundary, 10.0);
    assert!(weights.validate().is_ok());
}

#[test]
fn test_balanced_weights() {
    let weights = LossWeights::balanced();
    assert_eq!(weights.data, 1.0);
    assert_eq!(weights.pde, 1.0);
    assert_eq!(weights.boundary, 1.0);
    assert!(weights.validate().is_ok());
}

#[test]
fn test_loss_weights_validation_negative() {
    assert!(LossWeights {
        data: -1.0,
        ..Default::default()
    }
    .validate()
    .is_err());
    assert!(LossWeights {
        pde: -1.0,
        ..Default::default()
    }
    .validate()
    .is_err());
    assert!(LossWeights {
        boundary: -1.0,
        ..Default::default()
    }
    .validate()
    .is_err());
}

#[test]
fn test_loss_weights_validation_infinite() {
    let weights = LossWeights {
        data: f64::INFINITY,
        ..Default::default()
    };
    assert!(weights.validate().is_err());
}
