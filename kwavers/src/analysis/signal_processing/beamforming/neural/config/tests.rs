use super::*;

#[test]
fn test_mode_default() {
    assert_eq!(
        NeuralBeamformingMode::default(),
        NeuralBeamformingMode::Hybrid
    );
}

#[test]
fn test_physics_parameters_default() {
    let params = PhysicsParameters::default();
    assert_eq!(params.reciprocity_weight, 1.0);
    assert_eq!(params.coherence_weight, 0.5);
    assert_eq!(params.sparsity_weight, 0.1);
}

#[test]
fn test_sensor_geometry_linear() {
    let geometry = SensorGeometry::linear_array(64, 0.0003, 40e6, 1540.0);
    assert_eq!(geometry.num_elements(), 64);
    assert_eq!(geometry.sampling_frequency, 40e6);

    assert!((geometry.positions[31][0] + geometry.positions[32][0]).abs() < 1e-10);
}

#[test]
fn test_sensor_geometry_phased() {
    let geometry = SensorGeometry::phased_array(8, 8, 0.0003, 0.0003, 40e6, 1540.0);
    assert_eq!(geometry.num_elements(), 64);
}

#[test]
fn test_config_validation_valid() {
    let config = NeuralBeamformingConfig::default();
    assert!(config.validate().is_ok());
}

#[test]
fn test_config_validation_invalid_architecture() {
    let config = NeuralBeamformingConfig {
        network_architecture: vec![5],
        ..Default::default()
    };
    assert!(config.validate().is_err());
}

#[test]
fn test_config_validation_invalid_physics_weight() {
    let mut config = NeuralBeamformingConfig::default();
    config.physics_parameters.reciprocity_weight = -1.0;
    assert!(config.validate().is_err());
}

#[test]
fn test_config_validation_invalid_learning_rate() {
    let mut config = NeuralBeamformingConfig::default();
    config.adaptation_parameters.learning_rate = 0.0;
    assert!(config.validate().is_err());
}

#[test]
fn test_config_validation_invalid_batch_size() {
    let config = NeuralBeamformingConfig {
        batch_size: 0,
        ..Default::default()
    };
    assert!(config.validate().is_err());
}

#[test]
fn test_config_validation_invalid_sensor_count() {
    let mut config = NeuralBeamformingConfig::default();
    config.sensor_geometry.positions = vec![[0.0, 0.0, 0.0]];
    assert!(config.validate().is_err());
}
