use super::*;

#[test]
fn test_default_config() {
    let config = MetaLearningConfig::default();
    assert_eq!(config.inner_lr, 0.01);
    assert_eq!(config.outer_lr, 0.001);
    assert_eq!(config.adaptation_steps, 5);
    assert_eq!(config.meta_batch_size, 8);
    assert_eq!(config.meta_epochs, 1000);
    assert!(!config.first_order);
    assert_eq!(config.physics_regularization, 0.1);
    assert_eq!(config.num_layers, 4);
    assert_eq!(config.hidden_dim, 64);
    assert_eq!(config.input_dim, 3);
    assert_eq!(config.output_dim, 1);
    assert_eq!(config.max_tasks, 100);
}

#[test]
fn test_new_config_valid() {
    let config =
        MetaLearningConfig::new(0.01, 0.001, 5, 8, 1000, false, 0.1, 4, 64, 3, 1, 100).unwrap();
    assert_eq!(config.inner_lr, 0.01);
    assert_eq!(config.adaptation_steps, 5);
}

#[test]
fn test_new_config_invalid_inner_lr() {
    let result = MetaLearningConfig::new(-0.01, 0.001, 5, 8, 1000, false, 0.1, 4, 64, 3, 1, 100);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("inner_lr"));
}

#[test]
fn test_new_config_invalid_outer_lr() {
    let result = MetaLearningConfig::new(0.01, 0.0, 5, 8, 1000, false, 0.1, 4, 64, 3, 1, 100);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("outer_lr"));
}

#[test]
fn test_new_config_invalid_adaptation_steps() {
    let result = MetaLearningConfig::new(0.01, 0.001, 0, 8, 1000, false, 0.1, 4, 64, 3, 1, 100);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("adaptation_steps"));
}

#[test]
fn test_new_config_invalid_meta_batch_size() {
    let result = MetaLearningConfig::new(0.01, 0.001, 5, 0, 1000, false, 0.1, 4, 64, 3, 1, 100);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("meta_batch_size"));
}

#[test]
fn test_new_config_invalid_physics_regularization() {
    let result = MetaLearningConfig::new(0.01, 0.001, 5, 8, 1000, false, -0.1, 4, 64, 3, 1, 100);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("physics_regularization"));
}

#[test]
fn test_fast_config() {
    let config = MetaLearningConfig::fast();
    assert_eq!(config.adaptation_steps, 1);
    assert_eq!(config.meta_batch_size, 4);
    assert!(config.first_order);
}

#[test]
fn test_high_quality_config() {
    let config = MetaLearningConfig::high_quality();
    assert_eq!(config.adaptation_steps, 10);
    assert_eq!(config.meta_batch_size, 16);
    assert_eq!(config.meta_epochs, 5000);
    assert!(!config.first_order);
}

#[test]
fn test_large_scale_config() {
    let config = MetaLearningConfig::large_scale();
    assert_eq!(config.num_layers, 8);
    assert_eq!(config.hidden_dim, 256);
    assert_eq!(config.meta_batch_size, 32);
}
