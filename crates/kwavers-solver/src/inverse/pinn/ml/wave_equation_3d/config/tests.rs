use super::*;

#[test]
fn test_config_default() {
    let config = PinnConfig3D::default();
    assert_eq!(config.hidden_layers, vec![100, 100, 100]);
    assert_eq!(config.num_collocation_points, 10000);
    assert_eq!(config.learning_rate, 1e-4);
    assert_eq!(config.batch_size, 1000);
    assert_eq!(config.max_grad_norm, 1.0);
}

#[test]
fn test_loss_weights_default() {
    let weights = LossWeights3D::default();
    assert_eq!(weights.data_weight, 1.0);
    assert_eq!(weights.pde_weight, 1.0);
    assert_eq!(weights.bc_weight, 1.0);
    assert_eq!(weights.ic_weight, 1.0);
}

#[test]
fn test_metrics_default() {
    let metrics = TrainingMetrics3D::default();
    assert_eq!(metrics.epochs_completed, 0);
    assert!(metrics.total_loss.is_empty());
    assert!(metrics.data_loss.is_empty());
    assert!(metrics.pde_loss.is_empty());
    assert!(metrics.bc_loss.is_empty());
    assert!(metrics.ic_loss.is_empty());
    assert_eq!(metrics.training_time_secs, 0.0);
}

#[test]
fn test_config_custom() {
    let config = PinnConfig3D {
        hidden_layers: vec![200, 200],
        num_collocation_points: 5000,
        learning_rate: 5e-4,
        batch_size: 500,
        max_grad_norm: 0.5,
        ..Default::default()
    };

    assert_eq!(config.hidden_layers, vec![200, 200]);
    assert_eq!(config.num_collocation_points, 5000);
    assert_eq!(config.learning_rate, 5e-4);
    assert_eq!(config.batch_size, 500);
    assert_eq!(config.max_grad_norm, 0.5);
}

#[test]
fn test_loss_weights_custom() {
    let weights = LossWeights3D {
        data_weight: 2.0,
        pde_weight: 10.0,
        bc_weight: 5.0,
        ic_weight: 3.0,
    };

    assert_eq!(weights.data_weight, 2.0);
    assert_eq!(weights.pde_weight, 10.0);
    assert_eq!(weights.bc_weight, 5.0);
    assert_eq!(weights.ic_weight, 3.0);
}

#[test]
fn test_metrics_update() {
    let mut metrics = TrainingMetrics3D {
        epochs_completed: 100,
        ..Default::default()
    };

    metrics.total_loss.push(0.1);
    metrics.total_loss.push(0.05);
    metrics.data_loss.push(0.04);
    metrics.pde_loss.push(0.03);
    metrics.bc_loss.push(0.02);
    metrics.ic_loss.push(0.01);
    metrics.training_time_secs = 123.45;

    assert_eq!(metrics.epochs_completed, 100);
    assert_eq!(metrics.total_loss.len(), 2);
    assert_eq!(metrics.total_loss[1], 0.05);
    assert_eq!(metrics.training_time_secs, 123.45);
}

#[test]
fn test_config_clone() {
    let config1 = PinnConfig3D::default();
    let config2 = config1.clone();

    assert_eq!(config1.hidden_layers, config2.hidden_layers);
    assert_eq!(
        config1.num_collocation_points,
        config2.num_collocation_points
    );
    assert_eq!(config1.learning_rate, config2.learning_rate);
}

#[test]
fn test_metrics_clone() {
    let mut metrics1 = TrainingMetrics3D {
        epochs_completed: 50,
        ..Default::default()
    };
    metrics1.total_loss.push(0.123);

    let metrics2 = metrics1.clone();

    assert_eq!(metrics1.epochs_completed, metrics2.epochs_completed);
    assert_eq!(metrics1.total_loss, metrics2.total_loss);
}
