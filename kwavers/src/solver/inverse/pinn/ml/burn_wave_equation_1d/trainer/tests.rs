use super::super::config::BurnPINNConfig;
use super::trainer_impl::BurnPINNTrainer;
use burn::backend::{Autodiff, NdArray};
use crate::core::constants::fundamental::SOUND_SPEED_AIR;
use ndarray::{Array1, Array2};

type TestBackend = Autodiff<NdArray<f32>>;

#[test]
fn test_trainer_creation() {
    let device = Default::default();
    let config = BurnPINNConfig {
        hidden_layers: vec![10, 10],
        ..Default::default()
    };
    let trainer = BurnPINNTrainer::<TestBackend>::new(config, &device);
    let _trainer = trainer.unwrap();
}

#[test]
fn test_trainer_with_invalid_config() {
    let device = Default::default();
    let config = BurnPINNConfig {
        hidden_layers: vec![],
        ..Default::default()
    };
    let trainer = BurnPINNTrainer::<TestBackend>::new(config, &device);
    assert!(trainer.is_err());
}

#[test]
fn test_train_basic() {
    let device = Default::default();
    let config = BurnPINNConfig {
        hidden_layers: vec![10, 10],
        learning_rate: 0.01,
        num_collocation_points: 100,
        ..Default::default()
    };
    let mut trainer = BurnPINNTrainer::<TestBackend>::new(config, &device).unwrap();
    let n = 20;
    let x_data = Array1::linspace(-1.0, 1.0, n);
    let t_data = Array1::linspace(0.0, 0.1, n);
    let u_data = Array2::zeros((n, 1));
    let result = trainer.train(&x_data, &t_data, &u_data, SOUND_SPEED_AIR, &device, 10);
    let metrics = result.unwrap();
    assert_eq!(metrics.epochs_completed, 10);
    assert_eq!(metrics.total_loss.len(), 10);
    assert!(metrics.training_time_secs > 0.0);
}

#[test]
fn test_train_mismatched_dimensions() {
    let device = Default::default();
    let config = BurnPINNConfig {
        hidden_layers: vec![10, 10],
        ..Default::default()
    };
    let mut trainer = BurnPINNTrainer::<TestBackend>::new(config, &device).unwrap();
    let x_data = Array1::linspace(-1.0, 1.0, 20);
    let t_data = Array1::linspace(0.0, 0.1, 30);
    let u_data = Array2::zeros((20, 1));
    let result = trainer.train(&x_data, &t_data, &u_data, SOUND_SPEED_AIR, &device, 10);
    assert!(result.is_err());
}

#[test]
fn test_train_invalid_u_shape() {
    let device = Default::default();
    let config = BurnPINNConfig {
        hidden_layers: vec![10, 10],
        ..Default::default()
    };
    let mut trainer = BurnPINNTrainer::<TestBackend>::new(config, &device).unwrap();
    let n = 20;
    let x_data = Array1::linspace(-1.0, 1.0, n);
    let t_data = Array1::linspace(0.0, 0.1, n);
    let u_data = Array2::zeros((n, 2));
    let result = trainer.train(&x_data, &t_data, &u_data, SOUND_SPEED_AIR, &device, 10);
    assert!(result.is_err());
}

#[test]
fn test_train_metrics_recording() {
    let device = Default::default();
    let config = BurnPINNConfig {
        hidden_layers: vec![5, 5],
        learning_rate: 0.01,
        num_collocation_points: 100,
        ..Default::default()
    };
    let mut trainer = BurnPINNTrainer::<TestBackend>::new(config, &device).unwrap();
    let n = 10;
    let x_data = Array1::linspace(-1.0, 1.0, n);
    let t_data = Array1::linspace(0.0, 0.1, n);
    let u_data = Array2::zeros((n, 1));
    let metrics = trainer
        .train(&x_data, &t_data, &u_data, SOUND_SPEED_AIR, &device, 5)
        .unwrap();
    assert_eq!(metrics.total_loss.len(), 5);
    assert_eq!(metrics.data_loss.len(), 5);
    assert_eq!(metrics.pde_loss.len(), 5);
    assert_eq!(metrics.bc_loss.len(), 5);
    for &loss in &metrics.total_loss {
        assert!(loss.is_finite());
    }
    for &loss in &metrics.data_loss {
        assert!(loss.is_finite());
    }
    for &loss in &metrics.pde_loss {
        assert!(loss.is_finite());
    }
    for &loss in &metrics.bc_loss {
        assert!(loss.is_finite());
    }
}

#[test]
fn test_pinn_accessor() {
    let device = Default::default();
    let config = BurnPINNConfig {
        hidden_layers: vec![10, 10],
        ..Default::default()
    };
    let trainer = BurnPINNTrainer::<TestBackend>::new(config, &device).unwrap();
    let _pinn = trainer.pinn();
}

#[test]
fn test_optimizer_accessor() {
    let device = Default::default();
    let config = BurnPINNConfig {
        hidden_layers: vec![10, 10],
        learning_rate: 0.001,
        ..Default::default()
    };
    let trainer = BurnPINNTrainer::<TestBackend>::new(config.clone(), &device).unwrap();
    let optimizer = trainer.optimizer();
    assert_eq!(optimizer.learning_rate(), config.learning_rate as f32);
}

#[test]
fn test_config_accessor() {
    let device = Default::default();
    let config = BurnPINNConfig {
        hidden_layers: vec![10, 10],
        learning_rate: 0.001,
        num_collocation_points: 5000,
        ..Default::default()
    };
    let trainer = BurnPINNTrainer::<TestBackend>::new(config.clone(), &device).unwrap();
    let trainer_config = trainer.config();
    assert_eq!(trainer_config.hidden_layers, config.hidden_layers);
    assert_eq!(trainer_config.learning_rate, config.learning_rate);
    assert_eq!(
        trainer_config.num_collocation_points,
        config.num_collocation_points
    );
}

#[test]
fn test_multiple_training_runs() {
    let device = Default::default();
    let config = BurnPINNConfig {
        hidden_layers: vec![5, 5],
        learning_rate: 0.01,
        ..Default::default()
    };
    let mut trainer = BurnPINNTrainer::<TestBackend>::new(config, &device).unwrap();
    let n = 10;
    let x_data = Array1::linspace(-1.0, 1.0, n);
    let t_data = Array1::linspace(0.0, 0.1, n);
    let u_data = Array2::zeros((n, 1));
    let metrics1 = trainer
        .train(&x_data, &t_data, &u_data, SOUND_SPEED_AIR, &device, 5)
        .unwrap();
    assert_eq!(metrics1.epochs_completed, 5);
    let metrics2 = trainer
        .train(&x_data, &t_data, &u_data, SOUND_SPEED_AIR, &device, 5)
        .unwrap();
    assert_eq!(metrics2.epochs_completed, 5);
}
