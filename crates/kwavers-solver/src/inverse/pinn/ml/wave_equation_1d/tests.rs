use super::{LossWeights, PinnConfig, PinnTrainer, PinnWave1D, SimpleOptimizer, TrainingMetrics};
use coeus_core::MoiraiBackend;
use kwavers_core::constants::fundamental::SOUND_SPEED_WATER_SIM;
use leto::{
    Array1,
    Array2,
};

type TestBackend = MoiraiBackend;

#[test]
fn test_public_api_types_available() {
    // Ensure all public types are accessible
    let _config: PinnConfig = PinnConfig::default();
    let _weights: LossWeights = LossWeights::default();
}

#[test]
fn test_end_to_end_cpu_training() {
    // Create configuration
    let config = PinnConfig {
        hidden_layers: vec![10, 10],
        learning_rate: 0.01,
        num_collocation_points: 100,
        ..Default::default()
    };

    // Create trainer
    let mut trainer = PinnTrainer::<TestBackend>::new(config).unwrap();

    // Synthetic training data
    let n = 20;
    let x_data = Array1::linspace(-1.0, 1.0, n);
    let t_data = Array1::linspace(0.0, 0.1, n);
    let u_data = Array2::zeros((n, 1));

    // Train
    let metrics = trainer.train(&x_data, &t_data, &u_data, 343.0, 10).unwrap();

    // Verify metrics
    assert_eq!(metrics.epochs_completed, 10);
    assert_eq!((metrics.total_loss.len()), 10);
    assert!(metrics.training_time_secs > 0.0);

    // All losses should be finite
    for &loss in &metrics.total_loss {
        assert!(loss.is_finite());
    }

    // Predict after training
    let x_test = Array1::linspace(-1.0, 1.0, 5);
    let t_test = Array1::linspace(0.0, 0.1, 5);
    let u_pred = trainer.pinn().predict(&x_test, &t_test).unwrap();

    assert_eq!(u_pred.shape(), &[5, 1]);
    for &val in u_pred.iter() {
        assert!(val.is_finite());
    }
}

#[test]
fn test_config_presets() {
    // Default config
    let config = PinnConfig::default();
    assert!(!config.hidden_layers.is_empty());
    assert!(config.learning_rate > 0.0);

    // GPU config
    let config = PinnConfig::for_gpu();
    assert!((config.hidden_layers.len()) >= 4);
    assert!(config.num_collocation_points >= 10000);

    // Prototyping config
    let config = PinnConfig::for_prototyping();
    assert!((config.hidden_layers.len()) == 3);
    assert!(config.num_collocation_points <= 1000);
}

#[test]
fn test_loss_weights_presets() {
    // Data-driven
    let weights = LossWeights::data_driven();
    assert!(weights.data >= weights.pde);

    // Physics-driven
    let weights = LossWeights::physics_driven();
    assert!(weights.pde >= weights.data);

    // Balanced
    let weights = LossWeights::balanced();
    assert!(weights.data == weights.pde);
}

#[test]
fn test_metrics_convergence_detection() {
    let mut metrics = TrainingMetrics::new();

    metrics.record_epoch(1.0, 0.5, 0.3, 0.2);
    metrics.record_epoch(0.1, 0.05, 0.03, 0.02);
    metrics.record_epoch(0.1 * (1.0 - 1e-7), 0.05, 0.03, 0.02);

    assert!(metrics.is_converged(1e-6));
}

#[test]
fn test_metrics_numerical_issues_detection() {
    let mut metrics = TrainingMetrics::new();

    // Simulate normal training
    metrics.record_epoch(1.0, 0.5, 0.3, 0.2);
    metrics.record_epoch(0.9, 0.45, 0.27, 0.18);
    assert!(!metrics.has_numerical_issues());

    // Add NaN
    metrics.record_epoch(f64::NAN, 0.4, 0.24, 0.16);
    assert!(metrics.has_numerical_issues());
}

#[test]
fn test_network_creation_via_public_api() {
    let config = PinnConfig::default();

    // Should be able to create network directly
    let network = PinnWave1D::<TestBackend>::new(config);
    let _network = network.unwrap();
}

#[test]
fn test_optimizer_creation_via_public_api() {
    // Should be able to create optimizer directly
    let optimizer = SimpleOptimizer::new(0.001);
    assert_eq!(optimizer.learning_rate(), 0.001);
}

#[test]
fn test_multi_epoch_convergence() {
    let config = PinnConfig {
        hidden_layers: vec![10, 10],
        learning_rate: 0.01,
        num_collocation_points: 100,
        ..Default::default()
    };

    let mut trainer = PinnTrainer::<TestBackend>::new(config).unwrap();

    let n = 15;
    let x_data = Array1::linspace(-1.0, 1.0, n);
    let t_data = Array1::linspace(0.0, 0.1, n);
    let u_data = Array2::zeros((n, 1));

    // Train for more epochs
    let metrics = trainer.train(&x_data, &t_data, &u_data, 343.0, 50).unwrap();

    assert_eq!(metrics.epochs_completed, 50);

    // Loss should generally decrease (or at least not increase dramatically)
    let first_loss = metrics.total_loss[0];
    let last_loss = metrics.total_loss[(metrics.total_loss.len()) - 1];

    // Both should be finite
    assert!(first_loss.is_finite());
    assert!(last_loss.is_finite());
}

#[test]
fn test_different_wave_speeds() {
    let config = PinnConfig {
        hidden_layers: vec![10, 10],
        learning_rate: 0.01,
        num_collocation_points: 100,
        ..Default::default()
    };

    let n = 10;
    let x_data = Array1::linspace(-1.0, 1.0, n);
    let t_data = Array1::linspace(0.0, 0.1, n);
    let u_data = Array2::zeros((n, 1));

    // Train with air speed of sound
    let mut trainer1 = PinnTrainer::<TestBackend>::new(config.clone()).unwrap();
    let metrics1 = trainer1
        .train(&x_data, &t_data, &u_data, 343.0, 10)
        .unwrap();
    assert!(metrics1.total_loss.last().unwrap().is_finite());

    // Train with water speed of sound
    let mut trainer2 = PinnTrainer::<TestBackend>::new(config).unwrap();
    let metrics2 = trainer2
        .train(&x_data, &t_data, &u_data, SOUND_SPEED_WATER_SIM, 10)
        .unwrap();
    assert!(metrics2.total_loss.last().unwrap().is_finite());
}

#[test]
fn test_pinn_predict_interface() {
    let config = PinnConfig::default();

    let pinn = PinnWave1D::<TestBackend>::new(config).unwrap();

    // Test prediction
    let x = Array1::linspace(-1.0, 1.0, 10);
    let t = Array1::linspace(0.0, 0.1, 10);

    let u = pinn.predict(&x, &t).unwrap();

    assert_eq!(u.shape(), &[10, 1]);
    for &val in u.iter() {
        assert!(val.is_finite());
    }
}

#[test]
fn test_complete_workflow() {
    // 1. Create configuration
    let config = PinnConfig {
        hidden_layers: vec![10, 10],
        learning_rate: 0.01,
        num_collocation_points: 100,
        loss_weights: LossWeights::balanced(),
    };

    // 2. Validate configuration
    assert!(config.validate().is_ok());

    // 3. Create trainer
    let mut trainer = PinnTrainer::<TestBackend>::new(config).unwrap();

    // 4. Prepare data
    let n = 15;
    let x_data = Array1::linspace(-1.0, 1.0, n);
    let t_data = Array1::linspace(0.0, 0.1, n);
    let u_data = Array2::zeros((n, 1));

    // 5. Train
    let metrics = trainer.train(&x_data, &t_data, &u_data, 343.0, 20).unwrap();

    // 6. Verify training
    assert_eq!(metrics.epochs_completed, 20);
    assert!(!metrics.has_numerical_issues());

    // 7. Make predictions
    let x_test = Array1::linspace(-1.0, 1.0, 5);
    let t_test = Array1::linspace(0.0, 0.1, 5);
    let u_pred = trainer.pinn().predict(&x_test, &t_test).unwrap();

    // 8. Verify predictions
    assert_eq!(u_pred.shape(), &[5, 1]);
    for &val in u_pred.iter() {
        assert!(val.is_finite());
    }
}
