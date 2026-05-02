use super::*;
use ndarray::{Array2, Array3};

#[test]
fn test_physics_loss_config_default() {
    let config = PhysicsLossConfig::default();
    assert!(config.validate().is_ok());
}

#[test]
fn test_physics_loss_config_validation() {
    let mut config = PhysicsLossConfig::default();
    config.sound_speed = 0.0;
    assert!(config.validate().is_err());

    config.sound_speed = 343.0;
    config.frequency = -100.0;
    assert!(config.validate().is_err());

    config.frequency = 1_000_000.0;
    config.history_window = 0;
    assert!(config.validate().is_err());
}

#[test]
fn test_physics_loss_creation() {
    let config = PhysicsLossConfig::default();
    let loss = PhysicsInformedLoss::new(config);
    assert!(loss.is_ok());
}

#[test]
fn test_wave_equation_residual_2d() {
    let config = PhysicsLossConfig::default();
    let loss = PhysicsInformedLoss::new(config).unwrap();

    let field = Array2::<f64>::zeros((5, 5));
    let residual = loss.wave_equation_residual_2d(&field);
    assert!((residual - 0.0).abs() < 1e-10);
}

#[test]
fn test_wave_equation_residual_3d() {
    let config = PhysicsLossConfig::default();
    let loss = PhysicsInformedLoss::new(config).unwrap();

    let field = Array3::<f64>::zeros((5, 5, 5));
    let residual = loss.wave_equation_residual_3d(&field);
    assert!((residual - 0.0).abs() < 1e-10);
}

#[test]
fn test_wave_number_computation() {
    let config = PhysicsLossConfig::default().with_wave_params(343.0, 1_000_000.0);
    let loss = PhysicsInformedLoss::new(config).unwrap();

    // k = 2πf/c = 2π·1e6/343 ≈ 18313
    let k_expected = 2.0 * std::f64::consts::PI * 1_000_000.0 / 343.0;
    assert!((loss.wave_number() - k_expected).abs() < 1.0);
}

#[test]
fn test_reciprocity_loss() {
    let forward =
        Array2::<f64>::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let reverse = forward.clone();

    let loss = PhysicsInformedLoss::reciprocity_loss(&forward, &reverse);
    assert!((loss - 0.0).abs() < 1e-10);
}

#[test]
fn test_reciprocity_loss_violation() {
    let forward = Array2::<f64>::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let reverse = Array2::<f64>::from_shape_vec((2, 2), vec![1.0, 2.0, 4.0, 5.0]).unwrap();

    let loss = PhysicsInformedLoss::reciprocity_loss(&forward, &reverse);
    assert!(loss > 0.0);
}

#[test]
fn test_coherence_loss_uniform_field() {
    let amplitudes = Array2::<f64>::from_elem((5, 5), 1.0);
    let phases = Array2::<f64>::from_elem((5, 5), 0.0);

    let loss = PhysicsInformedLoss::coherence_loss(&amplitudes, &phases);
    assert!((loss - 0.0).abs() < 1e-10);
}

#[test]
fn test_weight_schedule_exponential() {
    let config =
        PhysicsLossConfig::default().with_schedule(WeightSchedule::Exponential { decay_rate: 0.1 });
    let mut loss = PhysicsInformedLoss::new(config).unwrap();

    let (_lambda_data1, lambda_physics1) = loss.compute_weight_schedule(1.0, 1.0).unwrap();

    loss.current_epoch = 10;
    let (_lambda_data2, lambda_physics2) = loss.compute_weight_schedule(1.0, 1.0).unwrap();

    assert!(lambda_physics2 < lambda_physics1);
}

#[test]
fn test_weight_schedule_linear() {
    let config =
        PhysicsLossConfig::default().with_schedule(WeightSchedule::Linear { total_epochs: 100 });
    let mut loss = PhysicsInformedLoss::new(config).unwrap();

    let (_lambda_data1, lambda_physics1) = loss.compute_weight_schedule(1.0, 1.0).unwrap();

    loss.current_epoch = 50;
    let (_lambda_data2, lambda_physics2) = loss.compute_weight_schedule(1.0, 1.0).unwrap();

    assert!(lambda_physics2 < lambda_physics1);
}

#[test]
fn test_weight_schedule_adaptive() {
    let config = PhysicsLossConfig::default().with_schedule(WeightSchedule::Adaptive);
    let loss = PhysicsInformedLoss::new(config).unwrap();

    let (_, lambda_physics) = loss.compute_weight_schedule(1.0, 100.0).unwrap();
    assert!(lambda_physics < 0.3);

    let (_, lambda_physics) = loss.compute_weight_schedule(100.0, 1.0).unwrap();
    assert!(lambda_physics > 0.2);
}

#[test]
fn test_total_loss_computation() {
    let config = PhysicsLossConfig::default();
    let mut loss = PhysicsInformedLoss::new(config).unwrap();

    let total = loss.compute_total_loss(1.0, 1.0).unwrap();
    assert!(total > 0.0);
    assert!(total.is_finite());
}

#[test]
fn test_loss_history_tracking() {
    let config = PhysicsLossConfig::default();
    let mut loss = PhysicsInformedLoss::new(config).unwrap();

    loss.compute_total_loss(1.0, 1.0).unwrap();
    loss.compute_total_loss(0.9, 0.8).unwrap();
    loss.compute_total_loss(0.8, 0.7).unwrap();

    let history = loss.loss_history();
    assert_eq!(history.len(), 3);
    assert_eq!(history[0].epoch, 0);
    assert_eq!(history[1].epoch, 1);
    assert_eq!(history[2].epoch, 2);
}

#[test]
fn test_loss_history_window() {
    let config = PhysicsLossConfig::default();
    let mut loss = PhysicsInformedLoss::new(config).unwrap();

    for _ in 0..30 {
        loss.compute_total_loss(1.0, 1.0).unwrap();
    }

    let history = loss.loss_history();
    assert!(history.len() <= 20);
}

#[test]
fn test_reset() {
    let config = PhysicsLossConfig::default();
    let mut loss = PhysicsInformedLoss::new(config).unwrap();

    loss.compute_total_loss(1.0, 1.0).unwrap();
    loss.compute_total_loss(1.0, 1.0).unwrap();
    assert_eq!(loss.current_epoch(), 2);

    loss.reset();
    assert_eq!(loss.current_epoch(), 0);
    assert_eq!(loss.loss_history().len(), 0);
}

#[test]
fn test_builder_pattern() {
    let config = PhysicsLossConfig::default()
        .with_loss_weights(0.7, 0.3)
        .with_wave_params(400.0, 2_000_000.0)
        .with_schedule(WeightSchedule::Linear { total_epochs: 50 })
        .without_history();

    assert!((config.lambda_data_init - 0.7).abs() < 1e-10);
    assert!((config.lambda_physics_init - 0.3).abs() < 1e-10);
    assert!((config.sound_speed - 400.0).abs() < 1e-10);
    assert!((config.frequency - 2_000_000.0).abs() < 1e-10);
    assert!(!config.track_history);
}
