use super::config::PhysicsNNTrainingConfig;
use super::dataset::{EpochTrainingMetrics, TrainingDataset};
use super::history::TrainingHistory;
use super::loss::{Optimizer, PhysicsLoss};
use crate::core::constants::numerical::TWO_PI;
use ndarray::Array2;

#[test]
fn test_training_config_default() {
    // default: num_epochs=100, batch_size=32, learning_rate=0.001, lambda_data+lambda_physics=1.0
    let config = PhysicsNNTrainingConfig::default();
    config.validate().unwrap();
    assert_eq!(config.num_epochs, 100, "default num_epochs must be 100");
    assert_eq!(config.batch_size, 32, "default batch_size must be 32");
    let lambda_sum = config.lambda_data + config.lambda_physics;
    assert!(
        (lambda_sum - 1.0).abs() < 1e-6,
        "lambda_data + lambda_physics = {lambda_sum} (must equal 1.0)"
    );
}

#[test]
fn test_training_config_validation() {
    let mut config = PhysicsNNTrainingConfig::default();
    config.num_epochs = 0;
    let err = config.validate().unwrap_err();
    assert!(
        format!("{err:?}").contains("num_epochs"),
        "zero num_epochs error must mention 'num_epochs'; got: {err:?}"
    );

    config.num_epochs = 100;
    config.learning_rate = 0.0;
    let err = config.validate().unwrap_err();
    assert!(
        format!("{err:?}").contains("learning_rate"),
        "zero learning_rate error must mention 'learning_rate'; got: {err:?}"
    );

    config.learning_rate = 0.001;
    config.lambda_data = 0.6;
    config.lambda_physics = 0.3; // sum = 0.9 ≠ 1.0
    let err = config.validate().unwrap_err();
    assert!(
        format!("{err:?}").contains("lambda"),
        "lambda-sum error must mention 'lambda'; got: {err:?}"
    );
}

#[test]
fn test_training_config_builder() {
    let config = PhysicsNNTrainingConfig::default()
        .with_epochs(200)
        .with_batch_size(64)
        .with_learning_rate(0.0001);

    assert_eq!(config.num_epochs, 200);
    assert_eq!(config.batch_size, 64);
    assert!(
        (config.learning_rate - 0.0001).abs() < 1e-12,
        "learning_rate = {} (expected 0.0001)",
        config.learning_rate
    );
}

#[test]
fn test_training_dataset_creation() {
    let inputs = Array2::<f64>::zeros((100, 10));
    let targets = Array2::<f64>::zeros((100, 1));
    let dataset = TrainingDataset::new(inputs, targets).unwrap();
    assert_eq!(dataset.len(), 100, "dataset must hold 100 samples");
}

#[test]
fn test_training_dataset_mismatched_sizes() {
    let inputs = Array2::<f64>::zeros((100, 10));
    let targets = Array2::<f64>::zeros((50, 1));
    let err = TrainingDataset::new(inputs, targets).unwrap_err();
    assert!(
        format!("{err:?}").contains("same number"),
        "mismatch error must mention 'same number'; got: {err:?}"
    );
}

#[test]
fn test_training_dataset_split() {
    let inputs = Array2::<f64>::zeros((100, 10));
    let targets = Array2::<f64>::zeros((100, 1));
    let dataset = TrainingDataset::new(inputs, targets).unwrap();

    let (train, val) = dataset.split(0.2).unwrap();
    assert_eq!(train.len() + val.len(), 100);
    assert!(train.len() > val.len());
}

#[test]
fn test_training_dataset_batch() {
    let inputs = Array2::<f64>::zeros((100, 10));
    let targets = Array2::<f64>::zeros((100, 1));
    let dataset = TrainingDataset::new(inputs, targets).unwrap();

    let batch = dataset.batch(0, 32).unwrap();
    assert_eq!(batch.len(), 32);

    let batch2 = dataset.batch(32, 32).unwrap();
    assert_eq!(batch2.len(), 32);
}

#[test]
fn test_physics_loss_reciprocity() {
    let forward = Array2::<f64>::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let reverse = forward.clone();
    let loss = PhysicsLoss::reciprocity_violation(&forward, &reverse);
    assert!((loss - 0.0).abs() < 1e-10);
}

#[test]
fn test_optimizer_default() {
    let optimizer = Optimizer::default();
    if let Optimizer::Adam { beta1, beta2, .. } = optimizer {
        assert!((beta1 - 0.9).abs() < 1e-10);
        assert!((beta2 - 0.999).abs() < 1e-10);
    } else {
        panic!("Default should be Adam");
    }
}

#[test]
fn test_training_history() {
    let mut history = TrainingHistory::new();
    assert_eq!(history.epochs.len(), 0);
    assert!(history.best_val_loss.is_infinite());

    let metrics = EpochTrainingMetrics {
        epoch: 0,
        train_loss: 1.0,
        train_data_loss: 0.8,
        train_physics_loss: 0.2,
        val_loss: 0.9,
        learning_rate: 0.001,
        gradient_norm: 0.1,
        time_per_epoch: 1.5,
    };

    history.add_epoch(metrics);
    assert_eq!(history.epochs.len(), 1);
    assert!((history.best_val_loss - 0.9).abs() < 1e-10);
}

#[test]
fn test_loss_weights_normalization() {
    let config = PhysicsNNTrainingConfig::default().with_loss_weights(2.0, 1.0);

    assert!((config.lambda_data - 2.0 / 3.0).abs() < 1e-10);
    assert!((config.lambda_physics - 1.0 / 3.0).abs() < 1e-10);
}

// ─── PhysicsLoss exact value-semantic tests ───────────────────────────────────

/// `reciprocity_violation` with a known non-zero difference is MSE = ||Δ||² / N.
///
/// forward = [[2, 0], [0, 0]], reverse = [[0, 0], [0, 0]]
/// diff = [[2, 0], [0, 0]] → sum_sq = 4.0, N = 4 → result = 1.0
#[test]
fn physics_loss_reciprocity_violation_exact_mse() {
    let forward = Array2::<f64>::from_shape_vec((2, 2), vec![2.0, 0.0, 0.0, 0.0]).unwrap();
    let reverse = Array2::<f64>::zeros((2, 2));
    let loss = PhysicsLoss::reciprocity_violation(&forward, &reverse);
    assert!(
        (loss - 1.0).abs() < 1e-14,
        "reciprocity_violation = {loss} (expected 1.0 = 4/4)"
    );
}

/// `reciprocity_violation` with mismatched dims returns infinity.
///
/// forward: (2,2), reverse: (2,3) → dims differ → f64::INFINITY.
#[test]
fn physics_loss_reciprocity_violation_mismatched_dims_is_infinity() {
    let forward = Array2::<f64>::zeros((2, 2));
    let reverse = Array2::<f64>::zeros((2, 3));
    let loss = PhysicsLoss::reciprocity_violation(&forward, &reverse);
    assert!(
        loss.is_infinite(),
        "mismatched-dim reciprocity_violation must be ∞, got {loss}"
    );
}

/// `coherence_violation` with a uniform row-jump of 1.0 radian on 2×2 grid.
///
/// phases = [[0, 0], [1, 1]]: two row-pairs each with phase_diff = 1.0.
/// violation = 2 × 1.0 = 2.0; N = phases.len() = 4 → result = 0.5.
#[test]
fn physics_loss_coherence_violation_row_jump_exact() {
    let phases = Array2::<f64>::from_shape_vec((2, 2), vec![0.0, 0.0, 1.0, 1.0]).unwrap();
    let loss = PhysicsLoss::coherence_violation(&phases);
    assert!(
        (loss - 0.5).abs() < 1e-14,
        "coherence_violation = {loss} (expected 0.5)"
    );
}

/// `coherence_violation` wraps phase_diff > π via 2π − phase_diff.
///
/// phases = [[0.0], [4.0]]: diff = 4.0 > π → normalized = 2π − 4.0 ≈ 2.2832.
/// violation = 2π − 4.0; N = 2 → result = (2π − 4.0) / 2.
#[test]
fn physics_loss_coherence_violation_wrap_above_pi_exact() {
    let phases = Array2::<f64>::from_shape_vec((2, 1), vec![0.0, 4.0]).unwrap();
    let loss = PhysicsLoss::coherence_violation(&phases);
    let expected = (TWO_PI - 4.0) / 2.0;
    assert!(
        (loss - expected).abs() < 1e-14,
        "wrap coherence_violation = {loss} (expected {expected})"
    );
}

/// `sparsity_violation` is mean absolute value = L1-norm / N.
///
/// weights = [[1, -2], [3, -4]]: L1 = 1+2+3+4 = 10; N = 4 → result = 2.5.
#[test]
fn physics_loss_sparsity_violation_exact_l1_mean() {
    let weights = Array2::<f64>::from_shape_vec((2, 2), vec![1.0, -2.0, 3.0, -4.0]).unwrap();
    let loss = PhysicsLoss::sparsity_violation(&weights);
    assert!(
        (loss - 2.5).abs() < 1e-14,
        "sparsity_violation = {loss} (expected 2.5 = 10/4)"
    );
}

/// `sparsity_violation` of an all-zero matrix is zero.
///
/// L1 = 0 → result = 0.
#[test]
fn physics_loss_sparsity_violation_zero_weights_is_zero() {
    let weights = Array2::<f64>::zeros((4, 4));
    let loss = PhysicsLoss::sparsity_violation(&weights);
    assert!(
        loss.abs() < 1e-14,
        "zero-weight sparsity_violation must be 0.0, got {loss}"
    );
}
