use super::config::TrainingConfig;
use super::dataset::{TrainingDataset, TrainingMetrics};
use super::history::TrainingHistory;
use super::loss::{Optimizer, PhysicsLoss};
use ndarray::Array2;

#[test]
fn test_training_config_default() {
    // default: num_epochs=100, batch_size=32, learning_rate=0.001, lambda_data+lambda_physics=1.0
    let config = TrainingConfig::default();
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
    let mut config = TrainingConfig::default();
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
    let config = TrainingConfig::default()
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

    let metrics = TrainingMetrics {
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
    let config = TrainingConfig::default().with_loss_weights(2.0, 1.0);

    assert!((config.lambda_data - 2.0 / 3.0).abs() < 1e-10);
    assert!((config.lambda_physics - 1.0 / 3.0).abs() < 1e-10);
}
