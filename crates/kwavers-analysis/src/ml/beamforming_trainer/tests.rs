use super::*;
use ndarray::Array2;

#[test]
fn test_trainer_creation() {
    // default: num_epochs=100, batch_size=32, learning_rate=0.001
    let config = PhysicsNNTrainingConfig::default();
    let physics_loss = PhysicsLoss::default();
    let trainer = BeamformingTrainer::new(config, physics_loss).unwrap();
    assert_eq!(
        trainer.config().num_epochs,
        100,
        "default num_epochs must be 100"
    );
    assert_eq!(
        trainer.config().batch_size,
        32,
        "default batch_size must be 32"
    );
    assert!(
        (trainer.config().learning_rate - 0.001).abs() < 1e-12,
        "default learning_rate must be 0.001"
    );
}

#[test]
fn test_trainer_empty_dataset() {
    let config = PhysicsNNTrainingConfig::default();
    let physics_loss = PhysicsLoss::default();
    let mut trainer = BeamformingTrainer::new(config, physics_loss).unwrap();

    let inputs = Array2::<f64>::zeros((0, 10));
    let targets = Array2::<f64>::zeros((0, 1));
    let empty_dataset = TrainingDataset::new(inputs, targets).unwrap();

    let err = trainer.train(&empty_dataset, None).unwrap_err();
    let msg = format!("{err:?}");
    assert!(
        msg.contains("empty"),
        "empty-dataset error must mention 'empty'; got: {msg}"
    );
}

#[test]
fn test_trainer_simple_training() {
    let mut config = PhysicsNNTrainingConfig::default();
    config.num_epochs = 5;
    config.batch_size = 10;
    config.verbose = false;

    let physics_loss = PhysicsLoss::default();
    let mut trainer = BeamformingTrainer::new(config, physics_loss).unwrap();

    let inputs = Array2::<f64>::zeros((100, 10));
    let targets = Array2::<f64>::ones((100, 1)) * 0.5;
    let dataset = TrainingDataset::new(inputs, targets).unwrap();

    let h = trainer.train(&dataset, None).unwrap();
    assert_eq!(h.epochs.len(), 5, "must record exactly 5 epoch metrics");
    assert!(
        h.best_val_loss.is_finite(),
        "best_val_loss must be finite after training"
    );
}

#[test]
fn test_trainer_with_validation_dataset() {
    let mut config = PhysicsNNTrainingConfig::default();
    config.num_epochs = 3;
    config.verbose = false;

    let physics_loss = PhysicsLoss::default();
    let mut trainer = BeamformingTrainer::new(config, physics_loss).unwrap();

    let inputs_train = Array2::<f64>::zeros((80, 10));
    let targets_train = Array2::<f64>::ones((80, 1));
    let train_dataset = TrainingDataset::new(inputs_train, targets_train).unwrap();

    let inputs_val = Array2::<f64>::zeros((20, 10));
    let targets_val = Array2::<f64>::ones((20, 1));
    let val_dataset = TrainingDataset::new(inputs_val, targets_val).unwrap();

    let h = trainer.train(&train_dataset, Some(&val_dataset)).unwrap();
    assert_eq!(h.epochs.len(), 3, "must record exactly 3 epoch metrics");
}

#[test]
fn test_trainer_config_access() {
    let config = PhysicsNNTrainingConfig::default().with_epochs(50);
    let physics_loss = PhysicsLoss::default();
    let trainer = BeamformingTrainer::new(config, physics_loss).unwrap();

    assert_eq!(trainer.config().num_epochs, 50);
    assert_eq!(trainer.physics_loss().reciprocity_weight, 0.5);
}
