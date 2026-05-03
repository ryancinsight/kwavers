use super::*;
use ndarray::Array2;

#[test]
fn test_trainer_creation() {
    let config = TrainingConfig::default();
    let physics_loss = PhysicsLoss::default();
    let trainer = BeamformingTrainer::new(config, physics_loss);
    assert!(trainer.is_ok());
}

#[test]
fn test_trainer_empty_dataset() {
    let config = TrainingConfig::default();
    let physics_loss = PhysicsLoss::default();
    let mut trainer = BeamformingTrainer::new(config, physics_loss).unwrap();

    let inputs = Array2::<f64>::zeros((0, 10));
    let targets = Array2::<f64>::zeros((0, 1));
    let empty_dataset = TrainingDataset::new(inputs, targets).unwrap();

    let result = trainer.train(&empty_dataset, None);
    assert!(result.is_err());
}

#[test]
fn test_trainer_simple_training() {
    let mut config = TrainingConfig::default();
    config.num_epochs = 5;
    config.batch_size = 10;
    config.verbose = false;

    let physics_loss = PhysicsLoss::default();
    let mut trainer = BeamformingTrainer::new(config, physics_loss).unwrap();

    let inputs = Array2::<f64>::zeros((100, 10));
    let targets = Array2::<f64>::ones((100, 1)) * 0.5;
    let dataset = TrainingDataset::new(inputs, targets).unwrap();

    let history = trainer.train(&dataset, None);
    assert!(history.is_ok());

    let h = history.unwrap();
    assert_eq!(h.epochs.len(), 5);
    assert!(h.best_val_loss.is_finite());
}

#[test]
fn test_trainer_with_validation_dataset() {
    let mut config = TrainingConfig::default();
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

    let history = trainer.train(&train_dataset, Some(&val_dataset));
    assert!(history.is_ok());

    let h = history.unwrap();
    assert_eq!(h.epochs.len(), 3);
}

#[test]
fn test_trainer_config_access() {
    let config = TrainingConfig::default().with_epochs(50);
    let physics_loss = PhysicsLoss::default();
    let trainer = BeamformingTrainer::new(config, physics_loss).unwrap();

    assert_eq!(trainer.config().num_epochs, 50);
    assert_eq!(trainer.physics_loss().reciprocity_weight, 0.5);
}
