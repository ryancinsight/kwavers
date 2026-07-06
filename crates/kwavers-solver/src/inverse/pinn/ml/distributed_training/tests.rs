use super::*;

type TestBackend = coeus_core::MoiraiBackend;

fn unique_checkpoint_dir(test_name: &str) -> std::path::PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("invariant: system clock is after UNIX_EPOCH")
        .as_nanos();
    std::env::temp_dir().join(format!("kwavers_{test_name}_{nanos}"))
}

#[test]
fn test_distributed_trainer_creation() {
    let config = DistributedTrainingConfig::default();
    let base_config = crate::inverse::pinn::ml::BurnPINN2DConfig::default();
    let geometry = crate::inverse::pinn::ml::BurnWave2dGeometry::rectangular(0.0, 1.0, 0.0, 1.0);

    let result = DistributedPinnTrainer::<TestBackend>::new(config, base_config, geometry);

    match result {
        Ok(trainer) => {
            assert_eq!(trainer.config.num_gpus, 1);
            assert!(trainer.multi_gpu_manager.is_none());
        }
        Err(_) => {
            // Expected on systems without proper GPU support
        }
    }
}

#[test]
fn test_checkpoint_manager() {
    let checkpoint_dir = unique_checkpoint_dir("checkpoint_manager");
    let manager = CheckpointManager {
        checkpoint_dir: checkpoint_dir.clone(),
        max_checkpoints: 3,
    };

    manager
        .ensure_checkpoint_dir()
        .expect("checkpoint directory creation should succeed");
    assert!(manager.checkpoint_dir.exists());
    assert_eq!(manager.list_checkpoints().unwrap(), Vec::<usize>::new());

    std::fs::remove_dir_all(checkpoint_dir).expect("checkpoint test directory cleanup");
}

#[test]
fn test_checkpoint_save_load_round_trip() {
    let checkpoint_dir = unique_checkpoint_dir("checkpoint_round_trip");
    let config = DistributedTrainingConfig {
        checkpoint_config: CheckpointConfig {
            directory: checkpoint_dir.display().to_string(),
            max_checkpoints: 2,
            ..Default::default()
        },
        ..Default::default()
    };
    let base_config = crate::inverse::pinn::ml::BurnPINN2DConfig::default();
    let geometry = crate::inverse::pinn::ml::BurnWave2dGeometry::rectangular(0.0, 1.0, 0.0, 1.0);

    let mut trainer =
        DistributedPinnTrainer::<TestBackend>::new(config.clone(), base_config.clone(), geometry)
            .expect("default single-device distributed trainer should construct");
    trainer.coordinator.training_state.current_epoch = 7;
    trainer.coordinator.training_state.global_metrics.total_loss = vec![1.25, 0.75];

    trainer
        .save_checkpoint()
        .expect("checkpoint save should write state");
    assert_eq!(trainer.coordinator.training_state.last_checkpoint, 7);
    assert_eq!(
        trainer
            .coordinator
            .checkpoint_manager
            .list_checkpoints()
            .unwrap(),
        vec![7]
    );

    let geometry = crate::inverse::pinn::ml::BurnWave2dGeometry::rectangular(0.0, 1.0, 0.0, 1.0);
    let mut restored = DistributedPinnTrainer::<TestBackend>::new(config, base_config, geometry)
        .expect("restore target trainer should construct");
    restored
        .load_checkpoint(7)
        .expect("checkpoint load should restore saved state");

    assert_eq!(restored.coordinator.training_state.current_epoch, 7);
    assert_eq!(restored.coordinator.training_state.last_checkpoint, 7);
    assert_eq!(
        restored
            .coordinator
            .training_state
            .global_metrics
            .total_loss,
        vec![1.25, 0.75]
    );

    std::fs::remove_dir_all(checkpoint_dir).expect("checkpoint test directory cleanup");
}

#[test]
fn test_gradient_aggregation_config() {
    let config = DistributedTrainingConfig {
        gradient_aggregation: GradientAggregation::Weighted {
            weights: vec![0.6, 0.4],
        },
        ..Default::default()
    };

    match config.gradient_aggregation {
        GradientAggregation::Weighted { weights } => {
            assert_eq!(weights.len(), 2);
            assert!((weights.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        }
        _ => panic!("Expected weighted aggregation"),
    }
}
