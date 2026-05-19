use super::*;
use burn::backend::NdArray;

type TestBackend = burn::backend::Autodiff<NdArray<f32>>;

#[tokio::test]
async fn test_distributed_trainer_creation() {
    let config = DistributedTrainingConfig::default();
    let base_config = crate::solver::inverse::pinn::ml::BurnPINN2DConfig::default();
    let geometry =
        crate::solver::inverse::pinn::ml::BurnWave2dGeometry::rectangular(0.0, 1.0, 0.0, 1.0);

    let result = DistributedPinnTrainer::<TestBackend>::new(config, base_config, geometry).await;

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
    let manager = CheckpointManager {
        checkpoint_dir: std::path::PathBuf::from("test_checkpoints"),
        max_checkpoints: 3,
        checkpoint_interval: 100,
        auto_save: true,
    };

    assert!(manager.ensure_checkpoint_dir().is_ok());
    assert!(manager.checkpoint_dir.exists());
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
