use super::*;

#[test]
fn test_transfer_learning_config() {
    let config = TransferLearningConfig {
        fine_tune_lr: 0.001,
        fine_tune_epochs: 50,
        freeze_strategy: FreezeStrategy::ProgressiveUnfreeze,
        adaptation_strength: 0.1,
        patience: 10,
        wave_speed: 1500.0,
    };

    assert_eq!(config.fine_tune_epochs, 50);
    assert_eq!(config.patience, 10);
}

#[test]
fn test_freeze_strategies() {
    let strategies = vec![
        FreezeStrategy::FullFineTune,
        FreezeStrategy::ProgressiveUnfreeze,
        FreezeStrategy::FreezeAllButLast,
        FreezeStrategy::FreezeFirstNLayers(2),
    ];

    for strategy in strategies {
        if let FreezeStrategy::FreezeFirstNLayers(n) = strategy {
            assert_eq!(n, 2);
        }
    }
}

#[test]
fn test_transfer_metrics() {
    let metrics = TransferMetrics {
        initial_accuracy: 0.6,
        final_accuracy: 0.85,
        transfer_efficiency: 0.025,
        training_time: std::time::Duration::from_secs(30),
        convergence_epochs: 10,
    };

    assert!(metrics.final_accuracy > metrics.initial_accuracy);
    assert!(metrics.transfer_efficiency > 0.0);
}
