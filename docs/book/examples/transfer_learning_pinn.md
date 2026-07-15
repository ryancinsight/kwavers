# Example: Transfer Learning PINN

**Crate**: `kwavers`  
**Run**: `cargo run -p kwavers --example transfer_learning_pinn --features pinn`  
**Source**: [`crates/kwavers/examples/transfer_learning_pinn.rs`](../../../../crates/kwavers/examples/transfer_learning_pinn.rs)

## What This Example Demonstrates

This example shows how a pre-trained 2-D wave PINN can be adapted to a new geometry using fine-tuning and selective layer freezing. It is the shortest example for learning the transfer-learning API in the PINN stack.

| Component | API | Value |
|---|---|---|
| Source model | `PinnWave2D::<Backend>::new(PinnConfig2D::default())` | Starts from a ready-made baseline model configuration |
| Transfer config | `TransferLearningConfig` | Uses 1e-4 fine-tuning LR, 10 epochs, and `FreezeAllButLast` |
| Adaptation call | `TransferLearner::transfer_to_geometry` | Transfers the model to a new rectangular target geometry |

## Key Code Snippet

```rust
let transfer_config = TransferLearningConfig {
    fine_tune_lr: 1e-4,
    fine_tune_epochs: 10,
    freeze_strategy: FreezeStrategy::FreezeAllButLast,
    adaptation_strength: 0.0,
    patience: 3,
    wave_speed: 1500.0,
};

let mut learner = TransferLearner::new(source_model, transfer_config);
```

## Expected Output (if applicable)

With the `pinn` feature enabled, the program prints initial/final accuracy and convergence epochs; otherwise it prints the feature-enablement command.

## Book Chapter

[← Inverse Problems and Physics-Informed Neural Networks](../inverse_problems_and_pinns.md)
