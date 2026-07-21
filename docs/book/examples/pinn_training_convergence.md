# Example: PINN Training Convergence

**Crate**: `kwavers`  
**Run**: `cargo run -p kwavers --example pinn_training_convergence --features pinn --release`  
**Source**: [`crates/kwavers/examples/pinn_training_convergence.rs`](../../../crates/kwavers/examples/pinn_training_convergence.rs)

## What This Example Demonstrates

This example is a verification harness for PINN training itself. It trains against analytical elastic-wave solutions, checks autodiff gradients against finite differences, and studies how accuracy changes under h-refinement.

| Component | API | Value |
|---|---|---|
| Experiment config | `ExperimentConfig` | Controls grid size, epochs, learning rate, and hidden-layer layout |
| Reference solution | `PlaneWaveAnalytical` | Provides a known solution for convergence and gradient checks |
| Runtime mode | `--features pinn --release` | Uses the PINN backend and the recommended release profile for training |

## Key Code Snippet

```rust
struct ExperimentConfig {
    /// Number of spatial points (N×N grid)
    num_points: usize,
    /// Number of epochs
    epochs: usize,
    /// Learning rate
    learning_rate: f64,
    /// Network hidden layer sizes
    hidden_layers: Vec<usize>,
}
```

## Expected Output (if applicable)

A complete run prints training progress, gradient-check results, and convergence summaries; without the `pinn` feature it prints the required run command.

## Book Chapter

[← Inverse Problems and Physics-Informed Neural Networks](../inverse_problems_and_pinns.md)
