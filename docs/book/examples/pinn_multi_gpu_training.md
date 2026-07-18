# Example: PINN Multi-GPU Training

**Crate**: `kwavers`  
**Run**: `cargo run -p kwavers --example pinn_multi_gpu_training --features pinn,gpu`  
**Source**: [`crates/kwavers/examples/pinn_multi_gpu_training.rs`](../../../../crates/kwavers/examples/pinn_multi_gpu_training.rs)

## What This Example Demonstrates

This example demonstrates multi-GPU Physics-Informed Neural Network (PINN) training. It shows how to distribute training across multiple GPUs for improved performance on large-scale problems.

## Multi-GPU Strategy

| Aspect | Description |
|--------|-------------|
| Data Parallelism | Split batch across GPUs |
| Model Parallelism | Split network across GPUs |
| Communication | NCCL for gradient synchronization |
| Scaling | Weak scaling efficiency |

## Key Code Snippet

```rust
// Create multiple GPU backends
let backends: Vec<_> = (0..num_gpus)
    .map(|i| MoiraiBackend::gpu_with_device(i))
    .collect()?;

// Create distributed PINN solver
let pinn = DistributedPINNSolver::new(
    domain,
    geometry,
    medium,
    config,
    backends,
)?;

// Train across GPUs
for epoch in 0..max_epochs {
    let loss = pinn.train_step()?;
    
    if epoch % 100 == 0 {
        println!("Epoch {}: loss = {:.6e} (Multi-GPU)", epoch, loss);
    }
}

// Report scaling efficiency
let efficiency = compute_scaling_efficiency()?;
println!("Multi-GPU efficiency: {:.1}%", efficiency);
```

## Book Chapter

[← Inverse Problems and Physics-Informed Neural Networks](../inverse_problems_and_pinns.md)
