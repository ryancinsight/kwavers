# Example: PINN GPU Training

**Crate**: `kwavers`  
**Run**: `cargo run -p kwavers --example pinn_gpu_training --features pinn,gpu`  
**Source**: [`crates/kwavers/examples/pinn_gpu_training.rs`](../../../crates/kwavers/examples/pinn_gpu_training.rs)

## What This Example Demonstrates

This example demonstrates GPU-accelerated Physics-Informed Neural Network (PINN) training using WGPU. It shows the performance benefits of GPU acceleration for neural network training.

## GPU Acceleration

| Aspect | Description |
|--------|-------------|
| Backend | WGPU (WebGPU API) |
| Device | CUDA or Vulkan GPU |
| Performance | Speedup vs. CPU training |
| Memory | GPU memory management |

## Key Code Snippet

```rust
// Create GPU backend
let backend = MoiraiBackend::gpu()?;

// Create PINN solver with GPU backend
let pinn = UniversalPINNSolver::new(
    domain,
    geometry,
    medium,
    config,
    backend,
)?;

// Train on GPU
for epoch in 0..max_epochs {
    let loss = pinn.train_step()?;
    
    if epoch % 100 == 0 {
        println!("Epoch {}: loss = {:.6e} (GPU)", epoch, loss);
    }
}

// Compare with CPU performance
let cpu_time = benchmark_cpu_training()?;
let gpu_time = benchmark_gpu_training()?;
println!("Speedup: {:.2}x", cpu_time / gpu_time);
```

## Book Chapter

[← Inverse Problems and Physics-Informed Neural Networks](../inverse_problems_and_pinns.md)
