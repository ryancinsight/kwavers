# Example: PINN Real-Time Inference

**Crate**: `kwavers`  
**Run**: `cargo run -p kwavers --example pinn_real_time_inference --features pinn`  
**Source**: [`crates/kwavers/examples/pinn_real_time_inference.rs`](../../../../crates/kwavers/examples/pinn_real_time_inference.rs)

## What This Example Demonstrates

This example demonstrates real-time Physics-Informed Neural Network (PINN) inference. It shows how to deploy a trained PINN model for fast inference on new input parameters.

## Real-Time Features

| Feature | Description |
|---------|-------------|
| Trained Model | Pre-trained PINN checkpoint |
| Fast Inference | Forward pass in milliseconds |
| Parameter Variation | Real-time parameter sweeps |
| Streaming Output | Continuous result updates |

## Key Code Snippet

```rust
// Load pre-trained PINN
let pinn = load_trained_pinn("checkpoint.bin")?;

// Real-time inference loop
loop {
    let input = get_new_parameters()?;
    let result = pinn.forward(&input)?;
    
    update_display(&result);
    
    if should_stop() {
        break;
    }
}
```

## Book Chapter

[← Inverse Problems and Physics-Informed Neural Networks](../inverse_problems_and_pinns.md)
