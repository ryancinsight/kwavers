# Example: Real-Time 3D Beamforming

**Crate**: `kwavers`  
**Run**: `cargo run -p kwavers --example real_time_3d_beamforming --features gpu`  
**Source**: [`crates/kwavers/examples/real_time_3d_beamforming.rs`](../../../../crates/kwavers/examples/real_time_3d_beamforming.rs)

## What This Example Demonstrates

This example demonstrates GPU-accelerated volumetric beamforming for real-time 4-D ultrasound. It covers streaming data handling, dynamic focusing, and CPU-vs-GPU benchmarking around the 3-D delay-and-sum processor.

| Component | API | Value |
|---|---|---|
| 3-D config | `BeamformingConfig3D` | Uses a 64×64×64 reconstruction volume and a 16×16×8 element grid in the demo setup |
| GPU backend | `WgpuBeamformingProvider` | Supplies the accelerator used by `BeamformingProcessor3D` |
| Feature gate | `--features gpu` | Required because the example intentionally exercises the GPU path |

## Key Code Snippet

```rust
let config = BeamformingConfig3D {
    volume_dims: (64, 64, 64),    // Smaller for demo
    num_elements_3d: (16, 16, 8), // 2,048 elements
    ..Default::default()
};

println!("Configuration:");
println!(
    "  Volume: {}×{}×{}",
    config.volume_dims.0, config.volume_dims.1, config.volume_dims.2
```

## Expected Output (if applicable)

The executable prints separate sections for delay-and-sum, dynamic focusing, streaming, and benchmarking; without GPU support it prints the required run command.

## Book Chapter

[← Transducer Arrays and Beamforming](../beamforming_and_image_formation.md)
