# Example: PINN 2D Wave Equation

**Crate**: `kwavers`  
**Run**: `cargo run -p kwavers --example pinn_2d_wave_equation --features pinn`  
**Source**: [`crates/kwavers/examples/pinn_2d_wave_equation.rs`](../../../../crates/kwavers/examples/pinn_2d_wave_equation.rs)

## What This Example Demonstrates

This example solves the 2-D acoustic wave equation with a physics-informed neural network and checks the learned field against a closed-form sinusoidal solution. It highlights rectangular-domain geometry, PDE residual losses, and analytical benchmarking.

| Component | API | Value |
|---|---|---|
| Reference PDE | `∂²u/∂t² = c²(∂²u/∂x² + ∂²u/∂y²)` | The network is trained against the residual of the 2-D wave equation |
| Geometry | `WaveGeometry2D` | Represents the rectangular spatial domain used by the PINN |
| Training stack | `PinnConfig2D` + `PinnTrainer2D` | Enabled through the `pinn` feature for physics-informed training |

## Key Code Snippet

```rust
fn analytical_solution_2d(x: f64, y: f64, t: f64, wave_speed: f64) -> f64 {
    let k = std::f64::consts::PI * 2.0_f64.sqrt();
    (x * std::f64::consts::PI).sin() * (y * std::f64::consts::PI).sin() * (k * wave_speed * t).cos()
}

#[cfg(feature = "pinn")]
/// Generate training data from analytical solution
fn generate_training_data(
    n_samples: usize,
    domain_size: f64,
```

## Expected Output (if applicable)

With `--features pinn`, the program prints training and error information; without it, the example exits with a reminder to enable the feature.

## Book Chapter

[← Inverse Problems and Physics-Informed Neural Networks](../inverse_problems_and_pinns.md)
