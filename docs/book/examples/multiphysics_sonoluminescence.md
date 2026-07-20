# Example: Multiphysics Sonoluminescence

**Crate**: `kwavers`  
**Run**: `cargo run -p kwavers --example multiphysics_sonoluminescence --features pinn`  
**Source**: [`crates/kwavers/examples/multiphysics_sonoluminescence.rs`](../../../crates/kwavers/examples/multiphysics_sonoluminescence.rs)

## What This Example Demonstrates

This example uses the universal PINN machinery to couple acoustic propagation, cavitation bubble dynamics, sonoluminescence emission, and electromagnetic wave transport. It is the broadest “sound-to-light” demo in the example set.

| Component | API | Value |
|---|---|---|
| Coupled solver | `UniversalPINNSolver::with_cavitation_sonoluminescence_coupling` | Registers all physics domains needed for the chain from ultrasound to light |
| Training plan | `UniversalTrainingConfig` | Runs 500 epochs with 2000 collocation points and adaptive sampling |
| Feature gate | `--features pinn` | Required to enable the PINN backend and multi-physics training code |

## Key Code Snippet

```rust
let mut solver = UniversalPINNSolver::<Backend>::with_cavitation_sonoluminescence_coupling()?;

// Configure training for multi-physics coupling
let training_config = UniversalTrainingConfig {
    epochs: 500, // Reduced for example - increase for production
    learning_rate: 0.001,
    collocation_points: 2000, // More points for coupled system
    boundary_points: 400,
    initial_points: 200,
    adaptive_sampling: true,
```

## Expected Output (if applicable)

With the PINN feature enabled, the program lists registered domains, trains the coupled model, and prints per-stage progress from acoustics through optical emission.

## Book Chapter

[← Nonlinear Acoustics](../nonlinear_acoustics.md)
