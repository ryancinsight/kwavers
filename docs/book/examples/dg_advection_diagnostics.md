# Example: DG Advection Diagnostics

**Crate**: `kwavers`  
**Run**: `cargo run -p kwavers --example dg_advection_diagnostics`  
**Source**: [`crates/kwavers/examples/dg_advection_diagnostics.rs`](../../../crates/kwavers/examples/dg_advection_diagnostics.rs)

## What This Example Demonstrates

This example provides diagnostic metrics for Discontinuous Galerkin (DG) methods applied to scalar and acoustic problems. It validates the DG implementation against analytical solutions.

## Metrics Computed

| Metric | Description |
|--------|-------------|
| Mass Conservation | L2 norm of density/pressure over time |
| Phase Accuracy | Comparison with analytical phase velocity |
| Amplitude Fidelity | Comparison with analytical solution amplitude |
| Relative L2 Error | Overall solution accuracy |

## Test Cases

| Test Case | Description |
|-----------|-------------|
| Periodic Advection | Scalar transport in periodic domain |
| One-Way Acoustic | Pressure wave propagating in one direction |
| Bidirectional Acoustic | Pressure and velocity waves in both directions |

## Key Code Snippet

```rust
// Periodic advection test
let mass = compute_mass_conservation(&field, &analytical_solution);
let phase = compute_phase_accuracy(&field, &analytical_solution);
let amplitude = compute_amplitude_fidelity(&field, &analytical_solution);
let relative_l2 = compute_relative_l2_error(&field, &analytical_solution);

println!("Mass conservation: {:.6e}", mass);
println!("Phase accuracy: {:.6e}", phase);
println!("Amplitude fidelity: {:.6e}", amplitude);
println!("Relative L2 error: {:.6e}", relative_l2);
```

## Book Chapter

[← Numerical Methods: FDTD and PSTD](../numerical_methods.md)
