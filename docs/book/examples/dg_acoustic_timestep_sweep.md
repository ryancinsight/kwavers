# Example: DG Acoustic Timestep Sweep

**Crate**: `kwavers`  
**Run**: `cargo run -p kwavers --example dg_acoustic_timestep_sweep`  
**Source**: [`crates/kwavers/examples/dg_acoustic_timestep_sweep.rs`](../../../../crates/kwavers/examples/dg_acoustic_timestep_sweep.rs)

## What This Example Demonstrates

This example performs a timestep refinement sweep for Discontinuous Galerkin (DG) methods. It tests different acoustic solvers (DG, FDTD, k-space FDTD, PSTD) with varying timesteps to show stability and accuracy characteristics.

## Solvers Tested

| Solver | Type |
|--------|------|
| DG | Discontinuous Galerkin |
| FDTD | Finite-Difference Time Domain |
| k-space FDTD | k-space corrected FDTD |
| PSTD | Pseudo-Spectral Time Domain |

## Timestep Refinement Levels

| Level | Description |
|-------|-------------|
| 1 | Coarsest timestep |
| 2 | Medium timestep |
| 3 | Fine timestep |
| 4 | Very fine timestep |

## Output Files

| File | Description |
|------|-------------|
| `timestep_sweep.png` | Error vs. timestep for all solvers |
| `timestep_sweep.csv` | Numerical data |

## Key Code Snippet

```rust
// Test different timesteps
let timesteps = compute_timestep_range()?;
let mut results = Vec::new();

for dt in timesteps {
    let (_, error) = run_solver_with_timestep(dt)?;
    results.push((dt, error));
}

// Plot sweep
plot_timestep_sweep(&"timestep_sweep.png", &results)?;
```

## Book Chapter

[← Numerical Methods: FDTD and PSTD](../numerical_methods.md)
