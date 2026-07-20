# Example: DG Acoustic Timestep Sweep

**Crate**: `kwavers`  
**Run**: `cargo run -p kwavers --example dg_acoustic_timestep_sweep`  
**Source**: [`crates/kwavers/examples/dg_acoustic_timestep_sweep.rs`](../../../crates/kwavers/examples/dg_acoustic_timestep_sweep.rs)

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
| 1 | 20 steps (coarsest timestep) |
| 2 | 40 steps |
| 3 | 80 steps (finest timestep) |

## Output Files

| File | Description |
|------|-------------|
| `timestep_sweep.png` | Error vs. timestep for all solvers |
| `timestep_sweep.csv` | Numerical data |

## Key Code Snippet

```rust
const STEP_COUNTS: [usize; 3] = [20, 40, 80];

let rows = run_timestep_sweep()?;
let out_dir = PathBuf::from("target/dg_acoustic_comparison");
fs::create_dir_all(&out_dir)?;
write_plot(&out_dir.join("timestep_sweep.png"), &rows)?;
write_csv(&out_dir.join("timestep_sweep.csv"), &rows)?;
```

## Book Chapter

[← Numerical Methods: FDTD and PSTD](../numerical_methods.md)
