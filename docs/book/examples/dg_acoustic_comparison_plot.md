# Example: DG Acoustic Comparison Plot

**Crate**: `kwavers`  
**Run**: `cargo run -p kwavers --example dg_acoustic_comparison_plot`  
**Source**: [`crates/kwavers/examples/dg_acoustic_comparison_plot.rs`](../../../crates/kwavers/examples/dg_acoustic_comparison_plot.rs)

## What This Example Demonstrates

This example generates comparison plots for different acoustic solvers on a common grid. It produces PNG and CSV outputs for visual and numerical comparison.

## Solvers Compared

| Solver | Description |
|--------|-------------|
| Native DG | Discontinuous Galerkin on native grid |
| Common DG | DG on common grid (interpolated) |
| Uniform DG | DG on uniform grid |
| FDTD | Finite-Difference Time Domain |
| k-space FDTD | k-space corrected FDTD |
| PSTD | Pseudo-Spectral Time Domain |

## Output Files

| File | Description |
|------|-------------|
| `target/dg_acoustic_comparison/gaussian_pressure.png` | Native, common-quadrature, and uniform-grid pressure and error panels |
| `target/dg_acoustic_comparison/gaussian_pressure.csv` | Pressure, absolute-error, and solver-matrix values used by the plot |

## Key Code Snippet

```rust
let series = run_embedded_gaussian_series()?;
let uniform = uniform_resampling(&series)?;
let out_dir = PathBuf::from("target/dg_acoustic_comparison");
fs::create_dir_all(&out_dir)?;

write_plot(&out_dir.join("gaussian_pressure.png"), &series, &uniform)?;
write_csv(&out_dir.join("gaussian_pressure.csv"), &series, &uniform)?;
```

## Book Chapter

[← Numerical Methods: FDTD and PSTD](../numerical_methods.md)
