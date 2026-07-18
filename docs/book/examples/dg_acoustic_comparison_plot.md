# Example: DG Acoustic Comparison Plot

**Crate**: `kwavers`  
**Run**: `cargo run -p kwavers --example dg_acoustic_comparison_plot`  
**Source**: [`crates/kwavers/examples/dg_acoustic_comparison_plot.rs`](../../../../crates/kwavers/examples/dg_acoustic_comparison_plot.rs)

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
| `pressure_traces.png` | Pressure field comparison over time |
| `error_traces.png` | Error comparison for each solver |
| `pressure_traces.csv` | Numerical pressure data |
| `error_traces.csv` | Numerical error data |

## Key Code Snippet

```rust
// Generate pressure traces for each solver
let solvers = vec![
    ("Native DG", run_native_dg),
    ("Common DG", run_common_dg),
    ("Uniform DG", run_uniform_dg),
    ("FDTD", run_fdtd),
    ("k-space FDTD", run_kspace_fdtd),
    ("PSTD", run_pstd),
];

for (name, solver) in solvers {
    let (time, pressure) = solver.run()?;
    plot_pressure_trace(&format!("pressure_{}.png", name), &time, &pressure);
    plot_error_trace(&format!("error_{}.png", name), &time, &pressure);
}
```

## Book Chapter

[← Numerical Methods: FDTD and PSTD](../numerical_methods.md)
