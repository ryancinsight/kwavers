# Example: DG Acoustic Convergence Plot

**Crate**: `kwavers`  
**Run**: `cargo run -p kwavers --example dg_acoustic_convergence_plot`  
**Source**: [`crates/kwavers/examples/dg_acoustic_convergence_plot.rs`](../../../crates/kwavers/examples/dg_acoustic_convergence_plot.rs)

## What This Example Demonstrates

This example generates convergence plots for Discontinuous Galerkin (DG) methods using p-refinement (increasing polynomial order). It produces PNG and CSV outputs showing error vs. polynomial order.

## Polynomial Orders Tested

| Order | Description |
|-------|-------------|
| 1 | Linear elements |
| 2 | Quadratic elements |
| 3 | Cubic elements |
| 4 | Quartic elements and the shared comparison quadrature |

## Output Files

| File | Description |
|------|-------------|
| `target/dg_acoustic_comparison/dg_order_convergence.png` | Convergence curve (error vs. order) |
| `target/dg_acoustic_comparison/dg_order_convergence.csv` | Per-order degrees of freedom, L2 errors, and mass error |

## Key Code Snippet

```rust
const ORDERS: [usize; 4] = [1, 2, 3, 4];

let rows = run_convergence_sweep()?;
let out_dir = PathBuf::from("target/dg_acoustic_comparison");
fs::create_dir_all(&out_dir)?;
write_plot(&out_dir.join("dg_order_convergence.png"), &rows)?;
write_csv(&out_dir.join("dg_order_convergence.csv"), &rows)?;
```

## Book Chapter

[← Numerical Methods: FDTD and PSTD](../numerical_methods.md)
