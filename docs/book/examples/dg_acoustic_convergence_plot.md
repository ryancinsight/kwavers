# Example: DG Acoustic Convergence Plot

**Crate**: `kwavers`  
**Run**: `cargo run -p kwavers --example dg_acoustic_convergence_plot`  
**Source**: [`crates/kwavers/examples/dg_acoustic_convergence_plot.rs`](../../../../crates/kwavers/examples/dg_acoustic_convergence_plot.rs)

## What This Example Demonstrates

This example generates convergence plots for Discontinuous Galerkin (DG) methods using p-refinement (increasing polynomial order). It produces PNG and CSV outputs showing error vs. polynomial order.

## Polynomial Orders Tested

| Order | Description |
|-------|-------------|
| 2 | Quadratic elements |
| 3 | Cubic elements |
| 4 | Quartic elements |
| 5 | Quintic elements |
| 6 | Sextic elements |

## Output Files

| File | Description |
|------|-------------|
| `convergence_dg.png` | Convergence curve (error vs. order) |
| `convergence_dg.csv` | Numerical convergence data |

## Key Code Snippet

```rust
// Test different polynomial orders
let orders = vec![2, 3, 4, 5, 6];
let mut errors = Vec::new();

for order in orders {
    let (_, error) = run_dg_polynomial(order)?;
    errors.push(error);
}

// Plot convergence
plot_convergence_curve(&"convergence_dg.png", &orders, &errors)?;
```

## Book Chapter

[← Numerical Methods: FDTD and PSTD](../numerical_methods.md)
