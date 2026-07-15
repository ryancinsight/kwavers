# Example: DG Acoustic 1D Diagnostics

**Crate**: `kwavers`  
**Run**: `cargo run -p kwavers --example dg_acoustic_1d_diagnostics`  
**Source**: [`crates/kwavers/examples/dg_acoustic_1d_diagnostics.rs`](../../../../crates/kwavers/examples/dg_acoustic_1d_diagnostics.rs)

## What This Example Demonstrates

This diagnostic executable audits the native 1-D discontinuous Galerkin acoustic solver. It reuses the shared fixture from `dg_acoustic_common.rs` to report pressure and velocity discrepancies, characteristic errors, conservation metrics, and a reference Gaussian-series matrix.

| Component | API | Value |
|---|---|---|
| Shared fixture | `dg_common/dg_acoustic_common.rs` | Keeps the executable and plotting example on the same numerical setup |
| Core metrics | `run_native_acoustic_diagnostic` | Computes relative L2, characteristic L2, mass error, and energy ratio |
| Resolution knobs | `ELEMENTS`, `POLYNOMIAL_ORDER`, `STEPS` | Printed at runtime so solver audits are reproducible |

## Key Code Snippet

```rust
let diagnostic = run_native_acoustic_diagnostic()?;
let series = run_embedded_gaussian_series()?;

println!("DG native 1-D acoustic diagnostic");
println!("elements: {ELEMENTS}, polynomial_order: {POLYNOMIAL_ORDER}, steps: {STEPS}");
println!("system: p_t + rho*c^2*u_x = 0, u_t + p_x/rho = 0");
println!();
println!(
    "{:<36} {:>16.6e}",
    "pressure_relative_l2", diagnostic.pressure_relative_l2
```

## Expected Output (if applicable)

The output is a dense diagnostic table followed by an embedded Gaussian pressure matrix for solver-to-solver comparison.

## Book Chapter

[← Numerical Methods: FDTD and PSTD](../numerical_methods.md)
