# Example: PSTD vs FDTD Comparison

**Crate**: `kwavers`  
**Run**: `cargo run -p kwavers --example pstd_fdtd_comparison`  
**Source**: [`crates/kwavers/examples/pstd_fdtd_comparison.rs`](../../../crates/kwavers/examples/pstd_fdtd_comparison.rs)

## What This Example Demonstrates

Solver comparison of three acoustic wave equation discretizations on the same
Gaussian pressure-pulse initial value problem in a homogeneous, lossless medium.

| Solver | Method | k-space correction |
|---|---|---|
| Classical FDTD | Staggered-grid finite differences | None |
| k-space FDTD | Staggered-grid + spectral k-space derivative | `Spectral` |
| PSTD | Pseudospectral time-domain | Full spectral |

## Theorem Validated

All three discretizations solve the same linear acoustic Cauchy problem:

```text
ρ₀ ∂u/∂t = −∇p
∂p/∂t = −ρ₀c₀² ∇·u
```

With `KSpaceCorrectionMode::Spectral`, the k-space FDTD uses the same
spectral derivative family as PSTD — so in short windows (before boundary
reflections), it should agree more closely with PSTD than classical FDTD.

## Key Code Snippet

```rust
use kwavers_solver::{FdtdSolver, KSpaceCorrectionMode, PstdSolver};

// Classical FDTD
let fdtd = FdtdSolver::new(&grid, &medium, KSpaceCorrectionMode::None)?;

// k-space-corrected FDTD
let kfdtd = FdtdSolver::new(&grid, &medium, KSpaceCorrectionMode::Spectral)?;

// PSTD
let pstd = PstdSolver::new(&grid, &medium)?;
```

## References

- Treeby & Cox (2010). J. Biomed. Opt. 15(2), 021314 — k-space pseudospectral acoustic modeling
- Yee (1966). IEEE Trans. Antennas Propag. 14(3), 302-307 — FDTD

## Book Chapter

[← Numerical Methods: FDTD and PSTD](../numerical_methods.md)
