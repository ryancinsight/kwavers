# Example: Minimal Demo

**Crate**: `kwavers`  
**Run**: `cargo run -p kwavers --example minimal_demo`  
**Source**: [`crates/kwavers/examples/minimal_demo.rs`](../../../crates/kwavers/examples/minimal_demo.rs)

## What This Example Demonstrates

A compact self-contained wave propagation demo using raw `leto::Array3` storage.
Initialises a Gaussian pressure pulse at the grid centre and evolves it with a
simple 2nd-order finite difference stencil, tracking peak energy over 10 steps.

| Aspect | Detail |
|---|---|
| Grid | 32×32×32, 1 mm spacing, water |
| Initial condition | Gaussian pulse: P₀ = 10⁵ Pa, σ = 2 voxels |
| Solver | FDTD, explicit, 2nd-order central difference |
| Timestep | CFL = 0.3 → Δt ≈ 200 ns |

## Key Code Snippet

```rust
use kwavers_core::constants::{DENSITY_WATER, SOUND_SPEED_WATER};
use kwavers_grid::Grid;
use leto::Array3;

let grid = Grid::new(32, 32, 32, 1e-3, 1e-3, 1e-3)?;

// Gaussian pressure pulse centred on the grid
let mut pressure = Array3::zeros((32, 32, 32));
for i in 0..32 { for j in 0..32 { for k in 0..32 {
    let r2 = ((i as f64 - 16.0).powi(2) + (j as f64 - 16.0).powi(2) + (k as f64 - 16.0).powi(2));
    pressure[[i, j, k]] = 1e5 * (-r2 / 8.0).exp();
}}}

// CFL-stable timestep
let dt = 0.3 * grid.dx / SOUND_SPEED_WATER; // ≈ 200 ns
```

## Expected Output

```
Simulation Parameters:
  Grid: 32x32x32 points
  Spacing: 1.0 mm
  Sound speed: 1500.0 m/s
  CFL number: 0.30
  Timestep: 200.00 ns

Step 10: Energy = 1.48e11 Pa²
Simulation complete!
```

## Book Chapter

[← Wave Physics Fundamentals](../foundations.md)
