# Example: Elastography Simulation

**Crate**: `kwavers`  
**Run**: `cargo run -p kwavers --example elastography_simulation`  
**Source**: [`crates/kwavers/examples/elastography_simulation.rs`](../../../crates/kwavers/examples/elastography_simulation.rs)

## What This Example Demonstrates

This example builds a high-resolution tissue block for elastography experiments. The setup uses a muscle-like background and explicitly constructs a stiff spherical inclusion in the shear-modulus map, which is the starting point for downstream displacement or reconstruction studies.

| Component | API | Value |
|---|---|---|
| Grid | `Grid::new` | 4 cm cubic domain sampled at 0.2 mm spacing |
| Background tissue | `HeterogeneousTissueMedium::new(...Muscle)` | Starts from a muscle absorption/tissue model |
| Stiff lesion | `mu` modulus map | Uses a 3 kPa background with a 5 mm-radius stiffer inclusion |

## Key Code Snippet

```rust
let domain_size = 0.04f64; // 4 cm cubic domain
let dx = 0.0002f64; // 0.2 mm spacing
let n = (domain_size / dx).round() as usize;
let grid = Grid::new(n, n, n, dx, dx, dx)?;

info!(
    "Created {}x{}x{} grid with {} mm spacing",
    n,
    n,
    n,
```

## Expected Output (if applicable)

The run logs grid creation and inclusion setup, then proceeds into the elastography phantom generation workflow.

## Book Chapter

[← Elastography: Imaging Tissue Mechanical Properties](../elastography.md)
