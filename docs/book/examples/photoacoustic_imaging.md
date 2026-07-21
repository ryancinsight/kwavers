# Example: Photoacoustic Imaging

**Crate**: `kwavers`  
**Run**: `cargo run -p kwavers --example photoacoustic_imaging`  
**Source**: [`crates/kwavers/examples/photoacoustic_imaging.rs`](../../../crates/kwavers/examples/photoacoustic_imaging.rs)

## What This Example Demonstrates

This example runs an end-to-end photoacoustic simulation. It defines a computational grid, creates a heterogeneous tissue phantom with blood vessels and tumor contrast, configures laser parameters, and then reconstructs an image from the generated acoustic response.

| Component | API | Value |
|---|---|---|
| Grid | `create_simulation_grid()` | Returns a 64×64×32 domain with 200 µm lateral and 400 µm axial spacing |
| Medium | `create_tissue_medium(&grid)` | Builds a heterogeneous photoacoustic target with vessel and tumor structure |
| Simulator | `PhotoacousticSimulator::new` | Combines medium, optical properties, and acquisition parameters into one workflow object |

## Key Code Snippet

```rust
let grid = create_simulation_grid()?;
println!("📊 Computational Grid:");
println!("   Dimensions: {} × {} × {}", grid.nx, grid.ny, grid.nz);
println!(
    "   Physical size: {:.1} × {:.1} × {:.1} mm",
    grid.nx as f64 * grid.dx * 1000.0,
    grid.ny as f64 * grid.dy * 1000.0,
    grid.nz as f64 * grid.dz * 1000.0
);
```

## Expected Output (if applicable)

The run prints grid and optical settings, fluence computation progress, and reconstruction/validation summaries.

## Book Chapter

[← Photoacoustic Imaging](../photoacoustics.md)
