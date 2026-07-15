# Example: Focused Ultrasound Water Tank

**Crate**: `kwavers`  
**Run**: `cargo run -p kwavers --example focused_ultrasound_water_tank`  
**Source**: [`crates/kwavers/examples/focused_ultrasound_water_tank.rs`](../../../../crates/kwavers/examples/focused_ultrasound_water_tank.rs)

## What This Example Demonstrates

Multi-solver comparison of focused ultrasound in a homogeneous water tank.
Four solvers (FDTD+CPML, PSTD+CPML, DG-2D, DG-3D) are driven with the same
phased line aperture and compared against the analytical O'Neil pressure field.

| Solver | Method | Reference |
|---|---|---|
| FDTD+CPML | Staggered-grid finite differences + CPML absorber | Yee (1966) |
| PSTD+CPML | k-space pseudospectral + CPML | Treeby & Cox (2010) |
| DG-2D | Nodal discontinuous Galerkin, 2D | Hesthaven & Warburton (2008) |
| DG-3D | Nodal DG, tensor-product 3D | Cockburn & Shu (2001) |

## Outputs

- `target/focused_water_tank/focused_water_tank.png` — peak-pressure maps
- `target/focused_water_tank/focused_water_tank_metrics.csv` — solver accuracy metrics
- `target/focused_water_tank/focused_water_tank_profiles.csv` — axial pressure profiles

## Key Code Snippet

```rust
use kwavers_transducer::PhasedArrayTransducer;

// Drive a 16-element phased array with analytical focus delays
let transducer = PhasedArrayTransducer::builder()
    .elements(16)
    .element_pitch(0.5e-3)   // λ/2 at 1.5 MHz in water
    .focus_point([0.0, 0.0, 30e-3]) // 30 mm focal depth
    .build()?;
```

## Book Chapter

[← Sources and Transducers](../sources_and_transducers.md)
