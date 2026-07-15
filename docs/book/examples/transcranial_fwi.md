# Example: Transcranial FWI

**Crate**: `kwavers`  
**Run**: `cargo run -p kwavers --example transcranial_fwi`  
**Source**: [`crates/kwavers/examples/transcranial_fwi.rs`](../../../../crates/kwavers/examples/transcranial_fwi.rs)

## What This Example Demonstrates

This example demonstrates adjoint-state full-wave inversion for transcranial ultrasound. By default it uses a self-contained skull phantom, but the same code path can be redirected to real NIfTI-based CT/MRI data for more realistic studies.

| Component | API | Value |
|---|---|---|
| Domain | `DX`, `NX`, `NY`, `NZ` | Defines a 64×2×64 coronal head cross-section at 3 mm spacing |
| Skull phantom | `HU_SCALP`, `HU_CORTICAL_*`, `HU_DIPLOE`, `HU_BRAIN` | Maps layered CT-style intensities into heterogeneous acoustic properties |
| FWI loop | forward FDTD → residual → adjoint → gradient update | Implements the standard adjoint-state inversion workflow |

## Key Code Snippet

```rust
const DX: f64 = 3.0e-3;
/// Grid dimensions (2-D coronal slice embedded in 3-D; ny=2 satisfies FDTD
/// staggered-stencil minimum while keeping the second y-plane acoustically
/// transparent — identical medium properties are assigned to both planes).
const NX: usize = 64;
const NY: usize = 2;
const NZ: usize = 64;

/// Skull phantom geometry — all radii in voxels from grid centre (32, 0, 32).
///
```

## Expected Output (if applicable)

The run prints phantom and inversion summaries and can be extended to real-image input when the optional NIfTI path is enabled.

## Book Chapter

[← Transcranial Ultrasound: Physics, Aberration Correction, and Therapy](../transcranial_ultrasound.md)
