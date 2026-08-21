# Example: Transcranial FWI

**Crate**: `kwavers`  
**Run**: `cargo run -p kwavers --example transcranial_fwi`  
**Source**: [`crates/kwavers/examples/transcranial_fwi.rs`](../../../crates/kwavers/examples/transcranial_fwi.rs)

## What This Example Demonstrates

This example demonstrates adjoint-state full-wave inversion for transcranial
ultrasound. Its input is explicit: `KWAVERS_SEISMIC_INPUT_MODE=synthetic`
selects the self-contained skull phantom, while `ct:<path>` selects a real CT
NIfTI volume. A CT read failure is returned as an input error rather than
silently switching models.

| Component | API | Value |
|---|---|---|
| Domain | `DX`, `NX`, `NY`, `NZ` | Defines a 64×2×64 coronal head cross-section at 3 mm spacing |
| Skull phantom | `seismic_imaging::medium::SkullModel` | Routes layered synthetic or explicit CT intensities through the validated provider-owned Hill model |
| Source | `DomainRickerWavelet` | Streams causal pressure samples from Aequitas-typed frequency, time, and pressure |
| FWI loop | forward FDTD → residual → adjoint → gradient update | Implements the standard adjoint-state inversion workflow |

## Key Code Snippet

```rust,ignore
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

The run prints provider-derived HU and sound-speed ranges plus inversion
summaries to stdout. No dataset is downloaded and no file is generated
implicitly.

## Book Chapter

[← Transcranial Ultrasound: Physics, Aberration Correction, and Therapy](../transcranial_ultrasound.md)
