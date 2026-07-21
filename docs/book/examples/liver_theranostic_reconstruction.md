# Example: Liver Theranostic Reconstruction

**Crate**: `kwavers`  
**Run**: `cargo run -p kwavers --example liver_theranostic_reconstruction`  
**Source**: [`crates/kwavers/examples/liver_theranostic_reconstruction.rs`](../../../crates/kwavers/examples/liver_theranostic_reconstruction.rs)

## What This Example Demonstrates

This example reconstructs liver acoustic-property maps for focused-ultrasound therapy planning. It compares straight-ray sound-speed estimation, Born inversion, reverse-time migration, and adjoint FWI, and it can use either a LiTS CT slice or a synthetic abdominal fallback phantom.

| Component | API | Value |
|---|---|---|
| Grid constants | `DX`, `NX`, `NY`, `NZ` | Uses a 3 mm spacing with an 80×5×80 slice-style reconstruction volume |
| Reconstruction family | SoS / Born / RTM / FWI | Produces four complementary views of the same liver target |
| Input data | `data/lits17_sample/*.nii` | Loads LiTS CT + segmentation when available, otherwise falls back to a synthetic phantom |

## Key Code Snippet

```rust
const DX: f64 = 3.0e-3;
const NX: usize = 80;
/// 2-D coronal slice embedded in 3-D.  RTM's 4th-order FD Laplacian requires
/// `ny ≥ 5` (interior slice `2..ny-2`), so we pad in y with acoustically
/// identical planes — the physics remains 2-D.
const NY: usize = 5;
const NZ: usize = 80;

// ─────────────────────────────────────────────────────────────────────────────
// Tissue acoustic constants
```

## Expected Output (if applicable)

A run prints which dataset path is used, steps through the reconstruction stages, and summarizes therapy-planning outputs derived from the recovered maps.

## Book Chapter

[← Theranostics: Combined Imaging and Therapy](../theranostics.md)
