# Example: Brain Theranostic Monitor

**Crate**: `kwavers`  
**Run**: `cargo run -p kwavers --example brain_theranostic_monitor`  
**Source**: [`crates/kwavers/examples/brain_theranostic_monitor.rs`](../../../crates/kwavers/examples/brain_theranostic_monitor.rs)

## What This Example Demonstrates

This example demonstrates an end-to-end theranostic brain workflow: CT/phantom-derived acoustics feed sparse full-wave inversion, therapy pulses perturb the medium, and the monitored slice is reconstructed from synthetic data rather than read directly from ground truth. The result is a realistic interleaved therapy-and-imaging loop.

| Component | API | Value |
|---|---|---|
| Volume model | `DX`, `NX`, `NY`, `NZ` | 3 mm spacing on a 40×24×40 brain/skull volume |
| Therapy aperture | `ARRAY_ELEMENTS`, `ARRAY_RADIUS_M` | 1024-element hemispherical array with 70 mm radius |
| Mode switch | `KW_THERAPY_MODE` | Selects `thermal` (default) or `cavitation` lesion evolution |

## Key Code Snippet

```rust
const DX: f64 = 3.0e-3; // m
const NX: usize = 40; // 120 mm lateral
const NY: usize = 24; // 72 mm elevation (true 3-D recon)
const NZ: usize = 40; // 120 mm depth

// ── Phantom geometry (voxels from grid centre) ──────────────────────────────
const R_HEAD: f64 = 16.0;
const R_SKULL_OUT: f64 = 15.0;
const R_SKULL_IN: f64 = 13.0;
const R_BRAIN: f64 = 12.0;
```

## Expected Output (if applicable)

A normal run prints therapy/imaging progress tables, lesion and dose metrics, and writes PNG frames for the monitored slice.

## Book Chapter

[← Theranostics: Combined Imaging and Therapy](../theranostics.md)
