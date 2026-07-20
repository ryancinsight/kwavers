# Example: Hybrid Lesion Monitor

**Crate**: `kwavers`  
**Run**: `cargo run -p kwavers --example hybrid_lesion_monitor`  
**Source**: [`crates/kwavers/examples/hybrid_lesion_monitor.rs`](../../../crates/kwavers/examples/hybrid_lesion_monitor.rs)

## What This Example Demonstrates

This example demonstrates the ADR-024 hybrid lesion-monitoring path. It combines frequency-domain convergent-Born-series inversion with passive acoustic mapping and then fuses both maps so that only mutually supported lesion growth remains in the final monitor image.

| Component | API | Value |
|---|---|---|
| Monitor slice | `N`, `SPACING_M` | 12×12 pixels at 1 mm spacing |
| Lesion evolution | `LESION_DC`, `N_FRAMES` | Applies up to a 60 m/s sound-speed rise over 5 growth frames |
| Passive map geometry | `RING_ELEMENTS`, `RING_DIAMETER_M` | Uses a 16-element ring aperture with 18 mm diameter |

## Key Code Snippet

```rust
let cfg = fd::FdMonitorConfig {
    ring_elements: RING_ELEMENTS,
    ring_diameter_m: RING_DIAMETER_M,
    spacing_m: SPACING_M,
    frequencies_hz: vec![3.0e5, 5.0e5],
    reference_sound_speed_m_s: C_BRAIN,
    min_sound_speed_m_s: 1480.0,
    max_sound_speed_m_s: 1800.0,
    fwi_iterations: 6,
    estimate_source_scaling: false,
```

## Expected Output (if applicable)

Each frame prints lesion-extent metrics and writes a `hybrid_lesion_frameNN.png` artifact showing the fused monitor image.

## Book Chapter

[← Theranostics: Combined Imaging and Therapy](../theranostics.md)
