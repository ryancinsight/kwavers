# Example: Advanced Ultrasound Imaging

**Crate**: `kwavers`  
**Run**: `cargo run -p kwavers --example advanced_ultrasound_imaging`  
**Source**: [`crates/kwavers/examples/advanced_ultrasound_imaging.rs`](../../../crates/kwavers/examples/advanced_ultrasound_imaging.rs)

## What This Example Demonstrates

This example walks through three modern ultrasound imaging modes in one place: synthetic aperture imaging, plane-wave compounding, and coded excitation with pulse compression. It is useful when you want to compare how kwavers configures different high-performance reconstruction pipelines.

| Component | API | Value |
|---|---|---|
| Synthetic aperture | `SyntheticApertureConfig` | 32 transmitters, 32 receivers, 5 MHz center frequency, 40 MHz sampling |
| Plane waves | `UltrasoundPlaneWaveConfig` + `PlaneWaveReconstruction` | Builds compounded images from multiple steering angles |
| Coded excitation | `CodedExcitationProcessor` | Applies pulse compression to improve penetration without losing resolution |

## Key Code Snippet

```rust
let sa_config = SyntheticApertureConfig {
    num_tx_elements: 32,
    num_rx_elements: 32,
    element_spacing: 0.3e-3, // 0.3mm
    sound_speed: 1540.0,
    frequency: 5e6,
    sampling_frequency: 40e6,
    num_tx_angles: 1,
};
```

## Expected Output (if applicable)

The console output is split into three sections—synthetic aperture, plane-wave imaging, and coded excitation—and ends with a completion banner.

## Book Chapter

[← Diagnostic Ultrasound Imaging](../diagnostics.md)
