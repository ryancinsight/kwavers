# Example: Phased Array Beamforming

**Crate**: `kwavers`  
**Run**: `cargo run -p kwavers --example phased_array_beamforming`  
**Source**: [`crates/kwavers/examples/phased_array_beamforming.rs`](../../../crates/kwavers/examples/phased_array_beamforming.rs)

## What This Example Demonstrates

This example demonstrates how kwavers builds and exercises phased-array transducers for electronic focusing, steering, custom delay patterns, and cross-talk studies. It is the concrete “how do I configure the array?” companion to the beamforming chapter.

| Component | API | Value |
|---|---|---|
| Simulation box | `Grid::new(64, 64, 64, 1e-4, ...)` | Creates a 6.4 mm cubic water volume at 100 µm spacing |
| Array model | `PhasedArrayConfig` | Configures a 32-element array and its beamforming mode |
| Drive signal | `SineWave::new(2.5e6, 1.0, 0.0)` | Uses a 2.5 MHz excitation for the array elements |

## Key Code Snippet

```rust
let array_config = PhasedArrayConfig {
    num_elements: 32,
    element_spacing: 0.3e-3, // λ/2 at 2.5 MHz
    element_width: 0.25e-3,
    element_height: 10e-3,
    center_position: (0.0, 0.0, 0.0),
    frequency: 2.5e6,
    enable_crosstalk: true,
    crosstalk_coefficient: 0.05,
};
```

## Expected Output (if applicable)

A full run prints separate sections for focusing, steering, custom patterns, and cross-talk, followed by a success banner.

## Book Chapter

[← Transducer Arrays and Beamforming](../beamforming_and_image_formation.md)
