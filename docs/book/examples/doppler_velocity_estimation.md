# Example: Doppler Velocity Estimation

**Crate**: `kwavers`  
**Run**: `cargo run -p kwavers --example doppler_velocity_estimation`  
**Source**: [`crates/kwavers/examples/doppler_velocity_estimation.rs`](../../../crates/kwavers/examples/doppler_velocity_estimation.rs)

## What This Example Demonstrates

This example shows a full Doppler processing path for vascular imaging. It starts from a vascular preset, synthesizes complex I/Q ensembles, estimates velocity with the Kasai autocorrelation method, suppresses clutter, and forms a color-flow style result.

| Component | API | Value |
|---|---|---|
| Preset | `AutocorrelationConfig::vascular()` | Builds a 7.5 MHz / 5 kHz PRF vascular configuration |
| Signal model | `Array3<Complex64>` | Stores ensemble, depth, and beam I/Q samples for the synthetic flow field |
| Estimator | `AutocorrelationEstimator` | Applies autocorrelation velocity estimation and wall filtering |

## Key Code Snippet

```rust
let doppler_config = AutocorrelationConfig::vascular();

println!(
    "  └─ Center frequency: {} MHz",
    doppler_config.center_frequency / 1e6
);
println!("  └─ PRF: {} kHz", doppler_config.prf / 1e3);
println!(
    "  └─ Ensemble size: {} pulses",
    doppler_config.ensemble_size
```

## Expected Output (if applicable)

The console prints imaging parameters, Nyquist velocity, processing stages, and summary statistics for the estimated flow field.

## Book Chapter

[← Sensors and Measurements](../sensors_and_measurements.md)
