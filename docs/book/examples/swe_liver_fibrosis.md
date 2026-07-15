# Example: SWE Liver Fibrosis

**Crate**: `kwavers`  
**Run**: `cargo run -p kwavers --example swe_liver_fibrosis`  
**Source**: [`crates/kwavers/examples/swe_liver_fibrosis.rs`](../../../../crates/kwavers/examples/swe_liver_fibrosis.rs)

## What This Example Demonstrates

This example turns shear-wave elastography into a clinical liver-fibrosis workflow. It builds ARFI push parameters, simulates viscoelastic shear-wave propagation, tracks displacement, reconstructs elasticity, and maps the result onto a fibrosis staging interpretation.

| Component | API | Value |
|---|---|---|
| Push pulse | `PushPulseParameters::new` | Uses a 4.5 MHz, 67 μs acoustic radiation-force excitation |
| Wave solver | `ElasticWaveSolver` | Propagates the induced shear wave through the liver model |
| Inversion | `ShearWaveInversionConfig::new(InversionMethod::TimeOfFlight)` | Reconstructs an elasticity map from the tracked displacements |

## Key Code Snippet

```rust
let swe_config = create_swe_parameters()?;

println!("📊 Simulation Setup:");
println!(
    "   Grid: {} × {} × {} ({} mm³)",
    grid.nx,
    grid.ny,
    grid.nz,
    (grid.nx as f64 * grid.dx * 1000.0) as usize
);
```

## Expected Output (if applicable)

The console prints clinical setup details, elasticity/stiffness estimates, and the inferred fibrosis severity summary.

## Book Chapter

[← Elastography: Imaging Tissue Mechanical Properties](../elastography.md)
