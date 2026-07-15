# Example: Photoacoustic Blood Oxygenation

**Crate**: `kwavers`  
**Run**: `cargo run -p kwavers --example photoacoustic_blood_oxygenation`  
**Source**: [`crates/kwavers/examples/photoacoustic_blood_oxygenation.rs`](../../../../crates/kwavers/examples/photoacoustic_blood_oxygenation.rs)

## What This Example Demonstrates

This example shows a full multi-wavelength oxygenation workflow: diffuse-light fluence is computed at several laser wavelengths, hemoglobin species are unmixed, and the recovered concentrations are converted into an sO₂ map for validation against known arterial and venous values.

| Component | API | Value |
|---|---|---|
| Wavelength set | `wavelengths = vec![532.0, 700.0, 800.0, 850.0]` | Covers strong absorption, isosbestic, and NIR regimes for hemoglobin spectroscopy |
| Optical solver | `DiffusionSolverConfig` + `DiffusionSolver` | Computes fluence maps used in the photoacoustic forward model |
| Unmixing | `estimate_oxygenation` + `SpectralUnmixingConfig` | Separates HbO₂/Hb and computes blood oxygen saturation |

## Key Code Snippet

```rust
let wavelengths = vec![
    532.0, // Green (Nd:YAG doubled) - strong Hb absorption
    700.0, // Red edge - near isosbestic point
    800.0, // NIR window - HbO₂ peak
    850.0, // NIR window - balanced penetration
];
println!("Wavelengths: {:?} nm", wavelengths);

// Computational grid (5mm × 5mm × 5mm at 0.2mm resolution)
let grid = Grid::new(25, 25, 25, 0.2e-3, 0.2e-3, 0.2e-3)?;
```

## Expected Output (if applicable)

The console prints the selected wavelengths, phase-by-phase timing, and oxygenation estimates that can be compared with the synthetic ground truth.

## Book Chapter

[← Photoacoustic Imaging](../photoacoustics.md)
