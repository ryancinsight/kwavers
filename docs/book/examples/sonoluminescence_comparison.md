# Example: Sonoluminescence Comparison

**Crate**: `kwavers`  
**Run**: `cargo run -p kwavers --example sonoluminescence_comparison`  
**Source**: [`crates/kwavers/examples/sonoluminescence_comparison.rs`](../../../../crates/kwavers/examples/sonoluminescence_comparison.rs)

## What This Example Demonstrates

This example directly compares bremsstrahlung and Cherenkov hypotheses for sonoluminescence using the same bubble-collapse trajectory. It is useful when you want the differences in thresholds, spectra, and coherence summarized in one executable.

| Component | API | Value |
|---|---|---|
| Mechanism table | bremsstrahlung vs Cherenkov | Contrasts thermal free-free emission with coherent threshold radiation |
| Bubble driver | `KellerMiksisModel` | Uses a 4 µm initial bubble and acoustic collapse dynamics for both hypotheses |
| Optical output | `IntegratedSonoluminescence` | Computes the emissions on a common simulation backbone for side-by-side comparison |

## Key Code Snippet

```rust
let bubble_params = BubbleParameters {
    r0: 4e-6,                       // 4 μm initial radius
    t0: 300.0,                      // 300 K ambient temperature
    gamma: 1.667, // Argon polytropic index (Monatomic gas is essential for SBSL)
    initial_gas_pressure: 101325.0, // 1 atm
    ..Default::default()
};

// Test different emission scenarios
run_bremsstrahlung_dominant_scenario(&grid, &bubble_params)?;
```

## Expected Output (if applicable)

The console prints mechanism comparison tables, collapse diagnostics, and a summary of which emission hypothesis dominates under the chosen conditions.

## Book Chapter

[← Nonlinear Acoustics](../nonlinear_acoustics.md)
