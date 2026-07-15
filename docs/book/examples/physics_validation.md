# Example: Physics Validation

**Crate**: `kwavers`  
**Run**: `cargo run -p kwavers --example physics_validation`  
**Source**: [`crates/kwavers/examples/physics_validation.rs`](../../../../crates/kwavers/examples/physics_validation.rs)

## What This Example Demonstrates

This example validates numerical building blocks against analytical physics results. The selected cases span Gaussian heat diffusion, wave-dispersion analysis, and acoustic absorption so you can see how the codebase checks correctness against known formulas.

| Component | API | Value |
|---|---|---|
| Diffusion benchmark | `test_heat_diffusion()` | Compares numerical spreading against a Gaussian analytical solution |
| Wave benchmark | `test_wave_dispersion()` | Measures numerical dispersion across a known wave solution |
| Absorption benchmark | `test_acoustic_absorption()` | Checks attenuation behavior against expected exponential decay |

## Key Code Snippet

```rust
    test_heat_diffusion()?;

    // Test 2: Wave propagation with dispersion analysis
    test_wave_dispersion()?;

    // Test 3: Acoustic absorption validation
    test_acoustic_absorption()?;

    Ok(())
}
```

## Expected Output (if applicable)

The executable prints a section for each validation problem and includes error summaries or tables for the numerical-vs-analytical comparison.

## Book Chapter

[← Validation and Benchmarking](../validation_and_benchmarking.md)
