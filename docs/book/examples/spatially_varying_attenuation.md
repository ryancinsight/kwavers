# Example: Spatially Varying Attenuation

**Crate**: `kwavers`  
**Run**: `cargo run -p kwavers --example spatially_varying_attenuation`  
**Source**: [`crates/kwavers/examples/spatially_varying_attenuation.rs`](../../../../crates/kwavers/examples/spatially_varying_attenuation.rs)

## What This Example Demonstrates

This example demonstrates heterogeneous power-law attenuation where both the absorption coefficient and the frequency exponent vary across space. The demo covers uniform tissue, tumor inclusions, and temperature-aware extensions relevant to therapy modeling.

| Component | API | Value |
|---|---|---|
| Baseline field | `SpatiallyVaryingAbsorption::uniform` | Creates a 100×100×100 soft-tissue volume with α₀=0.75 Np/m and γ=1.1 |
| Lesion editing | `add_spherical_inclusion` | Adds a 1 cm tumor with elevated absorption and stronger frequency dependence |
| Thermal extension | `with_temperature_dependence` | Shows how the same model can be adapted for HIFU-style heating studies |

## Key Code Snippet

```rust
let uniform = SpatiallyVaryingAbsorption::uniform(
    100, 100, 100, 0.75, // α₀ = 0.75 Np/m
    1.1,  // γ = 1.1 (typical soft tissue)
)?;

let stats = uniform.statistics();
println!("   α₀: {:.3} Np/m (uniform)", stats.alpha_0_mean);
println!("   γ:  {:.2} (uniform)", stats.gamma_mean);

// Calculate absorption at different frequencies
```

## Expected Output (if applicable)

The output reports attenuation statistics, frequency scaling, and differences between the background, lesion, and temperature-dependent models.

## Book Chapter

[← Media and Tissue Models](../media_and_tissue_models.md)
