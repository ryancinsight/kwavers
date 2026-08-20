# Example: Single-Bubble Sonoluminescence

**Crate**: `kwavers`  
**Run**: `cargo run -p kwavers --example single_bubble_sonoluminescence`  
**Source**: [`crates/kwavers/examples/single_bubble_sonoluminescence.rs`](../../../crates/kwavers/examples/single_bubble_sonoluminescence.rs)

This bounded example executes eight Keller–Miksis steps for one bubble cell.
Each step refreshes the integrated emission field from the updated state.
Blackbody and bremsstrahlung are reported as Aequitas
`VolumetricPowerDensity` values. The Cherenkov spectrum is queried separately
in arbitrary spectral units and is not added to the dimensioned power field.

```bash
cargo run -p kwavers --example single_bubble_sonoluminescence
```

The output is input-sensitive: changing the acoustic pressure or initial bubble
state changes the reported radius, temperature, and emission components.

[← Nonlinear Acoustics](../nonlinear_acoustics.md)
