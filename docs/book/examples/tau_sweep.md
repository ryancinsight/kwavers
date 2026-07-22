# Example: Tau Parameter Sweep

**Crate**: `kwavers`  
**Run**: `cargo run -p kwavers --example tau_sweep`  
**Source**: [`crates/kwavers/examples/tau_sweep.rs`](../../../crates/kwavers/examples/tau_sweep.rs)

## Overview

Sweeps the frequency power-law absorption exponent `τ` (tau) over a range of values to characterise its effect on acoustic attenuation and dispersion in tissue models.

## Physics

The frequency-dependent absorption follows:

```text
α(f) = α₀ · f^τ [dB/cm]
```

where τ ≈ 1 for soft tissue and τ = 2 for water. The Kramers–Kronig relations link the τ-dependence of absorption to the corresponding dispersion.

## Output

Prints a table of attenuation coefficient, phase speed, and group speed for each τ value at the simulation centre frequency.

## Part Reference

Part I — Media and Tissue Models
