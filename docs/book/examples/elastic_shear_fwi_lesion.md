# Example: Elastic Shear FWI Lesion

**Crate**: `kwavers-solver`
**Run**: `cargo run -p kwavers-solver --release --example elastic_shear_fwi_lesion`
**Source**: [`crates/kwavers-solver/examples/elastic_shear_fwi_lesion.rs`](../../../crates/kwavers-solver/examples/elastic_shear_fwi_lesion.rs)

## What This Example Demonstrates

Adjoint-state elastic full-waveform inversion on a synthetic stiff-inclusion phantom. Recovers the shear-modulus map of a stiff lesion in a compressible medium (Poisson ≈ 0.25), demonstrating the `kwavers_solver::inverse::elastography::elastic_fwi` engine.

## Physics

Elastic wave equation with P- and S-wave decomposition. The medium is deliberately compressible so the elastic CFL does not demand excessive time steps. Outputs true / initial / recovered shear-modulus maps as CSV for the book figure script.
