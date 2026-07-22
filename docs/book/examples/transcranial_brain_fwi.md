# Example: Transcranial Brain FWI

**Crate**: `kwavers-solver`
**Run**: `cargo run -p kwavers-solver --example transcranial_brain_fwi`
**Source**: [`crates/kwavers-solver/examples/transcranial_brain_fwi.rs`](../../../crates/kwavers-solver/examples/transcranial_brain_fwi.rs)

## What This Example Demonstrates

Genuine (non-inverse-crime) transcranial brain FWI reconstruction. Unlike the MOFI registration demo, the brain anomaly is unknown to the starting model and is recovered pixel-wise from noisy multi-shot data with the skull frozen to its known value (`invert_multi_source_masked`; Guasch 2020). The result is deliberately imperfect — honest behaviour of an ill-posed limited-aperture inversion.

## Key Concepts

- Multi-source masked inversion (skull frozen, brain recovered)
- Additive measurement noise for realism
- Ill-posed limited-aperture inversion behavior
