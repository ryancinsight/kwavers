# Example: Field Surrogate Demo

**Crate**: `kwavers`
**Run**: `cargo run -p kwavers --example field_surrogate_demo`
**Source**: [`crates/kwavers/examples/field_surrogate_demo.rs`](../../../crates/kwavers/examples/field_surrogate_demo.rs)

## What This Example Demonstrates

Field-surrogate PINN training: builds a Penttinen-Gaussian kernel cube (4 corners: 0.5/1.0 MHz × 15/30 MPa) with realistic per-frequency focal-spot scaling, runs Coeus Adam + Helmholtz-residual training for 2000 steps, and exports training history and axial cross-sections for plotting.

## Outputs

- `target/field_surrogate_demo/training_history.csv` — per-step (data, Helmholtz, total) losses
- `target/field_surrogate_demo/axial_lines.csv` — prediction vs target cross-sections at held-out frequencies
