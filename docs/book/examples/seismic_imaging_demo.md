# Example: Seismic Imaging Demo

**Crate**: `kwavers`
**Run**: `cargo run -p kwavers --example seismic_imaging_demo`
**Source**: [`crates/kwavers/examples/seismic_imaging_demo.rs`](../../../crates/kwavers/examples/seismic_imaging_demo.rs)

## What This Example Demonstrates

Transcranial ultrasound full-waveform inversion (FWI) — brain reconstruction from synthetic ultrasound data. Demonstrates the complete pipeline: skull CT phantom → acoustic forward simulation → adjoint-state gradient → iterative model update → brain image.

## Physics

Skull bone-volume-fraction acoustic model (Aubry 2003) with fractional-Laplacian absorption (Treeby & Cox 2010). The 2D quasi-3D grid (NX=64, NY=2, NZ=64) exercises the full 3D solver on a thin slab.

## Key Concepts

- Skull CT phantom construction with bone-volume-fraction → sound-speed mapping
- Adjoint-state FWI with L2 misfit
- Multi-shot acquisition and gradient accumulation
