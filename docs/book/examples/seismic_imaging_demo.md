# Example: Seismic Imaging Demo

**Crate**: `kwavers`  
**Run**: `cargo run -p kwavers --example seismic_imaging_demo`  
**Source**: [`crates/kwavers/examples/seismic_imaging_demo.rs`](../../../crates/kwavers/examples/seismic_imaging_demo.rs)

## What This Example Demonstrates

This example demonstrates full-waveform inversion (FWI) for seismic imaging through a realistic skull phantom. It shows the complete imaging pipeline from CT phantom to reconstructed brain image.

## Physical Pipeline

```text
Skull CT phantom  →  c(x), ρ(x)  →  FDTD forward  →  synthetic traces
                                                              │
                              ← adjoint source ←  L2 residual
                              │
                              FDTD adjoint (time-reversed, back-propagated)
                              │
                              gradient ∂J/∂c  →  model update  →  brain image
```

## Skull Phantom Geometry

The phantom is a coronal cross-section of a human head modelled as concentric shells:

```text
┌─────────────────────────────────────────────────────┐
│               water coupling bath                   │
│         ┌─────────────────────────┐                 │
│         │   scalp  (HU ≈  40)    │                  │
│         │  ┌─────────────────┐   │                  │
│         │  │  outer cortical │   │                  │
│         │  │  bone (HU≈720) │   │  ← z (depth)     │
│         │  │  ┌───────────┐  │   │                  │
│         │  │  │  diploe   │  │   │                  │
│         │  │  │ (HU≈380) │  │   │                  │
│         │  │  │ ┌───────┐ │  │   │                  │
│         │  │  │ │ inner │ │  │   │                  │
│  SRC    │  │  │ │ cort. │ │  │  RECV               │
│  (left  │  │  │ │┌─────┐│ │  │  (right             │
│  arc)   │  │  │ ││brain││ │  │   arc)              │
│         │  │  │ │└─────┘│ │  │                     │
└─────────┴──┴──┴─┴───────┴─┴──┴─────────────────────┘
            ↑ x (lateral, left→right)
```

## Full-Ring Acquisition Geometry

16 active element locations uniformly distributed around a full ring at R_ARRAY = 20 voxels from the grid centre. 8 transmit in sequence while 15 act as receivers. Full-ring coverage eliminates shadow zones.

## Key Code Snippet

```rust
let grid = Grid::new(NX, NY, NZ, DX, DX, DX)?;
let fwi = FwiProcessor::new(FwiParameters {
    max_iterations: 1,
    frequency: F0_HZ,
    nt: nt_fine,
    dt,
    n_trace: N_RECEIVERS,
    n_depth: 1,
    step_size: STEP_SIZE,
    ..FwiParameters::default()
});

for &element_index in &TRANSMIT_ELEMENT_INDICES {
    let geometry = build_shot(element_index, F0_HZ, nt_fine, dt);
    let observed = fwi.generate_synthetic_data(&true_model, &geometry, &grid)?;
    shots.push((geometry, observed));
}
```

## FWI Objective

```text
J(c) = (dt / 2) Σ_{r,t} [d_syn(r,t; c) − d_obs(r,t)]²

∂J/∂m(x) = −∫₀ᵀ λ(x, T−t) ∂²p(x,t)/∂t² dt,   m = c⁻²
∂J/∂c(x) = −2 c(x)⁻³ ∂J/∂m(x)
```

## Book Chapter

[← Transcranial Ultrasound: Physics, Aberration Correction, and Therapy](../transcranial_ultrasound.md)
