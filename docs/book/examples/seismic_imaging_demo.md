# Example: Seismic Imaging Demo

**Crate**: `kwavers`  
**Run**: `cargo run -p kwavers --example seismic_imaging_demo`  
**Source**: [`crates/kwavers/examples/seismic_imaging_demo.rs`](../../../../crates/kwavers/examples/seismic_imaging_demo.rs)

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
let domain: ElectromagneticDomain<Backend> = ElectromagneticDomain::new(
    EMProblemType::Electrostatic,
    8.854e-12,                   // Vacuum permittivity
    4e-7 * std::f64::consts::PI, // Vacuum permeability
    0.0,                         // No conductivity
    vec![0.01, 0.01],            // 1cm x 1cm domain
)
.add_pec_boundary(BoundaryPosition::Top)
.add_pec_boundary(BoundaryPosition::Bottom)
.add_pec_boundary(BoundaryPosition::Left)
.add_pec_boundary(BoundaryPosition::Right);

let geometry = UniversalSolverGeometry2D::rectangle(0.0, 0.01, 0.0, 0.01);
```

## FWI Objective

```text
J(c) = (dt / 2) Σ_{r,t} [d_syn(r,t; c) − d_obs(r,t)]²

∂J/∂m(x) = −∫₀ᵀ λ(x, T−t) ∂²p(x,t)/∂t² dt,   m = c⁻²
∂J/∂c(x) = −2 c(x)⁻³ ∂J/∂m(x)
```

## Book Chapter

[← Transcranial Ultrasound: Physics, Aberration Correction, and Therapy](../transcranial_ultrasound.md)
