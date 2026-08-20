# Example: Seismic Imaging Demo

**Crate**: `kwavers`
**Run**: `cargo run -p kwavers --example seismic_imaging_demo`
**Source**: [`crates/kwavers/examples/seismic_imaging_demo.rs`](../../../crates/kwavers/examples/seismic_imaging_demo.rs)

## What This Example Demonstrates

Transcranial ultrasound full-waveform inversion (FWI) — brain reconstruction
from an explicitly selected synthetic or CT input. The complete pipeline is:
skull model → acoustic forward simulation → adjoint-state gradient → iterative
model update → brain image.

## Input mode

The default is the deterministic analytical phantom. Select a real CT input
without changing the workflow with
`KWAVERS_SEISMIC_INPUT_MODE=ct:<path>`. A failed explicit CT load is an error;
it never changes the run to a synthetic model. Outputs default below the
example's `output/` directory (override with the first positional argument).

## Physics

The consumer supplies only the CT Hounsfield-unit volume. The shared
`seismic_imaging::medium::SkullModel` sends it to
`HeterogeneousSkull::from_ct_hill` with Aequitas-typed canonical cortical-bone
properties. Density follows Voigt volume averaging, sound speed follows the
Hill average of the Voigt and Reuss bulk moduli, and attenuation is mixed at
the 1 MHz reference frequency. The 2D quasi-3D grid (NX=64, NY=2, NZ=64)
exercises the full 3D solver on a thin slab.

## Key Concepts

- Provider-owned skull CT mapping with the validated `AcousticSkullProperties`
  configuration SSOT
- Aequitas-typed frequency, time, and pressure passed to the provider-owned
  `DomainRickerWavelet` without an intermediate sample allocation
- Adjoint-state FWI with L2 misfit
- Multi-shot acquisition and gradient accumulation
