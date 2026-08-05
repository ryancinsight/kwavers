# ADR 104: Thermal-acoustic coupling quantities

- Status: Accepted
- Date: 2026-08-06
- Driver: `KWAVERS-AEQ-MET-67`

## Context

Thermal-acoustic coupling exposed temperature slopes, absorption, intensity,
pressure, velocity, density, frequency, elapsed time, and accumulated energy
density as raw scalars. The nonlinear heating law
`(B/A)·P²·ω²/(ρ·c³)` also returned an untyped power value. Dimensional analysis
places that result at `kg·m⁻²·s⁻³`, equivalent to `W/m⁴`, a spatial gradient
of volumetric power density.

## Decision

Carry the coupling boundary through Aequitas quantities. Temperature slopes use
the provider-owned `VelocityPerTemperature`, `MassDensityPerTemperature`, and
`ReciprocalLengthPerTemperature` dimensions. Nonlinear heating returns
`VolumetricPowerDensityGradient<f64>`. Celsius fields and Leto arrays remain
scalar numerical-storage boundaries, with conversion performed once at those
boundaries.

The Eunomia boundary is real-valued for these SI quantities. Complex or
quadrature signal values retain one existing observable unit. No imaginary SI
temperature, coefficient, heating, or gradient unit is introduced.

## Alternatives rejected

- Raw scalars were rejected because unrelated physical quantities can be
  transposed at constructors and updates.
- Naming the nonlinear result `power` was rejected because it hides the
  `W/m⁴` contract and permits dimensional misuse.
- Local Kwavers dimensions and scalar adapters were rejected because Aequitas
  owns the shared vocabulary and compatibility wrappers preserve the obsolete
  boundary.

## Verification

`cargo check -p kwavers-physics --lib --offline -j 1` passes on the active
branch, and the typed residue scan finds no raw temperature-coefficient fields
in the coupling contract. The attempted bounded native test collection was
terminated after exceeding the 300-second budget while peer builds occupied
the shared target and disk availability fell below 100 MB. The residual is
external to the coupling implementation.
