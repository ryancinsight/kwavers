# ADR 104: Thermal-acoustic coupling quantities

- Status: Accepted
- Date: 2026-08-05
- Driver: `KWAVERS-AEQ-MET-67`

## Context

The thermal-acoustic coupling module exposed temperature coefficient slopes,
absorption, intensity, pressure, velocity, density, frequency, elapsed time,
and accumulated energy density as raw `f64` values. The nonlinear heating
formula was also returned as an untyped power value even though its existing
law is `(B/A)·P²·ω²/(ρ·c³)`.

Dimensional analysis gives `kg·m⁻²·s⁻³`, equivalent to `W/m⁴`: a spatial
gradient of volumetric power density.

## Decision

Carry physical inputs and outputs through the public coupling contracts as
Aequitas quantities. Use provider-owned temperature-slope dimensions for
sound speed, density, and absorption. Return nonlinear heating as
`VolumetricPowerDensityGradient<f64>` with coherent `W/m⁴` units. Keep Celsius
grid arrays and Leto storage scalar because they are numerical storage
boundaries; convert once at those boundaries.

The Eunomia boundary remains real-valued for these SI quantities. Complex or
quadrature values in adjacent signal domains continue to share one existing
observable unit. No imaginary SI temperature, coefficient, heating, or
gradient unit is introduced.

## Alternatives rejected

- Retain raw scalars: rejected because unrelated physical quantities can be
  transposed at constructors and updates.
- Keep the nonlinear result named `power`: rejected because it hides the
  `W/m⁴` contract and permits downstream dimensional misuse.
- Add local Kwavers dimensions or scalar adapters: rejected because Aequitas
  owns shared dimensional vocabulary and compatibility wrappers preserve the
  obsolete boundary.

## Verification

The implementation compiles against merged Aequitas `3c51a27`. CI-profile
Nextest passes 1,550/1,550 tests with one configured skip; the downstream
`kwavers` package passes 530/530 CI-profile Nextest tests. Strict package
Clippy, doctests, Rustdoc, formatting, and diff checks pass. The clean lock
resolves Ritk `cfeebc7`, Eunomia `0.8.0`, and rkyv `0.8.17` only; the hosted
exact-head matrix remains the delivery gate.
