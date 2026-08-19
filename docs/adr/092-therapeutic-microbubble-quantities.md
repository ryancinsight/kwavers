# ADR 092 — Aequitas therapeutic microbubble quantities

Status: Accepted — 2026-08-01

## Context

The therapeutic microbubble domain exposed SI-valued `f64` fields for radius,
position, velocity, acceleration, pressure, amount of substance, shell
properties, energy, drug mass, time, radiation force, and streaming velocity.
The Keller–Miksis adapter and drug-release model need scalar inputs, but those
formula boundaries do not justify untyped public contracts.

## Decision

Use Aequitas quantities in the public Kwavers microbubble state, Marmottant
shell, radiation-force, streaming-velocity, field-sampling, and dynamics
contracts. The vector contracts use named `Position3D`, `Velocity3D`,
`PressureGradient3D`, and `Direction3D` value objects so each component keeps
its physical dimension.

The scalar boundary is explicit and one-way:

- typed quantities enter Keller–Miksis, finite-difference, drug-release, and
  other numerical formulas through `into_base()`;
- solver-owned scalar state, dense Leto arrays, and dimensionless empirical
  coefficients remain numerical/storage boundaries;
- formula results are reconstructed into typed quantities before returning to
  the public domain contract.

Aequitas owns the shared SI vocabulary, including `Acceleration` and
`PressureRate`; Kwavers does not define consumer-local unit wrappers.

## Eunomia compatibility

This therapeutic state is real-valued and ordered, so it does not gain an
imaginary physical unit. Eunomia `Complex<T>` remains valid for a genuine
complex acoustic phasor at an existing dimensional boundary. If a future
complex phasor drives this model, its Hermitian magnitude or other specified
real observable is computed at the numerical boundary before entering the
real-valued force, energy, or state metrics.

## Verification

The provider dimensional laws cover `Velocity / Time = Acceleration` and
`Pressure / Time = PressureRate`. Kwavers microbubble tests cover typed state
construction, shell transitions and piecewise surface tension, energy and
resonance formulas, Bjerknes and drag forces, and streaming behavior. The
remaining Keller–Miksis and drug-payload scalars are explicit formula or
storage boundaries rather than public SI metric fields.
