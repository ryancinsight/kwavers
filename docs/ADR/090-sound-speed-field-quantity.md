# ADR 090 — Aequitas sound-speed field quantity

- Status: Proposed
- Date: 2026-07-31
- Owner: `KWAVERS-AEQ-MET-51`

## Context

Sound-speed-shift reconstruction and benchmark results still expose dense
Leto `Array2<f64>` values through public fields named with the `m_s` unit
suffix. The numerical solver and Leto provider require real scalar storage,
but the public result contract should preserve the velocity meaning instead
of exposing a dimensionless array.

## Decision

Introduce the public `SoundSpeedShiftField` value type. It owns the Leto
`Array2<f64>` storage, exposes provider storage only through an explicit
storage accessor, and exposes velocity-valued iteration through Aequitas
`Velocity`. Reconstruction images, streaming views, batch frames, and OpenPros
truth fields use this type and unit-free field names. Solver kernels extract
the provider array only at their Leto boundary.

The field is real-valued. Eunomia complex scalar support remains available to
numerical providers that carry complex data, but a real sound-speed shift has
no imaginary physical component and does not receive a complex unit.

## Alternatives rejected

- Public `Array2<f64>`: leaves the physical unit implicit at the result seam.
- `Array2<Velocity>` in Leto kernels: couples the provider numerical element
  type to the domain quantity wrapper and breaks the existing solver seam.
- Complex-valued sound-speed field: misrepresents the real inversion result.

## Verification

The diagnostics test-target check, focused sound-speed-shift Nextest,
warning-denied all-target Clippy, doctests, RustDoc, formatting, diff checks,
and a public-contract scan will pass before acceptance.
