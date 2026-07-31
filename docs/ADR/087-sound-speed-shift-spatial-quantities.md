# ADR 087 — Aequitas sound-speed-shift spatial quantities

- Status: Proposed
- Date: 2026-07-31
- Owner: `KWAVERS-AEQ-MET-48`

## Context

The sound-speed-shift time contracts now use Aequitas `Time`, but the same
public configuration still represented reference sound speed, pixel spacing,
curved-path sagitta, and finite-frequency length scales as raw metres/metres
per second with unit-suffixed names. These values reach ray construction,
validation, and the forward model, so their dimensions remain implicit across
the configuration boundary.

## Decision

Use Aequitas `Velocity` for reference sound speed and `Length` for grid
spacing, curved-path sagitta, finite-frequency wavelength, and support radius.
Rename the public fields and enum members without unit suffixes. Extract base
scalars only in the ray, propagation, validation, and solver formulas that
require numerical coordinates or ratios.

Solver-owned `PlanarPoint` coordinates and dense Leto image storage remain
explicit numerical boundaries for later slices. This migration is
real-valued; it introduces no physical phasor and therefore no imaginary
physical unit for Eunomia compatibility.

## Alternatives rejected

- Raw spatial scalars: retain dimensional ambiguity at public boundaries.
- Local length/velocity wrappers: duplicate provider ownership.
- Complex or imaginary spatial quantities: misrepresent real geometry and
  reference speed.

## Verification

The implementation will pass the diagnostics test-target check, focused
sound-speed-shift Nextest, warning-denied all-target Clippy, doctests, RustDoc,
formatting, diff checks, and a public-contract scan for unit-suffixed raw
spatial configuration values.
