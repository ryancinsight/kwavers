# ADR 087 — Aequitas sound-speed-shift spatial quantities

- Status: Accepted
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

The diagnostics test-target check passes. Focused sound-speed-shift Nextest run
`963e5434-b270-4e26-9e42-abec4c4b646f` passes 34/34 with 165 skipped;
warning-denied all-target Clippy, one executable doctest with five ignored,
RustDoc, package formatting, diff checks, and the public-contract scan pass.
The scan leaves only declared solver-coordinate, dense image, and benchmark
error-metric storage boundaries.
