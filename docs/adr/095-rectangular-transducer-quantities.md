# ADR 095 — Aequitas rectangular-transducer quantities

Status: Accepted — 2026-08-02

## Context

The public `RectangularTransducer` contract represented width, height, and
center frequency as untyped `f64` values. Its wavenumber method accepted a raw
sound speed, and element-size calculation divided by public element counts
without an invalid-count check. The fast-nearfield solver repeated the same
untyped speed and density boundary.

## Decision

Use Aequitas quantities at the public contracts:

- `Length` for transducer width, height, and element sizes;
- `Frequency` for center frequency;
- `Velocity` for medium sound speed;
- `MassDensity` for medium density;
- `ReciprocalLength` for wavenumber.

`element_size` and `wavenumber` return `KwaversResult` and reject zero or
unrepresentable element counts, non-finite or non-positive dimensions and
frequency, and invalid sound speed. Scalar extraction is limited to the
Green-function, FFT, and direct-sum numerical formulas. All in-repository
callers are migrated without compatibility wrappers.

## Eunomia compatibility

The fast-nearfield complex pressure fields remain Eunomia-compatible numeric
buffers. Real and quadrature components share the field's existing physical
unit; they do not create an imaginary physical unit. A real observable is
formed at a numerical or reporting boundary when required.

## Verification

The exact locked `kwavers-transducer`, `kwavers-solver`, and
`kwavers-simulation` package checks pass. Focused Nextest passes 2/2 for the
transducer contract (`8e15dcb4-76e5-4ef3-9768-0e9051705be4`) and 6/6 for FNM
(`3d2af317-dfce-4fae-99d7-74c61ca554d9`). Warning-denied Clippy passes for the
affected library targets and the `kwavers` benchmark target. Transducer and
solver doctests, formatting, and typed/complex residue scans pass. The
simulation doctest and FNM benchmark smoke commands each exceed the
300-second shared-target collection bound without a diagnostic; neither is
claimed as green. Rustdoc passes for all three affected packages. The
benchmark compiles as an all-target check. No runtime performance claim is
made by this contract migration.
