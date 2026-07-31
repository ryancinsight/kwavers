# ADR 083: Aequitas f-k migration quantities

- **Status:** Proposed
- **Date:** 2026-07-31
- **Decision owner:** `KWAVERS-AEQ-MET-44`
- **Scope:** `kwavers-diagnostics::workflows::fk_migration`

## Context

The Stolt f-k migration entrypoint accepts lateral sample spacing, temporal
sample interval, and sound speed as raw `f64` values even though those values
define the physical wavenumber, frequency, and depth grid. Zero or non-finite
values also reach the FFT-bin formulas without a typed validation boundary.

## Decision

Accept Aequitas `Length`, `Time`, and `Velocity` for spacing, sample interval,
and sound speed. Convert to coherent SI scalars only inside the FFT and
dispersion formulas. Return `KwaversResult<Array2<f64>>` so non-finite or
non-positive physical inputs fail before FFT allocation. Keep RF samples,
complex FFT storage, and scalar bin calculations at their explicit numerical
boundaries.

The workflow is real-valued. Complex FFT samples are an internal numerical
representation, not a physical phasor contract, so no imaginary physical unit
is introduced and Eunomia complex-unit behavior is unchanged.

## Alternatives

- Retain raw `f64` inputs: rejected because the public migration contract
  remains unit-ambiguous and invalid spacings can enter the formulas.
- Add consumer-local spacing or velocity wrappers: rejected because Aequitas
  owns the shared physical quantity vocabulary.
- Return a zero image for invalid inputs: rejected because it masks invalid
  physics and makes failure indistinguishable from a valid empty result.

## Verification plan

- Run package test-target check and the focused f-k migration Nextest filter.
- Assert typed parameter conversion, reflector depth/focus behavior, and
  invalid-input errors with value-semantic tests.
- Run warning-denied Clippy, doctests, RustDoc, formatting, and diff checks.
- Scan the public signature and document the real/complex boundary.

