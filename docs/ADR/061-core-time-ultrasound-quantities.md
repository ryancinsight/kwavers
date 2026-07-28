# ADR 061: Type Core Time and Ultrasound Configuration Quantities

## Status

Accepted and implemented in `KWAVERS-AEQ-MET-23`.

## Context

`kwavers-core::time::Time` and `StabilityConstraints` exposed time steps,
durations, grid lengths, wave speeds, and thermal diffusivity as raw scalars.
`kwavers-imaging::UltrasoundConfig` likewise exposed center and sampling
frequencies as raw scalars. These public contracts allowed incompatible units
to cross the core and imaging boundaries without a dimensional check.

## Decision

Use the existing Aequitas provider quantities at the public boundary:

- `Time` and `StabilityConstraints` use `Time`, `Length`, `Velocity`, and
  `ThermalDiffusivity`.
- `UltrasoundConfig` uses `Frequency` for center and sampling frequency.
- CFL numbers, dynamic range, and dense numerical time vectors remain scalar
  because they are dimensionless or explicit numerical-array boundaries.

Scalar extraction is confined to CFL, diffusion, and array-generation
formulas. No scalar compatibility facade is retained.

## Consequences

This is a pre-release breaking change for callers constructing core time or
imaging ultrasound configurations. The API now rejects dimensional mixups at
compile time while preserving the existing numerical formulas and array
storage boundaries.

## Verification

The affected package suite passes 133/133 Nextest tests. Core doctests pass
3/3 and imaging doctests pass 4/4. Warning-denied Clippy and Rustdoc pass for
both packages; touched-file rustfmt and `git diff --check` pass.
