# ADR 054: Typed therapy-integration physical quantities

- Status: accepted
- Date: 2026-07-27

## Context

The therapy-integration public contracts represented session duration,
acoustic frequency, pressure, focal geometry, timestamps, intensity, and
thermal values as unit-documented `f64` fields. This allowed values with
different dimensions or unit conventions to cross configuration, orchestration,
safety, and reporting boundaries without compiler enforcement. The intensity
tracker also stored CEM43 state as an untyped numeric value even though
Kwavers already owns the validated `CumulativeEquivalentMinutes` contract.

## Decision

Use the existing Aequitas quantities directly in the therapy-integration
contracts:

- `Time` for durations, timestamps, intervals, and response windows;
- `Frequency` for acoustic frequency and PRF;
- `Pressure` for peak negative pressure;
- `Length` and `Volume` for focal geometry and treatment volume;
- `Intensity` for W/m² metrics;
- `ThermodynamicTemperature` for absolute temperatures;
- `TemperatureDifference` for temperature rises; and
- `CumulativeEquivalentMinutes` for CEM43 dose.

Dense Leto arrays remain scalar storage boundaries. Scalar extraction is
allowed only inside numerical formulas, mesh/index arithmetic, explicit unit
conversion helpers, and legacy kernels whose public contract is not yet typed.
Dimensionless thermal/mechanical/cavitation indices remain scalar.

## Alternatives rejected

- Keeping unit-suffixed scalar fields was rejected because names do not encode
  dimensional validity and permit mixed-unit callers.
- Adding therapy-owned wrapper types was rejected because Aequitas already
  owns the dimensions and wrappers would duplicate conversion and validation.
- Typing dense pressure and temperature arrays was rejected for this increment:
  their element representation is the storage/mesh boundary, not a scalar
  physical contract.

## Consequences

This is a pre-release public breaking change. Callers must construct typed
quantities and use explicit base-unit extraction only when entering a formula,
mesh calculation, or untyped legacy kernel. The public therapy path now shares
the same dimensional vocabulary as the physics and HIFU-planning paths.

## Verification

The implementation is covered by the `kwavers-therapy --tests` package check,
focused Nextest (349/349 passed, one skipped, four slow), doctests (8 passed,
one ignored), Rustdoc (exit 0), and warning-denied Clippy (exit 0) in a clean
linked lane. The shared Atlas overlay currently has a duplicate Aequitas
worktree-package collision that can block main-tree Clippy before compilation;
the linked lane verifies the package without changing peer-owned lockfiles or
overlay configuration.
