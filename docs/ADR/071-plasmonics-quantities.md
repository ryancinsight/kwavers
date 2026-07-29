# ADR 071: Aequitas contracts for plasmonics metrics

## Status

Accepted and implemented under `KWAVERS-AEQ-MET-33`.

## Context

The public plasmonics APIs carried nanoparticle radii, optical wavelengths,
positions, lattice spacing, concentration, cross-sections, and resonance
frequency as raw SI scalars. Mie polarizability was returned as a bare complex
value even though its SI unit is farad-square-metre. The same physical
boundaries appeared in both the standalone Mie/array calculators and the
electromagnetic equation trait, so typing only one surface would leave a
parallel untyped API.

## Decision

Use Aequitas `Length`, `NumberDensity`, `Area`, `Frequency`,
`Polarizability`, and `ReciprocalVolume` at the public boundaries. Relative
dielectric functions, volume fractions, local/effective-medium dielectric
values, field enhancement, and Purcell factors remain dimensionless. The
Johnson-Christy, Drude, Mie, depolarization, and dense-array kernels extract
base scalars only at their formula boundaries.

`Polarizability<eunomia::Complex64>` carries both phasor components under one
`FaradSquareMeter` unit. Eunomia owns the complex representation and its
linear real-factor scaling; no imaginary unit or consumer-local wrapper is
introduced. `ReciprocalVolume` is distinct from entity-count `NumberDensity`
despite sharing the same SI exponents, preventing semantic substitution of a
geometric coupling coefficient for a particle population.

## Alternatives rejected

- Keep raw plasmonics scalars: rejected because public callers could mix metres,
  nanometres, and optical frequency units without a type error.
- Return complex polarizability without a dimension: rejected because the
  polarizability is a physical SI metric, not a dimensionless numerical
  intermediate.
- Reuse `NumberDensity` for near-field coupling: rejected because equal
  exponents do not make entity-count and geometric reciprocal-volume semantics
  interchangeable.
- Model the imaginary polarizability component as a separate unit: rejected
  because it is quadrature data for the same physical phasor.

## Verification

The provider tests cover complex polarizability unit round-trip and semantic
dimension registration. Kwavers plasmonics tests retain the Johnson-Christy
interpolation, cross-section conservation, enhancement, effective-medium
closed forms, and coherent hot-spot oracles after the typed migration. The
package check, warning-denied Clippy, targeted rustfmt, and residue scans are
The Aequitas provider check and focused complex-unit nextest pass. The delivered
Kwavers revision passes package check, focused nextest (10/10), warning-denied
Clippy, targeted rustfmt, and the raw-signature residue scan.
