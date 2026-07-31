# ADR 089 — Aequitas sound-speed error quantities

- Status: Proposed
- Date: 2026-07-31
- Owner: `KWAVERS-AEQ-MET-50`

## Context

The OpenPros benchmark exposes mean absolute error and root mean square error
for reconstructed sound-speed shifts as unit-suffixed raw scalars. These
values are velocity errors and remain physical quantities after the numerical
image comparison. The normalized error, correlation, objective values, and
regularization weights have different numerical contracts and must not be
given a velocity unit.

## Decision

Use Aequitas `Velocity` for the public MAE and RMSE fields, removing unit
suffixes from their names. Compute image differences as provider-native
scalars inside the numerical comparison, then construct `Velocity` at the
benchmark metric boundary. Keep NRMSE, Pearson correlation, objective values,
weights, and Leto `Array2<f64>` image storage as explicit dimensionless,
formula, or provider-storage boundaries.

This real-valued workflow introduces no physical phasor or imaginary physical
unit. Eunomia complex scalar support remains available for genuinely complex
numerical fields, but is not applicable to a real sound-speed error metric.

## Alternatives rejected

- Raw MAE/RMSE fields: retain dimensional ambiguity at the public benchmark
  contract.
- Velocity-typed Leto arrays: couple the provider storage and solver kernels to
  a domain quantity wrapper instead of preserving the existing storage seam.
- Complex velocity errors: misrepresent the real benchmark residual.

## Verification

The diagnostics test-target check, focused sound-speed-shift Nextest,
warning-denied all-target Clippy, doctests, RustDoc, formatting, diff checks,
and a public-contract scan will pass before acceptance.
