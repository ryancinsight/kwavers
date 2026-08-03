# ADR 097 — Aequitas sensor-beamformer quantities

Status: Accepted — 2026-08-03

## Context

The public `SensorBeamformer` contract exposes sensor positions, sampling and
steering frequencies, sound speed, and steering angles as raw scalar values.
`SensorProcessingParams` also exposes element spacing, array aperture, focal
length, F-number, and spatial-Nyquist frequency without dimensional types.
Those values cross the public boundary and feed delay, direction, phase, and
array-sampling formulas.

## Decision

Use Aequitas `Length<f64>` for each sensor coordinate, `Frequency<f64>` for
sampling and steering frequency, `Angle<f64>` for both spherical angles,
`Velocity<f64>` for sound speed, and `Length<f64>` for focal length. Return
F-number as `Dimensionless<f64>` and the spatial-Nyquist result as
`Frequency<f64>`. Convert to base scalars only at Euclidean, trigonometric,
phase, or dense numeric-buffer formula boundaries. Migrate all in-repository
callers in the same change without compatibility wrappers.

## Eunomia compatibility

The steering matrix remains `eunomia::Complex<f64>` representation data. Real
and quadrature components share the same observable signal unit; an imaginary
SI unit is not a physical dimension. Any reduction to a real observable stays
at an explicit numerical or reporting boundary.

## Verification

The slice requires analytical delay/phase and derived-metric regressions,
locked affected-package checks, focused Nextest, warning-denied Clippy,
doctests, Rustdoc, raw-public-signature and complex-boundary scans, and the
hosted repository-owned matrix.
