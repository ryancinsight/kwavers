# ADR 096 — Aequitas beamforming configuration quantities

Status: Accepted — 2026-08-02

## Context

The shared `BeamformingCoreConfig` contract exposes sound speed, sampling
frequency, and reference frequency as raw `f64` values. These values cross the
public configuration boundary and feed delay, signal-axis, and array-design
formulas in the beamforming processor.

## Decision

Use Aequitas `Velocity<f64>` for sound speed and `Frequency<f64>` for sampling
and reference frequencies. Convert to base scalars only inside the numerical
formulas that require scalar array indexing or trigonometric evaluation.
Migrate all in-repository callers in the same change without compatibility
wrappers.

## Eunomia compatibility

Complex beamforming buffers remain representation data under their existing
observable physical unit. Real and quadrature components share that unit; no
imaginary SI unit is introduced. Reduction to a real observable occurs only at
an explicit numerical or reporting boundary.

## Verification

The slice requires value-semantic configuration and processor regressions,
locked affected-package checks, focused Nextest, warning-denied Clippy,
doctests, Rustdoc, raw-physical-signature and complex-boundary scans, and
hosted repository-owned gates.
