# ADR 085 — Aequitas real-time SIRT quantities

- Status: Accepted
- Date: 2026-07-31
- Owner: `KWAVERS-AEQ-MET-46`

## Context

The public `RealTimeSirtConfig` and `ReconstructionFrame` contracts represented
frame rate, computation budgets, frame timestamps, and elapsed computation
time as raw floating-point values with unit-bearing names. Convergence error
and frame quality values were also unlabelled even though they are ratios or
logarithmic ratios.

## Decision

Use Aequitas `Frequency` for target and measured frame rate, `Time` for frame
timestamps, per-frame computation time, and the maximum computation budget,
and `Dimensionless` for convergence error, SNR, artifact level, spatial
smoothness, and the numerical dynamic-range metric. Rename public fields to
remove unit suffixes and extract base scalars only at elapsed-time and
formula boundaries.

RF/image arrays remain explicit numerical storage boundaries. Output
smoothing sigma is measured in grid points, and the intensity threshold has
the same unspecified numerical amplitude domain as the image; neither is
invented as a physical Aequitas unit. The workflow is real-valued. Its
numerical arrays do not represent physical phasors, so Eunomia compatibility
requires no imaginary physical unit.

## Alternatives rejected

- Raw `f64` fields: preserve unit ambiguity and suffix-driven API semantics.
- Local time/rate wrappers: duplicate Aequitas provider semantics.
- Treating RF/image amplitudes as pressure or intensity: invents a physical
  dimension not declared by this reconstruction contract.
- Introducing complex or imaginary units: misrepresents a real-valued SIRT
  and quality path.

## Verification

The implementation passes the diagnostics test-target check and focused
real-time-SIRT Nextest run `4b47f3d4-45c5-4ac3-9ca5-7745296e063e`:
14/14 tests passed with 185 skipped. Warning-denied all-target Clippy,
diagnostics doctests (1 executable, 5 ignored), RustDoc, package formatting,
and diff checks pass. The public-contract scan finds no unit-bearing raw
scalar in the real-time SIRT public configuration, frame, or quality fields.
Value-semantic tests cover typed defaults, configuration profiles, frame
timing/rate values, convergence error, and quality metrics.
