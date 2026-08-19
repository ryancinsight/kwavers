# ADR 093 — Aequitas ultrafast scheduler quantities

Status: Accepted — 2026-08-02

## Context

The public `kwavers-transducer::ultrafast::sequencer` contract exposed SI-valued
`f64` fields for sound speed, maximum depth, PRF, event times, frame rate, and
tilt angles. The scheduler's equations require scalar values, but callers also
need the units to remain explicit while constructing and consuming schedules.
The schedule indices and element identifiers are discrete indices, not physical
metrics.

## Decision

Use Aequitas `Velocity`, `Length`, `Frequency`, `Time`, and `Angle` in the
public scheduler state and schedule events. Convert to base-unit scalars only at
the PRF and timing formula boundaries:

- `PRF_max = c / (2 z_max)` extracts metres per second and metres;
- `f_frame = PRF / N` extracts hertz and the discrete event count;
- event times are reconstructed as Aequitas seconds after formula evaluation;
- angle ordering preserves `Angle` values without scalar round trips.

The scheduler rejects non-finite or non-positive physical configuration values
and rejects zero-angle schedules before constructing timing metrics. No consumer
wrapper or compatibility path is retained.

## Eunomia compatibility

Scheduler metrics are real-valued timing and geometry quantities. They do not
require an imaginary physical unit. If a future Eunomia complex phasor is used
to drive a scheduling or imaging observable, its specified real observable is
formed at the numerical boundary before entering these real-valued metrics;
both components of a complex representation retain the same existing physical
unit.

## Verification

The typed scheduler tests cover the analytical PRF limit, compound frame-rate
formula, sequential and interleaved event timing, flash and STA schedules,
angle preservation, and invalid geometry/empty-schedule rejection. The exact
standalone lock graph passes the package all-target check and Nextest
`7b7dfdab-3df2-4ed2-ad07-e5ce49e003dd` (218/218). Package Clippy with
`-D warnings`, doctests, Rustdoc, targeted Rustfmt, and the raw physical-field
and complex-unit residue scans also pass.
