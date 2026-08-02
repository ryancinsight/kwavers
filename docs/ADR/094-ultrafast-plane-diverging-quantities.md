# ADR 094 — Aequitas ultrafast plane and diverging-wave quantities

Status: Accepted — 2026-08-02

## Context

After the scheduler migration in ADR 093, the adjacent ultrafast plane-wave
and diverging-wave APIs still exposed SI-valued `f64` geometry, sound speed,
frequency, angle, delay, and dimensionless aperture metrics. This allowed
metres, seconds, hertz, and radians to cross public contracts without a type
check. Image-coordinate arrays had the same problem even though their dense
delay outputs are intentionally stored as `f64` Leto arrays for kernel
consumption.

## Decision

Use Aequitas quantities at the public ultrafast geometry boundaries:

- `Length` for element positions, virtual-source depth, and image coordinates;
- `Velocity` for sound speed;
- `Frequency` for sampling and PRF metrics;
- `Angle` for plane-wave tilt angles;
- `Time` for scalar transmit, receive, and STA delays;
- `Dimensionless` for F-number and scalar Hann weights.

Formula code extracts base-unit scalars only at the numerical boundary. Dense
Leto delay and apodization arrays remain scalar storage outputs because their
consumers require contiguous numeric buffers; their units are documented as
seconds or dimensionless weights. The former `angles_degrees` conversion API
is removed so callers retain the typed angle contract. No compatibility
wrapper is retained; all in-repository callers and tests use the typed APIs.
Invalid apodization indices and invalid PRF inputs return typed errors instead
of panicking or constructing non-finite metrics.

The implemented equations remain unchanged:

```text
τ_tx = (sqrt((x - x_i)^2 + (z + F)^2) - F) / c
τ_rx = sqrt((x - x_j)^2 + z^2) / c
τ_sta = τ_tx + τ_rx
PRF_max = c / (2 z_max)
```

## Eunomia compatibility

These geometry and timing metrics are real-valued SI quantities and do not
introduce an imaginary physical unit. If Eunomia supplies a complex phasor,
its real observable is formed at the numerical boundary before entering these
APIs; real and imaginary components share the phasor's existing physical unit.

## Verification

The focused `kwavers-transducer` package check and Nextest cover all ultrafast
tests, including analytical delay, PRF, symmetry, apodization, typed image
coordinates, and invalid-index behavior. The exact locked package check,
Nextest `f1d0db2a-5e11-450b-831e-a4290847d6ee` (219/219, one ignored), including
invalid-index apodization and invalid-depth PRF regressions, package
Clippy with `-D warnings`, one executable doctest with six ignored, package
Rustdoc, targeted Rustfmt, `git diff --check`, and typed/complex residue scans
pass. `cargo-semver-checks 0.48.0` cannot compare the package because it is not
published to crates.io; the command fails before comparison with
`kwavers-transducer not found in registry`. Hosted API review is the available
public-surface review evidence.
