# ADR 098 — Aequitas PAM and neural sensor quantities

Status: Accepted — 2026-08-03

## Context

Passive-acoustic mapping and neural traditional delay-and-sum beamforming
still expose physical values as raw scalars. PAM configuration carries sound
speed, sampling and band frequencies, spatial resolution, focal coordinates,
and integration time. Detected events carry coordinates, occurrence time, and
optional peak frequency. Neural sensor geometry carries positions, pitch,
sampling frequency, and sound speed; steering angles feed a delay formula.

The PAM signal arrays are uncalibrated `f64` representation data. Their
integrated spectrum and event intensity therefore have no defensible SI unit
until a calibration contract is added. A noise-floor multiplier is
dimensionless by definition and is not the same quantity as the signal
threshold.

## Decision

Use Aequitas `Length<f64>` for sensor coordinates, pitch, focal point, and
spatial resolution; `Velocity<f64>` for sound speed; `Frequency<f64>` for
sampling, frequency-band endpoints, and peak frequency; and `Time<f64>` for
event occurrence and integration time. Use `Angle<f64>` at the neural DAS
steering boundary. Convert to base scalars only for trigonometric functions,
sample-index/delay arithmetic, FFT-bin mapping, and dense Leto storage.

Keep PAM `detection_threshold` dimensionless as a noise-floor multiplier. Keep
the uncalibrated PAM-wide `threshold` and event `intensity` at their explicit
signal-representation boundary until a calibration contract establishes a
physical observable; assigning `Intensity` now would misrepresent arbitrary
signal magnitudes.

## Eunomia compatibility

Eunomia `Complex` values at FFT or steering boundaries represent real and
quadrature components of one observable signal. The quadrature component is
not an imaginary SI dimension. No complex-valued Aequitas physical quantity or
compatibility wrapper is introduced.

## Verification

The slice requires analytical linear/phased-array geometry and delay
regressions, invalid-input tests, locked affected-package checks, focused
Nextest, warning-denied Clippy, doctests, Rustdoc, raw-public-signature and
complex-boundary scans, and the hosted repository-owned matrix.
