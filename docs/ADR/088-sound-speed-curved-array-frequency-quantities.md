# ADR 088 — Aequitas curved-array and benchmark frequency quantities

- Status: Proposed
- Date: 2026-07-31
- Owner: `KWAVERS-AEQ-MET-49`

## Context

Sound-speed-shift spatial scales now use Aequitas, but public curved-array
geometry still exposed radius and angles as raw unit-suffixed scalars. The
OpenPros waveform expectation also exposed peak frequency as raw hertz. These
values cross constructors and results before reaching coordinate and
wavelength formulas.

## Decision

Use Aequitas `Length` for curved-array radius, `Angle` for first angle,
angular pitch, and aperture result, and `Frequency` for OpenPros peak
frequency. Rename public fields and methods without unit suffixes. Extract
base scalars only for trigonometry, validation, and the derived wavelength
formula.

Solver-owned `PlanarPoint` coordinates, dense Leto images, and benchmark
error-metric storage remain explicit boundaries for later slices. This work
is real-valued and introduces no physical phasor or imaginary physical unit
for Eunomia compatibility.

## Alternatives rejected

- Raw radius/angles/frequency: retain dimensional ambiguity at public seams.
- Local wrappers: duplicate Aequitas ownership.
- Complex or imaginary geometry/frequency: misrepresent the real acquisition.

## Verification

The implementation will pass the diagnostics test-target check, focused
sound-speed-shift Nextest, warning-denied all-target Clippy, doctests, RustDoc,
formatting, diff checks, and a public-contract scan for raw curved-array and
waveform frequency values.
