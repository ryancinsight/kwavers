# ADR 088 — Aequitas curved-array and benchmark frequency quantities

- Status: Accepted
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

The diagnostics test-target check passes. Focused sound-speed-shift Nextest
run `9544d32a-ea02-4c1f-b57a-1e4944a68a30` passes 34/34 tests with 165
skipped; warning-denied all-target Clippy, doctests, RustDoc, package
formatting, diff checks, and the public-contract scan pass. The remaining raw
values are solver-owned coordinates, dense Leto storage, and benchmark
error-metric storage, each retained as an explicit numerical/storage boundary.
Workspace-wide rustfmt remains blocked by the Windows filename-length limit;
package formatting passes.
