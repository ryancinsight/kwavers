# ADR 086 — Aequitas sound-speed-shift time quantities

- Status: Proposed
- Date: 2026-07-31
- Owner: `KWAVERS-AEQ-MET-47`

## Context

Sound-speed-shift tomography accepted measured differential travel-time shifts
as raw `f64` values with `_s` names. The same scalar vectors crossed the
curved-array, prediction, fixed-acquisition, and batch APIs, while numerical
operators multiplied them by the reference sound speed. This left seconds
ambiguous at every public boundary and made unit mistakes possible when
combining acquisitions.

## Decision

Use Aequitas `Time` for `SoundSpeedShiftSample` measurements, predicted
travel-time vectors, curved-array scan inputs, fixed-acquisition frame APIs,
and retained batch frame APIs. Rename public time-shift fields and arguments to
remove unit suffixes. Convert to base seconds only inside the path-integral
operator and validation formulas.

Reference sound speed, grid spacing, curved-array geometry, and dense
speed-shift image arrays remain explicit boundaries for the next migration
slice. They are not silently wrapped in local scalar adapters. The workflow is
real-valued and contains no physical phasor, so Eunomia compatibility requires
no imaginary physical unit.

## Alternatives rejected

- Raw seconds: preserve the current unit ambiguity.
- Local time wrappers: duplicate Aequitas and split provider ownership.
- A scalar compatibility overload: retain the prohibited dual contract.
- Complex or imaginary time: misrepresent real travel-time measurements.

## Verification

The implementation will pass the diagnostics test-target check, focused
sound-speed-shift Nextest, warning-denied all-target Clippy, doctests, RustDoc,
formatting, diff checks, and a public-contract scan for unit-suffixed raw time
shifts. Value-semantic tests will cover sample construction, curved-array row
ordering, prediction, fixed-acquisition reconstruction, and batch streaming.
