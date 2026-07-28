# ADR 060: Typed ultrafast transducer quantities

- Status: accepted
- Date: 2026-07-27

## Context

The ultrafast transducer stack exposed angles, pulse-repetition frequencies,
sound speed, element positions, depths, and durations as unit-documented raw
scalars. The prior metric audit therefore overstated closure: the sequencer,
plane-wave, and diverging-wave public contracts were still weaker than the
typed design and propagation modules.

## Decision

Use Aequitas `Angle`, `Frequency`, `Length`, `Time`, and `Velocity` in the
ultrafast sequencer events/schedules, plane-wave configuration and processor,
and diverging-wave configuration and processor. Rename no typed operation to a
unit-specific variant and do not retain scalar compatibility wrappers. Extract
base values only inside delay, apodization, frame-rate, and other numerical
formula boundaries. Keep `Array1<f64>`/`Array2<f64>`/`Array3<f64>` delay and
weight tables as scalar numerical-array outputs because those arrays are the
mesh/formula boundary rather than physical scalar contracts.

## Consequences

This is a pre-release public breaking change. Physical-unit mismatches at
ultrafast call sites are rejected by the type system while delay and weighting
algorithms retain their established numerical representation and behavior.

## Verification

`kwavers-transducer` package check and full Nextest pass with 218 tests passed
and one skipped test. Doctests pass 2/2 with six ignored. Warning-denied
Clippy, Rustdoc, rustfmt, and diff checks pass for the increment. Existing
shared-graph unused-patch and linker warnings are outside this ADR.
