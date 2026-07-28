# ADR 059: Typed transducer design and propagation quantities

- Status: accepted
- Date: 2026-07-27

## Context

`ApertureDesignSpec`, `ArrayDesign`, and
`FocusedLinearArrayPropagationSpec` exposed lengths, frequency, sound speed,
current, pressure-per-current, and acoustic impedance as unit-documented raw
scalars. `FocusedPressureMap` also returned focal pressure, intensity, and
beam extents as raw values. This left the array-design and propagation
boundary weaker than the already typed transducer physics modules.

## Decision

Use Aequitas `Length`, `Frequency`, `Velocity`, `ElectricCurrent`,
`PressurePerElectricCurrent`, `AcousticImpedance`, `Pressure`, and `Intensity`
at the public design and propagation boundaries. Rename unit-suffixed fields
and methods to dimension-neutral names and migrate all in-repository callers
and value tests. Keep scalar extraction inside wavelength, phase, pressure,
intensity, width-search, validation, and grid/report conversion boundaries.
The driver may convert typed propagation results to its established report
DTO units only at that explicit presentation boundary.

## Consequences

This is a pre-release public breaking change. Array geometry and focused
propagation retain their numerical behavior while incompatible physical units
become unrepresentable at callers.

## Verification

The transducer package check and Nextest suite pass 218/218 with one skipped
test; its doctests pass 2/2 with six ignored. The driver `kwavers` feature
check and Nextest suite pass 489/489; its doctest target has no tests. Both
packages pass warning-denied Clippy and Rustdoc after the design-doc link
correction; touched Rust files pass formatting and diff checks.
