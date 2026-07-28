# ADR 058: Typed neuromodulation protocol quantities

- Status: accepted
- Date: 2026-07-27

## Context

`PulseTrainProtocol`, `PulseTrainDosimetry`, and `itrusst_assess` expose
frequency, duration, pressure, medium properties, intensity, total time, and
temperature rise as unit-documented `f64` values. The protocol is a public
cross-module contract and already has an Aequitas dependency.

## Decision

Use Aequitas `Frequency`, `Time`, `Pressure`, `MassDensity`, `Velocity`,
`Intensity`, and `TemperatureDifference` at the public protocol and dosimetry
boundaries. Rename unit-suffixed fields and methods to their dimension-neutral
names and migrate all in-repository callers and value tests. Keep mechanical
index, duty cycle, safety booleans, and CEM43 clinical-model thresholds at
their existing dimensionless or consumer-semantic boundaries. Extract SI
scalars only inside the pressure-intensity, mechanical-index, FDA-limit, and
pulse-phase formulas.

## Consequences

This is a pre-release public breaking change. The protocol retains the same
Blackmore/ITRUSST formulas and value semantics while invalid unit combinations
become unrepresentable at callers.

## Verification

The protocol reference tests and doctest cover nested duty-cycle definitions,
pulse-envelope boundaries, published theta-burst dosimetry, FDA screening,
and ITRUSST threshold branches. Focused physics package check, Nextest,
doctests, warning-denied Clippy, Rustdoc, and leaf formatting verify the
implementation.
