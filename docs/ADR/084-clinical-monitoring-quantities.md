# ADR 084 — Aequitas clinical monitoring quantities

- Status: Proposed
- Date: 2026-07-31
- Owner: `KWAVERS-AEQ-MET-45`

## Context

The public clinical-monitoring reconstruction contract represented processing
time, frame rate, spatial resolution, temperature rise, mechanical index, and
quality metrics as raw `f64` values. Safety events also carried two unrelated
physical meanings through the same scalar fields. This allowed milliseconds,
seconds, hertz, millimetres, degrees Celsius, and dimensionless values to be
mixed without a type-level boundary. SNR is a logarithmic ratio and has no
dedicated Aequitas decibel unit in this contract.

## Decision

Use Aequitas `Time`, `Frequency`, `Length`, `TemperatureDifference`,
`ThermodynamicTemperature`, and `Dimensionless` at the public contract. Carry
the heterogeneous safety-event value and limit through `MonitoringMetric`,
which preserves the physical meaning of temperature rise, mechanical index,
and dimensionless quality/resource metrics. Safety checks return
`KwaversResult<()>` and propagate event-log failures.

Extract base scalars only at the running-mean, threshold, display, and report
formula boundaries. `SystemTime`, frame/error counters, and dense numerical
storage remain infrastructure or numerical boundaries. SNR remains a
dimensionless logarithmic ratio because no decibel unit is defined here.

The monitoring workflow is real-valued. Eunomia compatibility therefore means
that no physical phasor or complex-valued metric is manufactured, and no
imaginary physical unit is introduced. If a future monitoring path carries a
complex signal, its complex representation remains at the numerical
formula/storage boundary while its physical quantity uses the corresponding
real Aequitas dimension.

## Alternatives rejected

- Raw scalars: retain unit ambiguity at every caller.
- Local unit wrappers: duplicate Aequitas semantics and split the provider
  source of truth.
- One scalar safety-event payload: loses the distinction between temperature,
  mechanical index, and dimensionless values.
- Discarding event-log errors: hides a real monitoring failure.

## Verification

The implementation will pass the diagnostics test-target check, focused
clinical-monitoring Nextest, warning-denied all-target Clippy, doctests,
RustDoc, formatting, diff checks, and a public-contract scan for remaining
unit-bearing raw scalars. Value-semantic tests cover typed defaults, safety
events, temperature and mechanical-index alerts, quality scoring, and running
processing-time averages.
