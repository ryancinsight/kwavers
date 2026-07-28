# ADR 057: Typed clinical histotripsy scenario quantities

- Status: accepted
- Date: 2026-07-27

## Context

`HistotripsyScenario` and `PulsePattern` exposed frequency, pressure, time,
volume, pulse-repetition frequency, and derived intensity as unit-suffixed
`f64` fields or results. Callers had to preserve the unit convention encoded
in identifiers, while the therapy integration surface already uses Aequitas
for the same physical dimensions.

## Decision

Use Aequitas `Frequency`, `Pressure`, `Time`, `Volume`, and `Intensity` for
the public scenario and pulse contracts. Rename unit-suffixed fields and
methods to canonical domain names. Extract SI scalars only inside the
mechanical-index, cavitation-probability, and duty-cycle formulas. Model the
absence of PRF for single-pulse patterns as `Option<Frequency<f64>>` rather
than manufacturing a NaN frequency.

## Alternatives rejected

- Retaining raw fields with unit-suffixed names was rejected because it keeps
  physical metrics untyped at a public therapy boundary.
- Adding typed accessors beside the old fields and methods was rejected
  because it would retain duplicate contracts and a compatibility facade.
- Returning a NaN `Frequency` for patterns without PRF was rejected because
  expected absence is represented by `Option`, not a special floating value.

## Consequences

This is a pre-release public breaking change. Scenario constructors and
in-repository callers construct typed quantities; formula consumers receive
the same dimensionless results and probability values as before.

## Verification

The clinical-scenario value tests, focused `kwavers-therapy` Nextest,
doctests, warning-denied Clippy, Rustdoc, and formatting gates verify the
implementation.
