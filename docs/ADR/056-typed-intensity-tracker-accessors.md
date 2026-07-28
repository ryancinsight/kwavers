# ADR 056: Typed intensity-tracker accessors

- Status: accepted
- Date: 2026-07-27

## Context

`IntensityTracker` stores its current and peak measurements as Aequitas
`Intensity`, but its public convenience accessors returned raw `f64` values
with `_w_cm2` names. This left a unit-conversion API beside the typed metric
API and required callers to infer the result unit from the identifier.

## Decision

Replace the unit-suffixed accessors with canonical typed accessors returning
`Intensity<f64>`. Keep W/cm² conversion at explicit presentation or
unit-conversion boundaries; the tracker itself reports SI base intensity in
W/m², matching its stored `Intensity` metrics.

## Alternatives rejected

- Retaining the `_w_cm2` methods was rejected because the names encode a
  presentation unit and preserve a raw scalar boundary after typing the
  underlying metric.
- Adding typed methods beside the raw methods was rejected because it would
  retain duplicate public contracts and compatibility surface.

## Consequences

This is a pre-release public breaking change. In-repository callers construct
and compare typed intensities; external callers must select an explicit unit
conversion when presenting W/cm².

## Verification

The intensity-tracker value tests, full `kwavers-therapy` focused Nextest,
doctests, warning-denied Clippy, and Rustdoc will gate the implementation.
