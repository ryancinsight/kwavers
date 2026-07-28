# ADR 055: Typed acoustic-solver physical quantities

- Status: accepted
- Date: 2026-07-27

## Context

The therapy acoustic solver exposed simulation durations and timestamps as
`f64`, maximum pressure as a unit-converted scalar, and SPTA intensity as
W/cm². The surrounding therapy orchestrator also passed scalar time intervals
through safety, heating, cavitation, chemical, microbubble, and lithotripsy
helpers. These contracts allowed seconds, pressure units, and intensity units
to cross public boundaries without type enforcement.

## Decision

Use the existing Aequitas quantities at the acoustic and orchestrator
boundaries:

- `Time` for solver durations, time steps, current time, and helper intervals;
- `Pressure` for maximum pressure;
- `Intensity` for SPTA intensity; and
- `Length` for the focal-depth heating input.

Dense Leto pressure, velocity, temperature, and activity arrays remain scalar
storage boundaries. Scalar extraction is restricted to backend stepping,
formula arithmetic, mesh/index arithmetic, and untyped numerical kernels.
SPTA is returned in Aequitas SI base units (W/m²); callers select display or
clinical units explicitly.

## Alternatives rejected

- Retaining W/cm² or MPa return values was rejected because unit-bearing names
  do not provide dimensional enforcement and conceal conversion at the API.
- Wrapping the acoustic solver in a therapy-owned metric facade was rejected
  because Aequitas already owns the physical dimensions.
- Converting dense field arrays element-by-element was rejected for this
  increment because Leto arrays are the mesh/storage boundary and conversion
  would allocate or duplicate the field representation.

## Consequences

This is a pre-release public breaking change. Acoustic callers construct typed
quantities and extract SI base values only at the explicit backend or formula
boundary. The solver and orchestrator now share the same dimensional contract
as the therapy configuration and intensity tracker.

## Verification

The committed revision passes the `kwavers-therapy` package check, focused
Nextest (349/349 passed, one skipped, two slow, 37.399 seconds), doctests
(8 passed, one ignored), warning-denied Clippy, and Rustdoc. Process-local
provider-path overrides were used to avoid the shared overlay's duplicate
Aequitas worktree package collision; the overlay and peer-owned lockfile were
not changed.
