# ADR 052: Typed HIFU planning physical quantities

- Status: accepted
- Date: 2026-07-24
- Class: [major]

## Context

HIFU planning exposed transducer frequency, focal dimensions, acoustic power,
target geometry, focal pressure, focal volumes, and schedule coordinates as
unit-suffixed `f64` values. The same planning package already used Aequitas
`Time` and `ThermodynamicTemperature` for dose results, so the remaining
physical boundaries could still mix millimetres, metres, watts, and pascals.

## Decision

- Store transducer frequency, focal geometry, power, and target geometry as
  Aequitas `Frequency`, `Length`, and `Power` values.
- Store focal pressure and focal volumes as Aequitas `Pressure` and `Volume`.
- Use the shared validated `CartesianPosition` for focal and schedule
  coordinates.
- Pass Aequitas `Frequency` and `Time` into focal-dose and schedule kernels;
  convert to base scalars only inside the closed-form arithmetic and axis-grid
  helpers. Retain mechanical index, duty cycle, and CEM43 as model-semantic
  scalar values.

## Alternatives rejected

- Preserve `_mm`, `_pa`, and `_mm3` scalar fields beside typed fields: rejected
  because two public contracts permit unit mixing and duplicate state.
- Add a local HIFU position or volume wrapper: rejected because Aequitas and
  the existing transducer `CartesianPosition` already own these seams.
- Type `ClinicalTherapyParameters` in this slice: rejected because it is a
  shared cross-therapy configuration boundary; HIFU converts its frequency and
  duration into typed values at the planning boundary without widening scope.

## Consequences

This is a pre-release public breaking change to HIFU planning constructors,
fields, and schedule outputs. The planning API now carries physical units in
its DTOs, while numerical model formulas retain explicit scalar boundaries.
The analytical O'Neil focal-width and acoustic-pressure formulas remain
unchanged in SI base units. Full package verification is externally blocked by
pre-existing peer `coeus-nn` compilation errors; rustfmt and the exact
source-level audit pass, and the blocked check is recorded in the child and
Atlas ledgers rather than hidden.
