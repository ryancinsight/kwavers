# ADR 053: Type derived grid metrics with Aequitas

- **Status:** Accepted
- **Date:** 2026-07-24
- **Change class:** [major]

## Context

`kwavers-grid::Grid` stored numerical spacing as raw scalars and exposed
derived spacing, domain size, volume, cell volume, and CFL timestep values as
raw `f64`. Callers could therefore pass a length where a time or volume was
expected without a type-level distinction. The raw coordinate arrays and grid
storage are numerical-kernel representations and are not public physical
metric reports.

## Decision

Use Aequitas `Length`, `Volume`, `Velocity`, and `Time` for the public derived
metric methods. Convert to base scalars only in `contains_point`, `bounds`, and
the stability arithmetic that operates on raw numerical arrays. The CFL method
accepts a typed sound speed and returns a typed time interval.

Retain raw `Grid` spacing fields and coordinate arrays as the explicit
coordinate/numerical-kernel boundary for this increment. A field descriptor or
full storage migration requires a separate contract because it affects the
entire simulation data plane.

## Alternatives

- Retain raw return values: rejected because the public derived metrics are
  physical quantities and already have Aequitas provider types.
- Add local wrapper types: rejected because it duplicates Aequitas ownership.
- Convert every grid field and coordinate array in this change: rejected because
  it expands a metric-boundary change into a simulation storage migration.

## Consequences

The derived metric API is breaking and all in-repository callers are migrated
in the same change. Value tests pin spacing, domain-size, volume, and CFL
semantics. The remaining scalar conversions are named kernel boundaries rather
than parallel public physical contracts.
