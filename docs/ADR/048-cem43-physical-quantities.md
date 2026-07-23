# ADR 048: Typed CEM43 and HIFU planning metrics

- Status: accepted
- Date: 2026-07-23
- Class: [major]

## Context

Kwavers used Aequitas `Time` for some CEM43 integration steps but returned
equivalent minutes, peak temperature, dwell duration, and time-to-ablation as
untyped scalars. The same contract was duplicated between the physics thermal
calculators and the HIFU planning schedule.

## Decision

- Add `CumulativeEquivalentMinutes<T>` in `kwavers-physics` as a validated,
  finite non-negative consumer-semantic type backed by Aequitas `Time<T>` in
  base seconds.
- Keep dense CEM43 arrays as storage representations, but return the typed
  value from maxima and point queries and accept typed update intervals and
  thresholds.
- Return HIFU planning peak temperatures as Aequitas
  `ThermodynamicTemperature` and dwell/time-to-dose values as Aequitas `Time`.
  An unreachable time-to-dose is represented as `None`, not an infinite
  scalar.
- Keep CEM43 distinct from Aequitas `AbsorbedDose`: it is cumulative
  equivalent time at a reference temperature, not deposited energy per mass.

The Sapareto–Dewey accumulation law and existing focal-geometry equations stay
unchanged. All in-repository callers and value-semantic tests migrate in the
same change; no scalar compatibility surface remains.

## Alternatives rejected

- Add an Aequitas absorbed-dose alias for CEM43: rejected because the clinical
  quantity has time semantics and is not J/kg.
- Keep raw arrays and only type constructors: rejected because maxima, point
  queries, thresholds, and planning results would still erase the contract.
- Encode an unreachable target time as `Time(INFINITY)`: rejected because the
  validated time contract excludes non-finite values; `Option<Time>` preserves
  the model's absence semantics.

## Consequences

This is a pre-release public breaking change across the physics and therapy
planning crates. Consumers must construct Aequitas `Time` values and inspect
typed CEM43/temperature results. Dense field storage remains allocation- and
layout-compatible, and the existing analytical CEM43 reference tests provide
the behavioral oracle.
