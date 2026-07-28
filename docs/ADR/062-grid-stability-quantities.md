# ADR 062: Type grid stability quantities

- Status: Accepted
- Date: 2026-07-27
- Change class: [arch] [major]

## Context

`kwavers-grid` exposed CFL, diffusion, nonlinear, and recommended time-step
contracts as raw `f64` values. Sound speed and thermal diffusivity were also
raw physical inputs, so a caller could pass incompatible quantities without a
type-level distinction. `kwavers-core::time` already owns the simulation-time
contract through Aequitas.

## Decision

Use Aequitas at the public stability boundary:

- accept `Velocity` for acoustic propagation speed;
- accept `ThermalDiffusivity` for diffusion limits;
- return `Time` from CFL, diffusion, nonlinear, and recommended timestep
  calculations;
- accept `Time` for the Courant-number and FDTD stability predicates;
- keep Courant number and the nonlinearity coefficient scalar because they are
  dimensionless model values;
- keep `Grid` spacing as scalar mesh storage, with the grid constructor as the
  existing validated mesh boundary;
- extract SI scalars only inside numerical formulas and legacy numerical
  integration maps.

This is a direct contract migration. No scalar compatibility facade or local
unit alias is retained.

## Alternatives rejected

- Leaving the API scalar would preserve the unit-confusion defect.
- Typing all mesh coordinates in this increment would mix storage layout with
  stability contracts and expand the migration beyond the identified gap.
- A Kwavers-local wrapper would duplicate Aequitas ownership and create a
  second physical-unit vocabulary.

## Verification

The acceptance oracle is value-semantic stability coverage: positive CFL and
diffusion bounds, stable/unstable predicate behavior, recommended-step bound
selection, migrated callers, and warning-denied package gates. Dense fields,
dimensionless ratios, and numerical-array boundaries remain explicitly scalar.
