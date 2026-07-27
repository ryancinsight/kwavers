# ADR 051: Typed thermal material and perfusion quantities

- Status: accepted
- Date: 2026-07-27
- Class: [major]

## Context

Kwavers stored thermophysical values in a Proteus bundle, but its public
material accessors returned conductivity, density, specific heat, and thermal
diffusivity as `f64`. The bioheat material contract also stored blood
perfusion and blood specific heat as raw values. The Pennes material rate
`w_b` is mass density per time (`kg/(m^3*s)`), so an incompatible rate could
cross the public boundary without dimensional validation.

## Decision

- Keep Proteus as the thermophysical source of truth.
- Use Aequitas `ThermalConductivity`, `MassDensity`,
  `SpecificHeatCapacity`, and `ThermalDiffusivity` for material accessors and
  stored properties.
- Use Aequitas `MassDensityRate` for material blood perfusion and typed blood
  specific heat at the material boundary.
- Convert to base scalars only at display, DTO, or numerical-stencil
  boundaries where the existing storage contract requires scalars.

## Alternatives rejected

- Retain raw accessors beside typed accessors: rejected because it preserves
  two public contracts and leaves unit mixing possible.
- Add a Kwavers-local perfusion newtype: rejected because the required rate
  dimension is reusable physical vocabulary owned by Aequitas.
- Move the thermophysical law out of Proteus: rejected because Proteus owns
  the validated material bundle and Kwavers must not duplicate it.

## Consequences

This is a pre-release public breaking change to thermal material constructors,
accessors, temperature-dependent properties, and their in-repository callers.
Rust callers construct typed values and inspect typed results; scalar
conversion is explicit at the formula and numerical-stencil boundaries.

## Verification

Kwavers-medium locked Nextest passes 191/191. The thermal/bubble physics
selection passes 361/361. Locked checks pass for `kwavers-medium`,
`kwavers-physics`, and `kwavers-simulation`; the latter also validates the
therapy constructor dependency and Pennes consumer path.
