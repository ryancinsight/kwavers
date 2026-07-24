# ADR 051: Typed thermal material and perfusion quantities

- Status: accepted
- Date: 2026-07-23
- Class: [major]

## Context

Kwavers stored thermophysical values in a Proteus bundle, but its public
material accessors returned conductivity, density, specific heat, and thermal
diffusivity as `f64`. The Pennes bio-heat contract also stored blood perfusion
`w_b` and blood specific heat as raw values. The Pennes equation defines
`w_b` with units kg/(m³·s), so callers could pass an incompatible rate without
the type system detecting it.

## Decision

- Keep Proteus as the thermophysical source of truth.
- Use Aequitas `ThermalConductivity`, `MassDensity`,
  `SpecificHeatCapacity`, and `ThermalDiffusivity` for material accessors and
  stored properties.
- Add Aequitas `MassDensityRate` for Pennes blood perfusion and use typed blood
  specific heat at the material boundary.
- Convert to base scalars only at the finite-difference arithmetic boundary,
  where the arrays and numerical stencil already use scalar storage.

## Alternatives rejected

- Retain raw accessors beside typed accessors: rejected because it preserves
  two public contracts and leaves unit mixing possible.
- Add a Kwavers-local perfusion newtype: rejected because the required rate
  dimension is reusable physical vocabulary and belongs in Aequitas.
- Move the thermophysical law out of Proteus: rejected because Proteus owns
  the validated material bundle and Kwavers should not duplicate it.

## Consequences

This is a pre-release public breaking change to thermal material constructors,
accessors, and Pennes property handling. Rust callers construct typed values
and inspect typed results; scalar conversion is explicit at display, legacy
DTO, and finite-difference kernel boundaries. Aequitas dimension-law tests
prove that mass-density rate multiplied by time has mass-density dimensions.
Kwavers-medium Nextest passes 191/191, and the thermal and bubble-dynamics
physics selection passes 361/361. The adjacent bubble heat-transfer correction
also uses `TemperatureDifference` for a temperature change rather than an
absolute thermodynamic temperature.
