# ADR 054: Typed thermal-diffusion configuration quantities

- Status: accepted
- Date: 2026-07-24
- Class: [major]

## Context

The thermal-diffusion and Pennes configuration carriers still exposed
perfusion rate, blood density, blood specific heat, arterial temperature, and
thermal relaxation time as unrelated `f64` fields. The coupled simulation
configuration also exposed conductivity, tissue density, specific heat,
metabolic heat, frequency, and thermal time step as raw values. These fields
cross public Rust configuration and Python conversion boundaries before the
finite-difference kernels, so unit mistakes were possible before numerical
storage was reached.

## Decision

- Use Aequitas `ReciprocalTime`, `MassDensity`, `SpecificHeatCapacity`,
  `ThermodynamicTemperature`, and `Time` in the thermal-diffusion, Pennes, and
  Cattaneo configuration carriers.
- Use Aequitas `ThermalConductivity`, `MassDensity`,
  `SpecificHeatCapacity`, `VolumetricPowerDensity`, `Frequency`,
  `ThermodynamicTemperature`, and `Time` in the coupled simulation config.
- Preserve the Pennes equation `ω_b ρ_b c_b (T_a - T)/(ρ c_p)` by composing
  typed configuration quantities, then convert once at the scalar
  finite-difference kernel boundary.
- Keep dense temperature fields, grid arithmetic, and PyO3 arguments as their
  existing scalar representations. PyO3 converts validated Python values into
  the typed Rust configuration; the core does not depend on Python.
- Serialize the typed coupled configuration's current field names as SI base
  values explicitly, and deserialize through a scalar representation only at
  that boundary.

## Alternatives rejected

- Keep raw fields beside typed fields: rejected because it duplicates the
  configuration contract and permits unit mixing.
- Represent the Pennes input rate as `MassDensityRate` and remove the explicit
  blood-density factor: rejected for this legacy diffusion path because its
  documented equation and defaults define `perfusion_rate` as `1/s`; the
  material-owned Pennes path continues to use Aequitas `MassDensityRate`.
- Wrap Aequitas quantities in local unit structs: rejected because Aequitas
  already owns the reusable physical vocabulary.
- Type dense arrays: rejected because they are numerical storage boundaries,
  not scalar configuration metrics.

## Consequences

This is a pre-release public breaking change to the thermal-diffusion and
coupled-simulation Rust configuration fields and builder signatures. Python
callers retain their scalar unit-facing arguments and receive the same °C/
second conventions; conversion is explicit in the binding. The serialization
oracle proves SI values round-trip, and the Pennes and hyperbolic numerical
oracles remain unchanged. Full Cargo verification is still subject to the
existing peer Hermes/Coeus/Mnemosyne dependency-graph blocker.
