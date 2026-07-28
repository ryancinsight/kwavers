# ADR 067: Type CEUS physical contracts with Aequitas

## Status

Accepted — 2026-07-27

## Context

The public contrast-enhanced ultrasound (CEUS) surface carried frequencies,
frame rates, bubble geometry, shell properties, concentrations, and scattering
inputs/results as unit-documented scalar values. The microbubble constructor
also converted micrometres and kilopascals internally, while population setup
accepted bubbles per millilitre without making the conversion visible at the
contract boundary. These shapes allowed physically incompatible values to be
passed between imaging, physics, and simulation crates.

## Decision

Use the Aequitas provider types directly across the public CEUS contracts:

- `Frequency` for transmit, frame-rate, and resonance quantities;
- `Length` for field of view, depth, bubble radius, and shell thickness;
- `Pressure` for acoustic drive and shell elasticity;
- `MassDensity` for liquid density;
- `DynamicViscosity` and `SurfaceTension` for shell/liquid properties;
- `Area` for scattering cross sections;
- `NumberDensity` for bubble concentration; and
- `ReciprocalLength` for population scattering response.

Callers provide SI-base values through Aequitas constructors. Scalar extraction
is limited to empirical formulas, numerical integration, dense field/mesh
construction, signal generation, and logging. Dimensionless controls,
probabilities, harmonic ratios, and model-specific coefficients remain scalar.
Dense perfusion and pressure arrays remain storage boundaries rather than
pretending that an array container itself carries an Aequitas dimension.

## Alternatives rejected

- Consumer-owned wrapper types: rejected because they duplicate Aequitas
  dimensions and create a second unit-conversion source of truth.
- Retaining implicit micrometre, kilopascal, or bubbles-per-millilitre
  conversion: rejected because the public contract hides the unit boundary.
- Typing dense arrays element-by-element: rejected for this increment because
  the storage and numerical backends remain scalar-array contracts; a typed
  field descriptor is a separate architecture item.

## Verification

The affected `kwavers-imaging`, `kwavers-physics`, and `kwavers-simulation`
packages pass 1,862/1,862 Nextest tests with one repository-declared skip.
Warning-denied Clippy, doctests, targeted rustfmt, and `git diff --check` pass.
Rustdoc exits successfully with one pre-existing private intra-doc-link
warning. CEUS tests cover closed-form resonance/scattering behavior, positive
and frequency-sensitive scattering, cloud dynamics, and typed defaults.
