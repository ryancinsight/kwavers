# ADR 070: Aequitas contracts for MEMS physical metrics

## Status

Accepted and implemented under `KWAVERS-AEQ-MET-32` by commits `1afd09768`
and `6d15b5850`.

## Context

The MEMS transducer modules exposed physical geometry, material, fluid, drive,
resonance, pressure, capacitance, damping, and sensitivity values as raw
`f64` values. This allowed metres, millimetres, hertz, density, sound speed,
voltage, and pressure to be mixed at public call sites. The crosstalk result is
a complex phasor, but its imaginary component is quadrature data rather than
an independent physical unit.

## Decision

The crosstalk boundary uses Aequitas `Area`, `Length`, `Frequency`,
`MassDensity`, and `Velocity` inputs and returns
`AcousticImpedance<eunomia::Complex64>`. CMUT/PMUT cells and plate helpers use
typed geometry, material, fluid, drive, resonance, pressure, capacitance,
power, spring-stiffness, and damping quantities. Sensitivity uses typed
pressure-per-potential, potential-per-pressure, and length-per-potential
contracts. The `Complex64` value is extracted only inside closed-form formulas
and distance helpers. The matrix stores typed acoustic impedances, including
its zero diagonal.

Dimensionless coupling, quality, bandwidth, and empirical coefficients stay
scalar because they do not carry SI dimensions.

## Alternatives rejected

- Keep raw crosstalk scalars: rejected because the public boundary would retain
  unit ambiguity.
- Add a local complex-impedance wrapper: rejected because Aequitas owns the
  physical dimension and Eunomia owns the scalar provider seam.
- Assign a separate unit to the imaginary component: rejected because it is the
  quadrature component of one phasor.

## Verification

The typed crosstalk tests preserve the closed-form magnitude and phase oracle,
reciprocity, inverse-distance scaling, zero diagonal, and invalid-length
behavior. CMUT, PMUT, plate, flexible-apodization, comparison, and sensitivity
tests preserve the corresponding closed-form scaling and ordering oracles.
`cargo check -p kwavers-transducer --tests` and
`cargo nextest run -p kwavers-transducer` pass with 219/219 tests and one
declared skip. Full-target Clippy remains blocked only by the peer-owned
`crates/kwavers-math/src/simd/mod.rs:6` `doc_overindented_list_items` error.
