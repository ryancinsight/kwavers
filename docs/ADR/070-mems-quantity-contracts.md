# ADR 070: Aequitas contracts for MEMS physical metrics

## Status

Accepted; implemented incrementally under `KWAVERS-AEQ-MET-32`.

## Context

The MEMS transducer modules exposed physical geometry, fluid properties, drive
frequency, and acoustic crosstalk as raw `f64` values. This allowed metres,
millimetres, hertz, density, and sound speed to be mixed at public call sites.
The crosstalk result is a complex phasor, but its imaginary component is
quadrature data rather than an independent physical unit.

## Decision

The crosstalk boundary uses Aequitas `Area`, `Length`, `Frequency`,
`MassDensity`, and `Velocity` inputs and returns
`AcousticImpedance<eunomia::Complex64>`. The `Complex64` value is extracted only
inside the closed-form monopole calculation and the distance helper. The
matrix stores typed acoustic impedances, including its zero diagonal.

CMUT/PMUT cell fields, plate stiffness/damping quantities, and sensitivity
contracts remain in the same vertical migration item. Dimensionless coupling,
quality, bandwidth, and empirical coefficients stay scalar because they do not
carry SI dimensions.

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
behavior. The remaining MET-32 surfaces are not claimed complete until their
public contracts are migrated and the transducer suite is rerun.
