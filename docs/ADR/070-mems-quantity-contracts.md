# ADR 070: Aequitas contracts for MEMS physical metrics

## Status

Accepted — 2026-08-05.

## Context

The MEMS crosstalk formula computes mutual radiation impedance as force per
source membrane velocity. Its coherent SI unit is `kg/s`. The former public
boundary accepted raw scalars and returned a complex value labeled
`AcousticImpedance`, whose physical contract is pressure per particle velocity
with unit `kg/(m²·s)`.

The CMUT, PMUT, and shared clamped-plate cell-output contracts also need
explicit names for their physical metrics. Flexible-array contracts remain a
separate audit item because the array owner is independently changing.

## Decision

Use Aequitas `Area`, `Length`, `Frequency`, `MassDensity`, and `Velocity` for
the real inputs. Return
`MechanicalImpedance<eunomia::Complex64>` from the scalar formula and store the
same quantity in the crosstalk matrix. Scalar extraction is restricted to the
wavenumber, magnitude, and Euclidean-distance formulas.

Use Aequitas `VolumeChargeDensity` with `CoulombPerCubicMeter` for the PMUT
charge-density gradient `|e₃₁,f|/t_p`, and `FlexuralRigidity` with `Joule` for
the shared plate rigidity `E h³/(12(1−ν²))`. CMUT and PMUT public cell metrics
use typed geometry, fluid, electrical, mechanical, acoustic, and dimensionless
quantities; formula arithmetic extracts only coherent scalar values.

Eunomia real and quadrature components are components of one observable
mechanical-impedance phasor, so both retain the single `kg/s` unit. No
imaginary SI unit or local complex wrapper is introduced.

## Alternatives rejected

- Keep raw `f64` and `Complex64`: rejected because unit-bearing callers could
  pass incompatible geometry and fluid quantities.
- Use `AcousticImpedance`: rejected because the crosstalk formula is force per
  velocity, not pressure per particle velocity.
- Use `DampingCoefficient`: rejected because its exponent vector matches but
  its semantic role is different at the public boundary.
- Assign an imaginary unit to the quadrature component: rejected because
  complex components share the observable quantity's unit.

## Verification

The Aequitas `dimension_laws` test verifies complex conversion through
`KilogramPerSecond`, `CoulombPerCubicMeter`, and `Joule`. Kwavers crosstalk,
CMUT, PMUT, plate, and comparison tests verify the value contracts, and the
Python MEMS binding library compiles with scalar conversion confined to its FFI
boundary.
