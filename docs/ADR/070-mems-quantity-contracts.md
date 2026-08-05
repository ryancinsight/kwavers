# ADR 070: Aequitas contracts for MEMS physical metrics

## Status

Accepted — 2026-08-05.

## Context

The MEMS crosstalk formula computes mutual radiation impedance as force per
source membrane velocity. Its coherent SI unit is `kg/s`. The former public
boundary accepted raw scalars and returned a complex value labeled
`AcousticImpedance`, whose physical contract is pressure per particle velocity
with unit `kg/(m²·s)`.

CMUT and PMUT cell-output contracts and flexible-array contracts remain
separate audit items; this decision covers the mutual-radiation crosstalk
boundary.

## Decision

Use Aequitas `Area`, `Length`, `Frequency`, `MassDensity`, and `Velocity` for
the real inputs. Return
`MechanicalImpedance<eunomia::Complex64>` from the scalar formula and store the
same quantity in the crosstalk matrix. Scalar extraction is restricted to the
wavenumber, magnitude, and Euclidean-distance formulas.

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
`KilogramPerSecond`. Kwavers crosstalk tests verify closed-form magnitude and
phase, reciprocity, inverse-distance scaling, zero diagonal, and degenerate
input behavior.
