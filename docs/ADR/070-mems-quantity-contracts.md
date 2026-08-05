# ADR 070: Aequitas contracts for MEMS physical metrics

## Status

Accepted — 2026-08-05.

### Revision 2026-08-05

The adjacent flexible-array metric boundary is now included in this accepted
quantity decision. The original MEMS slice left that boundary separate because
its array owner was changing independently.

## Context

The MEMS crosstalk formula computes mutual radiation impedance as force per
source membrane velocity. Its coherent SI unit is `kg/s`. The former public
boundary accepted raw scalars and returned a complex value labeled
`AcousticImpedance`, whose physical contract is pressure per particle velocity
with unit `kg/(m²·s)`.

The CMUT, PMUT, and shared clamped-plate cell-output contracts also need
explicit names for their physical metrics. The flexible-array public boundary
also needs explicit contracts for timestamps, focus and speed inputs, delays,
calibration quality, curvature, deformation strain, stress, safety limits, and
strain-energy density.

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

For the flexible-array boundary, use Aequitas `Time` for update and snapshot
timestamps and delays, `Length` for focus coordinates, curvature radius, and
position uncertainty, `Velocity` for sound speed, `Angle` for orientation
uncertainty, `Dimensionless` for calibration confidence, quality ratios, strain,
and safety limits, `ReciprocalLength` for Menger curvature, `Pressure` for
stress, and `EnergyPerVolume` for the `½ ε σ` strain-energy density. Dense Leto
position, normal, measurement, signal, and source arrays remain mesh/storage
boundaries. The curvature implementation uses the three-point Menger law rather
than the former dimensionless turning-angle average.

## Alternatives rejected

- Keep raw `f64` and `Complex64`: rejected because unit-bearing callers could
  pass incompatible geometry and fluid quantities.
- Use `AcousticImpedance`: rejected because the crosstalk formula is force per
  velocity, not pressure per particle velocity.
- Use `DampingCoefficient`: rejected because its exponent vector matches but
  its semantic role is different at the public boundary.
- Assign an imaginary unit to the quadrature component: rejected because
  complex components share the observable quantity's unit.
- Return `Energy` for flexible deformation: rejected because `½ ε σ` lacks an
  integrated element volume and therefore has units `J/m³`.
- Preserve turning-angle curvature: rejected because the strain and flex-
  derating formulas require `1/m`, not a dimensionless angle.

## Verification

The Aequitas `dimension_laws` test verifies complex conversion through
`KilogramPerSecond`, `CoulombPerCubicMeter`, and `Joule`. Kwavers crosstalk,
CMUT, PMUT, plate, and comparison tests verify the value contracts, and the
Python MEMS binding library compiles with scalar conversion confined to its FFI
boundary. Flexible beamforming tests verify typed delay and Menger-curvature
contracts; geometry tests verify flat zero curvature and the `½ ε σ` energy-
density value. A focused flexible run passed 6/6 before a clean rebuild
exhausted the shared disk; the later rebuild is blocked before Kwavers by the
live Melinoe missing-import defect documented in the child gap audit.
