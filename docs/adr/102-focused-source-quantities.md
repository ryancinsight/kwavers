# ADR 102: Focused-source physical quantities

- Status: Accepted
- Date: 2026-08-03
- Driver: `KWAVERS-AEQ-MET-63`

## Decision

Focused source geometry and drive contracts use Aequitas quantities at their
public Rust boundaries:

- lengths for radii, diameters, centers, foci, positions, and spacing;
- angles for polar spans, arc orientation, phase, and steering;
- frequencies for operating frequency;
- pressures for source amplitudes;
- areas and times for element weights and delays;
- volumes, pressures, angles, and dimensionless values for hemispherical
  validation metrics.

Scalar extraction is limited to validation, trigonometric and propagation
formulas, mesh/index addressing, and explicit FFI or serialization boundaries.
Clinical adapters that retain legacy `Point3` or array values convert at the
boundary rather than passing raw scalars into the source-domain API.

Eunomia complex values remain numerical phasors, not a second physical unit.
Real and quadrature components represent one observable pressure or signal
quantity; focused geometry remains real and no imaginary SI unit is introduced.

Beam steering preserves the physical aperture, retargets each element normal,
and recomputes focus delays through the typed focus contract.

## Alternatives rejected

- Retaining raw `f64` fields would permit metre/radian/pascal/frequency mixups
  at the source boundary.
- Adding imaginary SI units would misrepresent quadrature storage as a distinct
  physical dimension and conflict with Eunomia's complex-number model.
- Rebuilding or moving elements during beam steering would change the physical
  aperture instead of changing the phase law.

## Verification

The focused transducer tests cover equal-area geometry, analytic cap area,
O'Neil focal pressure, invalid domains, multi-bowl apodization, and steering
normal updates. Downstream therapy and diagnostics adapters are compiled
against the typed API; legacy scalar extraction remains at their declared
domain boundaries.
