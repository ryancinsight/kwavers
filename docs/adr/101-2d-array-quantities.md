# ADR 101 — Aequitas contracts for 2-D array geometry and beam control

- Status: Accepted
- Date: 2026-08-03
- Scope: `kwavers-transducer::array_2d` and its direct Kwavers Python and
  simulation callers
- Driver: `KWAVERS-AEQ-MET-62`

## Context

`TransducerArray2DConfig`, `Array2dElement`, and `TransducerArray2D` expose
length, position, speed, frequency, delay, and steering-angle values as raw
scalars. The flat-array state is encoded as `f64::INFINITY` in the radius and
focus fields. That representation admits non-finite values into a physical
contract and makes flat versus finite curvature a value-level convention.
The implementation also adds element width to the documented center-to-center
spacing when computing positions, so the public spacing contract does not
match its geometry.

## Decision

Use Aequitas at the Rust contract boundary:

- `Length<f64>` for element width, elevation length, center-to-center spacing,
  Cartesian coordinates, aperture width, radius, and focus distances.
- `Velocity<f64>` for sound speed, `Frequency<f64>` for operating frequency,
  `Time<f64>` for element delays, and `Angle<f64>` stored in radians for
  steering.
- `ArrayCurvature::{Flat, Cylindrical { radius: Length<f64> }}` for the
  surface state. A finite positive radius is required for the cylindrical
  variant; flatness is not represented by an infinite length.
- `Option<Length<f64>>` for electronic and elevation focus. `None` means no
  focus; no infinite physical distance sentinel is retained.
- `[Length<f64>; 3]` for element and center coordinates.

The Rust builder and core array constructors accept these typed values. The
Python binding keeps its existing scalar SI/degrees interface as an explicit
FFI serialization boundary and converts to/from Aequitas there. Mesh indexing,
distance, trigonometric, and delay formulas extract scalars only inside their
formula or mesh boundaries.

Center-to-center spacing is used directly as the element pitch. Element width
controls aperture extent and validation (`width <= spacing`) but is not added
to the pitch a second time.

## Eunomia compatibility

The array geometry is real. A signal carried by Eunomia may have real and
quadrature components, but they remain components of one observable signal
unit. This change introduces no imaginary length, complex geometry, or complex
SI unit.

## Alternatives rejected

- Retaining `f64::INFINITY` for flatness or no focus: rejected because a
  non-finite scalar is not a physical length and bypasses typed validation.
- Adding local scalar compatibility accessors to the Rust API: rejected by the
  migration policy; only the Python boundary retains scalar serialization.
- Modeling steering in degrees: rejected because Aequitas stores angles in the
  coherent SI radian unit and the degree conversion belongs at the Python
  boundary.

## Verification

- Unit and analytical regressions cover flat pitch, finite cylindrical sag,
  aperture width, finite/no-focus state, and SI/radian round trips.
- Strict Clippy and focused Nextest cover the transducer, Python, and direct
  simulation callers.
- Hosted repository-owned feature, wheel, coverage, safety, and architecture
  gates run at the final source head.
