# ADR 064: Type hemispherical-array quantities

- Status: Accepted
- Date: 2026-07-27
- Scope: `kwavers-transducer` hemispherical-array public contracts

## Context

The hemispherical-array module exposed radius, aperture, focal length,
element positions and radii, phase offsets, steering frequency and sound
speed, focal pressure/volume, and validation pressure as unit-documented
`f64` values. Callers could therefore pass incompatible physical dimensions
without a type-level failure, and the metrics surface was not discoverable
through the transducer crate root. The Source implementation also ignored the
frequency supplied to its constructor and always generated a 650 kHz waveform.

## Decision

Use the existing Aequitas contracts at the public boundary:

- `Length` for geometry, positions, element radii, and focal-point locations;
- `Angle` for phase offsets and steering range;
- `Frequency`, `Velocity`, and `Time` for steering inputs;
- `Pressure` and `Volume` for focal metrics and validation limits.

Normals, apodization, efficiency, grating-lobe ratios, and element counts are
dimensionless or structural and remain scalar. Scalar extraction occurs only
where the implementation crosses a mesh/layout formula, `Source` position,
`Signal`, or logging boundary. `HemisphericalArrayMetrics` is re-exported by
the crate root so the typed metric contract is part of the discoverable public
surface. The nominal source sound-speed constant is also exposed as an
Aequitas `Velocity`; the skull attenuation coefficient remains a
frequency-normalized model parameter rather than a standalone SI metric. The
Source waveform retains the configured `Frequency` and extracts its base value
only at the signal formula boundary.

## Alternatives rejected

- Retaining raw `f64` values with unit-suffixed documentation preserves the
  dimensional bug at every call site.
- Adding local wrapper types would duplicate Aequitas ownership and create a
  second conversion surface.
- Adding scalar compatibility constructors would retain the ambiguous API and
  violate the migration's provider-first boundary.

## Verification

The transducer package and `brain_theranostic_monitor` example checks pass.
The full transducer Nextest run passes 219/219 with one skipped test;
transducer doctests pass 2/2 with six ignored; warning-denied Clippy passes;
Rustdoc exits successfully; touched-file rustfmt and `git diff --check` pass.
