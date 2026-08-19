# ADR 105: B-mode scan-conversion geometry quantities

- Status: Accepted
- Date: 2026-08-06
- Driver: `KWAVERS-AEQ-MET-69`

## Context

The B-mode scan converter maps polar beam samples to a Cartesian raster. Its
public `ScanGeometry` and `CartesianGrid` contracts described angles and
metres in documentation while carrying them as raw `f64` values. This allowed
an angle, length, or unit-scaled value to be supplied without a type-level
boundary. RF and image arrays are dense numerical storage and are not metric
contracts.

## Decision

Carry beam angles as Aequitas `Angle<f64>` and apex/range/grid extents as
`Length<f64>`. Convert to radians and metres only inside scan-conversion
validation and the trigonometric, interpolation-index, and Cartesian-raster
formulas. Validate finite ordered extents, positive steps, and non-negative
apex offsets before construction.

The contract is real-valued. If a future complex RF path reaches this display
stage, its real and quadrature components retain one existing observable
signal unit. Eunomia does not receive an imaginary angle or length unit.

## Alternatives rejected

- Retain raw scalars with unit-suffixed names: rejected because names do not
  prevent unit transposition at a public constructor.
- Wrap dense RF/image arrays in quantities: rejected because their element type
  is numerical storage, not a physical metric boundary.
- Add local geometry wrappers or forwarding constructors: rejected because
  Aequitas owns the shared vocabulary and compatibility layers preserve the
  obsolete contract.

## Verification

The affected `kwavers-analysis` library check, warning-denied all-target
Clippy, B-mode Nextest filter (9/9), doctests, Rustdoc, formatting, and diff
checks pass. The overlay reports existing unused local patches and linker
diagnostics outside this change.
