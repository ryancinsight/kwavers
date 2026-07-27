# ADR 050: Typed vasculature physical quantities

- Status: Accepted
- Date: 2026-07-26

## Context

The vasculature analysis API reported diameter and length in voxel-index units,
returned centerline coordinates as untyped scalars, and returned Doppler speed
as `f64`. The caller therefore had to apply an undocumented spacing convention,
which made anisotropic images physically ambiguous. The functional-ultrasound
diagnostics workflow already owns the construction `Grid` and is the correct
boundary for forwarding its spacing.

## Decision

`VesselSegmentation::segment` accepts a validated `[Length; 3]` voxel-spacing
array. Vessel classification and axial extent calculations operate in physical
coordinates and return Aequitas `Length` values. Centerline coordinates are
`[Length; 3]`. Doppler estimation accepts Aequitas `Frequency`, `Velocity`, and
`Angle` values and returns `Velocity`; scalar extraction is limited to the
formula kernel. `FunctionalUltrasoundGPS` captures the grid spacing at
construction and forwards it to segmentation.

The equivalent circular diameter uses physical voxel volume and axial extent:

```text
V = N · Δx · Δy · Δz
A = V / L
d = sqrt(4A / π)
```

## Alternatives rejected

- Retaining voxel-unit scalars and documenting caller-side scaling leaves the
  dimensional contract unenforced and fails for anisotropic spacing.
- Adding a local scalar wrapper duplicates Aequitas semantics and preserves
  primitive obsession at the public boundary.
- Adding a new provider perfusion dimension is outside this decision; the
  thermal/perfusion contract remains the separate MET-06 audit item.

## Consequences

This is a breaking public API change. Existing callers must provide physical
spacing and consume typed geometry/velocity. The diagnostics workflow now
preserves the grid's physical contract without a second conversion path.

## Verification

The anisotropic-spacing oracle checks physical length and equivalent diameter;
invalid spacing is rejected; Doppler tests check the analytical equation and
invalid inputs. The focused vasculature Nextest lane passes 22/22 and the
locked `kwavers-analysis` package check passes. Broader package and diagnostics
gates remain blocked by peer-owned `mnemosyne-local` compile errors and are
not used as evidence for this local result.
