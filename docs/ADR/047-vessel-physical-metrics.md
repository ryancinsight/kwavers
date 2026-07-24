# ADR 047: Type functional-ultrasound vessel physical metrics

- Status: Accepted
- Date: 2026-07-23
- Class: [major] [arch]

## Context

Functional-ultrasound vasculature analysis accepted image voxels without a
spacing contract. Vessel diameter and total length therefore remained voxel
counts, centerline coordinates were indices, and Doppler frequency, sound
speed, and velocity crossed the public boundary as unrelated scalars.

## Decision

Require a validated `VoxelSpacing` containing positive finite Aequitas
`Length<f64>` components for every axis. Scale vessel geometry in physical
metres before principal-axis classification; return typed `Length<f64>` for
diameter, total length, and centerline coordinates. Accept Aequitas
`Frequency<f64>` and `Velocity<f64>` at the Doppler boundary and return typed
`Velocity<f64>`. Confidence, orientation, flow direction, voxel masks, and
variance-like image values remain dimensionless or representation data.

`VoxelSpacing` is owned by `kwavers-analysis` because it validates the image
geometry consumed by segmentation. The diagnostics GPS workflow accepts that
validated value and passes it through without reconstructing or converting it.

## Rejected alternative

Retaining raw fields with unit-bearing comments would leave voxel spacing
implicit and allow metre/millimetre/index mistakes at callers. Storing spacing
as three unrelated scalars would permit partial or non-positive geometry.

## Verification

- Invalid spacing rejects zero, negative, and non-finite components.
- Anisotropic spacing scales centerline coordinates, physical length, and
  equivalent diameter against the closed-form volume/length relation.
- Doppler velocity matches `f_d c / (2 f_0 cos(theta))` with typed inputs and
  preserves invalid-angle/frequency rejection.
- Affected analysis and diagnostics package gates were attempted on the exact
  implementation revision. Formatting and metadata pass; compilation stops in
  the current peer `mnemosyne` checkout before Kwavers sources because three
  `mnemosyne-heap` matches omit `TierSelection::Hbm` and `TierSelection::Gddr`.
