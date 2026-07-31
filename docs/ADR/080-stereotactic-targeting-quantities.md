# ADR 080 — Aequitas stereotactic targeting quantities

- Status: Accepted
- Date: 2026-07-31
- Board item: `KWAVERS-AEQ-MET-41`

## Context

`kwavers-diagnostics::functional_ultrasound::targeting` exposes anterior-
posterior, medial-lateral, and dorsal-ventral coordinates as `f64` values
documented in millimetres. The Bregma reference and Euclidean targeting
distance use the same untyped values. Coordinate confidence is a public
fraction but has no dimension marker. These fields cross the targeting API,
trajectory planner, and atlas conversion boundary.

The atlas provider API is intentionally millimetre-based and returns raw arrays
for voxel indexing. That boundary must remain explicit; it must not force raw
millimetres into the domain coordinate contract.

## Decision

Represent stereotactic AP/ML/DV coordinates, Bregma, and targeting distances
with Aequitas `Length<f64>` stored in SI base units. Represent confidence with
`Dimensionless<f64>`. The constructor and public fields use these typed values;
trajectory interpolation and validity/safety comparisons remain typed.

Convert to millimetres only at the existing `BrainAtlas::voxel_to_mm`,
`BrainAtlas::mm_to_voxel`, and explicit array/serialization boundaries. The
conversion uses Aequitas `Millimeter`, so the public contract cannot mix SI
metres with atlas millimetres implicitly.

The metric family is real-valued. It has no phasor, spectral, or imaginary
component, so Eunomia `Complex` support is not involved. Future coherent
imaging data remains at the existing Eunomia-backed complex formula or dense
storage boundary.

## Rejected alternative

Keeping millimetre scalars in the public struct would preserve a unit contract
only in comments and permit accidental mixing with SI `Length`. Adding a
forwarding raw-scalar constructor would retain the same ambiguity and violate
the repository's no-compatibility-facade migration rule.

## Verification

All in-repository targeting callers and tests compile with the typed contract.
The package test-target check passes; focused targeting Nextest passes 10/10
with 184 skipped; warning-denied all-target Clippy passes; doctests pass with 1
executable and 5 ignored; RustDoc, formatting, and diff checks pass. The
residue audit finds no raw public coordinate, confidence, distance, or Bregma
field; scalar extraction is limited to the atlas millimetre boundary and
value-semantic tests. Shared unused-provider-patch and linker warnings remain
outside this decision.
