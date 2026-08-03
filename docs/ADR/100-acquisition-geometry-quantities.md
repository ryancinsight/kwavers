# ADR 100 — Aequitas shared acquisition-geometry quantities

Status: Accepted — 2026-08-03

## Context

The shared `TransducerGeometry` contract exposes Cartesian element positions
through `ElementPosition`, but each coordinate is a raw metre scalar. The
transcranial bowl and multi-row ring implementations also accept raw radius,
diameter, and row-spacing values. These contracts feed generic Born/PCG and
frequency-domain FWI paths, where unit-bearing geometry is the stable public
boundary and scalar values are required only by numerical formulas, mesh
indexing, and trigonometric operations.

## Decision

Use Aequitas `Length<f64>` for the `x`, `y`, and `z` coordinates of
`ElementPosition`, and for bowl radius plus ring diameter and row spacing.
Migrate every direct caller in the transducer, diagnostics, physics, solver,
and Python adapter closure in one change. Extract metres only at the existing
Euclidean, rotation/trigonometric, mesh/index, and numerical-kernel
boundaries. Do not retain raw-field aliases, forwarding constructors, or
compatibility wrappers.

## Eunomia compatibility

Acquisition geometry is a real spatial quantity. A downstream complex signal
may retain real and quadrature components under one observable signal unit,
but neither component changes the geometry dimension. No imaginary SI length
or complex-valued physical coordinate is introduced.

## Verification

Analytical ring coordinates, row spacing, bowl radius, topology, and invalid
parameter tests must preserve their value semantics. Focused transducer,
physics, diagnostics, solver, and Python-boundary checks must pass through the
repository's sanctioned runners, followed by formatting, Clippy, Rustdoc,
residue scans, and the hosted repository-owned matrix.

Refs: `KWAVERS-AEQ-MET-61` in `backlog.md`.
