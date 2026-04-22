# Geometry and Meshing Ownership Inventory

## Canonical owner

- `gaia`

## `kwavers` responsibilities

- modality-specific geometry assembly
- scenario-level sensor position specification
- no generic mesh-generation kernels as public canonical ownership

## Current tranche truth

- photoacoustic sensor geometry is still represented as explicit Cartesian positions
- a `MeshProvider` trait now exists in the solver interface layer for future Gaia integration
