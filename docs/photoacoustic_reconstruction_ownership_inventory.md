# Photoacoustic Reconstruction Ownership Inventory

## Canonical public owners

- `kwavers/src/solver/photoacoustics/reconstruction/`
- `kwavers/src/solver/inverse/reconstruction/photoacoustic/`

## Retained algorithms

- line-sensor FFT reconstruction
- planar-sensor FFT reconstruction
- time-reversal reconstruction
- universal back-projection support in the inverse reconstruction layer

## Ownership rule

The simulation-facing canonical owner is `solver/photoacoustics/reconstruction`.
Low-level inverse algorithms may remain in `solver/inverse/reconstruction/photoacoustic`,
but canonical modality orchestration must depend on them through the retained vertical.
