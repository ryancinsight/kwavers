# GPU Ownership Inventory

## Canonical owners

- `apollofft-wgpu` for FFT backend execution
- `kwavers/src/gpu` and `kwavers/src/solver/backend/gpu` for solver orchestration and non-FFT kernels

## Photoacoustic tranche truth

- canonical CPU photoacoustic path compiles and tests pass
- GPU capability reporting is present in the canonical photoacoustic solver vertical
- full canonical GPU acoustic propagation remains an open implementation gate

## Acceptance note

Photoacoustic acceptance remains blocked until the retained GPU path is implemented, validated, and benchmarked.
