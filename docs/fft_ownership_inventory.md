# FFT Ownership Inventory

## Canonical owner

- `apollo/crates/apollofft`
- `apollo/crates/apollofft-wgpu`

## Allowed `kwavers` responsibilities

- thin adapter imports
- solver-layer trait consumption through `solver::interface::FourierBackend`
- modality-specific orchestration

## Current retained photoacoustic consumers

- `kwavers/src/solver/inverse/reconstruction/photoacoustic/fourier.rs`
- `kwavers/src/solver/photoacoustics/reconstruction/*`

## Non-canonical pattern to remove over time

- direct planner/cache ownership inside `kwavers` algorithms
- backend-specific branching in modality code
