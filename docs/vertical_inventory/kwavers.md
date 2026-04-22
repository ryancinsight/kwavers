# `kwavers` Inventory

## Overall status

`kwavers` is the canonical scientific core, but it currently contains a mix of:

- canonical candidates
- legacy duplicate verticals
- compatibility adapters
- incomplete prototypes

## Canonical candidates

- `domain/imaging/photoacoustic/`
- `physics/photoacoustics/`
- `solver/photoacoustics/`
- `simulation/photoacoustics/`
- `solver/forward/fdtd/`
- `solver/forward/pstd/`
- `domain/grid/`
- `domain/field/`

## Legacy duplicates or competing owners

- `solver/multiphysics/photoacoustic.rs`
- `physics/electromagnetic/photoacoustic.rs`
- `simulation/modalities/photoacoustic/*`
- duplicate FFT compatibility usage spread across analysis and inverse modules

## Incomplete or high-risk areas

- `gpu/`
- `solver/backend/gpu/`
- `solver/forward/pstd/gpu_pstd/`
- monolithic and partially duplicated inverse/reconstruction surfaces
- signal-processing modules with legacy FFT compatibility imports

## Immediate tranche-one priorities

1. maintain CPU/reference-correct canonical photoacoustic ownership
2. continue Apollo FFT migration
3. close shared field/grid ownership
4. reduce warning debt in active canonical paths
5. keep GPU deferred from default acceptance
