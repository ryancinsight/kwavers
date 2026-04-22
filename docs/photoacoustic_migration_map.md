# Photoacoustic Migration Map

## Canonical retained owners

- `kwavers/src/domain/imaging/photoacoustic/`
- `kwavers/src/physics/photoacoustics/`
- `kwavers/src/solver/photoacoustics/`
- `kwavers/src/simulation/photoacoustics/`
- `pykwavers/src/bindings/photoacoustic.rs`

## Legacy owners to eliminate from the public surface

- `kwavers/src/simulation/modalities/photoacoustic/*`
- `kwavers/src/solver/multiphysics/photoacoustic.rs`
- `kwavers/src/physics/electromagnetic/photoacoustic.rs`

## Ownership transition rules

### Domain

- Owns scenario/config/material and validation metadata
- Must not depend on legacy modality orchestration

### Physics

- Owns governing equations, thermoelastic reports, confinement checks
- Must not remain the public owner of the legacy electromagnetic photoacoustic module

### Solver

- Owns optical solve orchestration, source generation, propagation, reconstruction, validation, and workspaces
- Must not depend on `simulation::modalities::photoacoustic::*`

### Simulation

- Owns orchestration only through `PhotoacousticRunner`
- Must not publicly re-export the legacy modality photoacoustic facade

### Python

- Binds only canonical `PhotoacousticScenario`, `PhotoacousticRunner`, and `PhotoacousticRunResult`

## Test migration targets

- Rewrite `kwavers/tests/photoacoustic_validation.rs` to canonical scenario/runner usage
- Rewrite `kwavers/tests/photoacoustic_proptest.rs` to canonical scenario/runner usage
- Remove remaining legacy-modality imports from integration tests as they are touched

## Deferred follow-on migrations

- Full deletion of legacy implementation files after all internal callers are removed
- GPU photoacoustic path consolidation
- Apollo FFT ownership tightening for retained transform-heavy stages
