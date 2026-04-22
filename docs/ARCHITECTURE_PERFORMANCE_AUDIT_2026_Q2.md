# Architecture & Performance Audit Report — Q2 2026

**Audit Date:** 2026-04-04  
**Scope:** `kwavers`, `pykwavers`, `apollo`, `gaia`, `ritk`  
**Workspace Root:** `D:\\kwavers`

## Executive Summary

The repository is in an active migration state. The CPU-side `kwavers` library can be brought to
compile and execute canonical photoacoustic tests, but the repo remains architecturally mixed:

- partial canonical verticalization exists for photoacoustics
- active parity infrastructure exists for `pykwavers` and `external/k-wave-python`
- Apollo already owns most reusable FFT planning
- GPU/WGPU paths are not yet aligned with the current API version and currently block the
  GPU-enabled `pykwavers` build
- the worktree is heavily dirty and contains both tracked legacy drift and untracked tranche work

## Current Build Matrix Snapshot

| Target | Status | Notes |
|---|---|---|
| `cargo check -p kwavers --lib` | Passes | Warning debt remains |
| `cargo test -p kwavers --test photoacoustic_vertical` | Passes | Monte Carlo and diffusion canonical slice pass |
| `cargo check -p pykwavers --no-default-features` | Passes | Minimal CPU binding path is viable |
| `cargo check -p pykwavers` | Fails | GPU/WGPU tranche blocked |
| `cargo check -p pykwavers` with default features before tranche reset | Fails | Default path incorrectly included `gpu` |

## Architecture Findings

### 1. Canonical verticals

Present and partially established:

- `kwavers/src/domain/imaging/photoacoustic/`
- `kwavers/src/physics/photoacoustics/`
- `kwavers/src/solver/photoacoustics/`
- `kwavers/src/simulation/photoacoustics/`

Still duplicated elsewhere:

- `kwavers/src/solver/multiphysics/photoacoustic.rs`
- `kwavers/src/physics/electromagnetic/photoacoustic.rs`
- `kwavers/src/simulation/modalities/photoacoustic/*`

### 2. Shared infrastructure

Current ownership intent is clear but not yet fully enforced:

- FFT ownership is moving to `apollo`
- reusable geometry should move to `gaia`
- reusable registration should move to `ritk`
- `kwavers::math::fft` still contains compatibility scaffolding and downstream legacy imports

### 3. Python surface

`pykwavers` is still monolithic (`pykwavers/src/lib.rs`) and exposes a mixed legacy/canonical
surface. The minimal CPU binding path works; the GPU-enabled path is deferred.

## Performance Findings

### CPU path

- current CPU canonical photoacoustic slice executes successfully
- warning debt remains around dead compatibility imports and some unused fields
- steady-state memory ownership is not yet systematically documented at the module level

### GPU path

Observed failures indicate current WGPU API drift including:

- removed or renamed `wgpu::Maintain`
- changed adapter/device request signatures
- updated `DeviceDescriptor` requirements
- stale queue/device ownership assumptions
- drift in backend error/config contracts

This confirms the program decision to defer GPU closure until CPU/reference correctness is closed.

## Scientific Validation Findings

Positive:

- canonical photoacoustic vertical tests exist and pass for the current CPU slice
- parity scaffolding and external references are present in the repo

Gaps:

- whole-repo scientific documentation is inconsistent
- many public modules still lack theorem/algorithm/proof-grade rustdoc
- large portions of the repo have no clear literature-validation status ledger

## Risk Summary

### High risk

- heavy dirty worktree means unrelated edits can collide with tranche work
- stale documentation can overstate current scientific completeness
- GPU-enabled Python builds currently misrepresent default build health if `gpu` is included by default

### Medium risk

- duplicate ownership across legacy and canonical verticals
- compatibility-layer residue in FFT and signal-processing paths
- monolithic `pykwavers` binding layout slowing API cleanup

## Recommended Immediate Actions

1. Freeze tranche-one acceptance on CPU/reference correctness and canonical ownership.
2. Default `pykwavers` to the minimal CPU path until the GPU tranche is complete.
3. Produce crate and vertical inventories before broader deletion/migration work.
4. Continue Apollo ownership migration and remove compatibility scaffolding incrementally.
5. Close the photoacoustic vertical scientifically before proceeding to broader acoustic-core closure.

## Audit Conclusion

The repository is suitable for a phased scientific remediation program, but not for all-at-once
claims of whole-repo closure. CPU/reference and canonical ownership work should continue first;
GPU modernization remains a separate mandatory tranche.
