# `apollo` Inventory

## Overall status

Apollo is the reusable FFT/NUFFT/planning backend and is already partially adopted by `kwavers`.

## Canonical responsibilities

- FFT/NUFFT plans
- backend capabilities
- CPU/GPU parity helpers
- reusable transform utilities
- validation through `apollo-validation`

## Current migration state

- `kwavers::math::fft` already re-exports Apollo plan types
- active `kwavers` code still contains compatibility-era wrappers and imports
- WGPU-backed Apollo integration exists but whole-repo GPU consumers are not yet aligned

## Immediate tranche-one priorities

1. remove compatibility-era FFT scaffolding from active `kwavers` call sites
2. move broadly reusable spectral helpers into Apollo when appropriate
3. use Apollo validation artifacts as part of FFT correctness evidence
