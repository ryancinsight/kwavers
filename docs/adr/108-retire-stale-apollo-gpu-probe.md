# ADR 108: Retire the stale Apollo GPU availability probe

## Status

Accepted

- Revision (2026-08-26): ADR 125 removed the remaining Apollo GPU facade and
  routed PSTD directly to Hephaestus. The no-boolean-probe decision remains;
  provider acquisition now follows the Hephaestus selector contract below.

## Context

`kwavers_math::fft::gpu_fft_available` delegated to a provider function that
Apollo no longer exposes. Apollo now requires callers to acquire a typed
Hephaestus `WgpuDevice`, construct `WgpuBackend`, and propagate device or plan
creation failures. The Kwavers wrapper has no in-repository callers and turns
an error-bearing provider boundary into a boolean.

## Decision

Delete the wrapper and its public re-export. Simulation consumers select
`FftBackend::Hephaestus`; direct device consumers construct a Hephaestus
provider and handle acquisition, planning, and execution errors at their own
operation boundary. No boolean capability proxy or Apollo GPU facade remains.

## Consequences

This is a breaking API removal. No compatibility re-export, boolean fallback,
or indirect Apollo GPU route remains. The selector preserves requested-backend
failures instead of silently changing execution providers.
