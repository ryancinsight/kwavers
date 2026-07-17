# ADR 037: Retire the stale Apollo GPU availability probe

## Status

Accepted

## Context

`kwavers_math::fft::gpu_fft_available` delegated to a provider function that
Apollo no longer exposes. Apollo now requires callers to acquire a typed
Hephaestus `WgpuDevice`, construct `WgpuBackend`, and propagate device or plan
creation failures. The Kwavers wrapper has no in-repository callers and turns
an error-bearing provider boundary into a boolean.

## Decision

Delete the wrapper and its public re-export. Consumers construct the Apollo
backend from their provider-acquired device and handle the backend/plan result
at their own error boundary.

## Consequences

This is a breaking API removal. No compatibility re-export or boolean fallback
remains. The all-feature Clippy gate compiles the consumer against Apollo's
current public contract.
