<a id="kw-ci-053"></a>

## KW-CI-053 — Update GPU PSTD parity contract [patch] — review

priority: verification; needs: none; scope: `crates/kwavers/tests/gpu_pstd_parity.rs` (removed), `.github/workflows/gpu-parity.yml`

- **Acceptance:** the ignored GPU parity tests call the provider-owned six-argument `GpuPstdSolver::run` API with `PstdOutputRequest::sensor_traces()` and consume `sensor_data`, with no compatibility wrapper or test simplification.
- **Re-validation finding (2026-10-03):** `gpu_pstd_parity.rs` no longer exists at origin/main (removed by 42be027a7d9); `gpu-parity.yml` remains.
- **Next step:** confirm the GPU parity coverage `gpu-parity.yml` runs calls the current `run` signature, then close.
