<a id="kw-gpu-device-error-substitute"></a>

## KW-GPU-DEVICE-ERROR-SUBSTITUTE — No test fails when a GPU PSTD request falls back to Leto on a device error [patch] — todo

priority: verification; needs: none; basis: b3b2c3a372c8ceaf6f330637383e4d2cd11c43c4; scope: `crates/kwavers-simulation/src/dispatch/pstd/gpu.rs`, `crates/kwavers-simulation/src/solver_adapters/gpu_pstd.rs`, `crates/kwavers-simulation/src/runner.rs`

- **Origin.** Split from `KW-GPU-048`, which #1053 closes. ADR 037 says the runner "returns typed errors for unsupported inputs or acquisition failure". #1053 pins refusals that precede device acquisition: an `InvalidInput` and a `FeatureNotAvailable` through the runner, and the same two through the factory. A fall-back to Leto taken only when the provider returns a device error is not covered. The doc comment of `run_gpu_pstd` names two: `SystemError::GpuNotAvailable` on a host without an adapter and `KwaversError::GpuError` for an adapter that fails. On a host whose adapter works neither is returned (derived), and no workflow runs the tests of the `gpu` build of `kwavers-simulation`.
- **Read at the basis.** `git grep -n -i acquisition -- crates/kwavers-simulation/src` finds three doc comments, three test names and one `expect_err` message; no test makes acquisition fail. `run_gpu_pstd` fixes the device provider (`WgpuPstdStateProvider`); `run_gpu_pstd_with_provider` (`crates/kwavers-gpu/src/pstd_gpu/runner.rs`) takes it as a type parameter.
- **Acceptance:** a test in which the provider returns a device error for a `SolverType::PSTD` request with `FftBackend::Hephaestus` asserts that error's variant and message, runs where the gate runs, and fails when the dispatch or the adapter returns a Leto result instead.
- **Next step:** choose between a provider type parameter on the dispatch and the adapter, and a test for the adapter-less hosted runner; the second needs a workflow that runs the `gpu` build's tests (`KW-SIM-TEST-COMPILE-GRAPH`).
