<a id="kw-pstd-zero-time-step"></a>

## KW-PSTD-ZERO-TIME-STEP — The Leto PSTD path accepts a time step of zero and zero steps [patch] — todo

priority: correctness; needs: none; basis: b3b2c3a372c8ceaf6f330637383e4d2cd11c43c4; scope: `crates/kwavers-solver/src/forward/pstd/config.rs`, `crates/kwavers-simulation/src/dispatch/pstd.rs`

- **Origin.** Observed in a probe on the basis tree and in a mutant run of #1053, on a Windows host with a GPU adapter and toolchain 1.97.0.
- **Observed.** `PSTDSolver::new` returned `Ok` for a configuration with `dt: 0.0` (the debug output of a failing `unwrap_err` in a mutant run). A request with `dt = 0.0`, three steps, a source and a sensor on a 7×4×3 grid came back from the Leto dispatch as `Ok` with sensor data of shape `[1, 3]`; the values were not read. The same request with zero steps and a time step of 1e-7 came back as `Ok` of shape `[1, 0]`. The GPU provider refuses both: "GPU PSTD requires time_steps > 0 and finite positive dt" (`crates/kwavers-gpu/src/pstd_gpu/runner.rs`).
- **Acceptance:** the Leto path returns a typed `InvalidInput` for a time step that is zero, negative or not finite, asserted by message through the solver constructor and through the runner; the zero-step case is decided, stated in the dispatch Rustdoc and asserted.
- **Dependency to carry.** Two refusal tests of #1053 (zero steps through the runner, a zero time step through the factory) send requests for which Leto returns `Ok`. Observed there on the runner path: with two requests for which the Leto dispatch returned an error, a fall-back that keeps the refusal when Leto fails too passed all 119 tests. The change that makes Leto refuse these inputs gives those two tests requests Leto still accepts, or a failing provider.
- **Next step:** read where `PSTDConfig` is validated and whether any caller passes zero on purpose.
