<a id="kw-sim-hybrid-runner-dispatch"></a>

## KW-SIM-HYBRID-RUNNER-DISPATCH — The runner sends a Hybrid request to the PSTD dispatch [patch] — todo

priority: correctness; needs: none; basis: b3b2c3a372c8ceaf6f330637383e4d2cd11c43c4; scope: `crates/kwavers-simulation/src/runner.rs`, `crates/kwavers-simulation/src/solver_factory.rs`

- **Origin.** Found while reading the runner for #1053; read, not run.
- **Read at the basis.** `SimulationRunner::run` maps `SolverType::Hybrid` with `FftBackend::Leto` to `dispatch::pstd::run`, or to `run_with_thermal` when the request carries thermal coupling (`runner.rs` lines 43 to 49). `crates/kwavers-simulation/src/dispatch/pstd.rs` does not contain `solver_type`. For the same selection `SimulationSolverFactory` builds `HybridSolver::new(hybrid_config_from(&config), grid, medium)` (`solver_factory.rs` lines 174 and 175). Whether a Hybrid request and a PSTD request return different sensor data through the runner is not known; no run was made.
- **Acceptance:** the runner and the factory agree on what `SolverType::Hybrid` executes: a test sends one Hybrid request through the runner and fails when the PSTD dispatch answers it, or the runner returns a typed `FeatureNotAvailable` for the selection and a test asserts its message.
- **Next step:** run one request as `Hybrid` and as `PSTD` through the runner and compare the sensor data; read `git log -S"SolverType::Hybrid" -- crates/kwavers-simulation/src/runner.rs` for the intent of the arm.
