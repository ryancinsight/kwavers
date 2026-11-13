<a id="kw-py-hybrid-refused"></a>

## KW-PY-HYBRID-REFUSED — Python callers offer a Hybrid solver that `Simulation.run` refuses [patch] — todo

priority: verification; needs: none; basis: 79a20024f0f7d3b35ebbe5d762f2773d7a273319; scope: `crates/kwavers-python/python/pykwavers/comparison.py`, `crates/kwavers-python/tests/`, `crates/kwavers-python/src/solver_type_bindings.rs`, `crates/kwavers-simulation/src/runner.rs`

- **Origin.** `SimulationRunner::run` returns `FeatureNotAvailable` for `SolverType::Hybrid` (it answered with the PSTD dispatch before, and `HybridSolver::new` takes no grid source or sensor recording). The Python callers below were written when Hybrid ran.
- **Callers that fail now (none run in CI).** `tests/test_phase5_features.py:153` (`test_hybrid_solver_implemented`); `tests/test_utilities.py:168` (loop over FDTD, PSTD, Hybrid); `tests/test_pstd_hybrid_solvers.py` lines 261-560 (every test that selects `SolverType.Hybrid`); `python/pykwavers/comparison.py` lines 89, 109, 122, 457, 482, 657-664 (`PYKWAVERS_HYBRID` and the `hybrid` key); `src/solver_type_bindings.rs:11,32` describe the variant as a solver, not as one `Simulation.run` accepts.
- **Also stale.** `xtask/PYKWAVERS_TASKS.md:90,263` list Hybrid in a comparison that `xtask/src` does not implement.
- **Acceptance:** Python callers either run a working Hybrid (the runner passes a request's grid source and sensor recording to `HybridSolver`, in `kwavers-solver`, with a value test against the FDTD or PSTD record) or stop offering it: the three test files assert the refusal message or drop their Hybrid cases, `comparison.py` has no `hybrid` key, and the binding docs say what `Simulation.run` accepts.
- **Next step:** decide between the two outcomes from the cost of routing the recording through `HybridSolver`; the example `compare_all_simulators.py` already omits Hybrid.
