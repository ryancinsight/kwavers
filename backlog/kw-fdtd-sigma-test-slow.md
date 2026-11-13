<a id="kw-fdtd-sigma-test-slow"></a>

## KW-FDTD-SIGMA-TEST-SLOW — The FDTD sigma-factor test runs at two thirds of its 60 s bound in a parallel run [patch] [perf] — todo

priority: tightening; needs: none; basis: 38eaf2f90eccd74d11ce3d0e836cbad7d4606ae0; scope: `crates/kwavers-solver/src/forward/fdtd/solver/tests.rs` (`sigma_factor_three_outperforms_the_kwave_default_at_grazing_incidence`, line 1593) and the FDTD step it runs

- **Finding.** The test calls `pml_grazing_reflection_sf(10, 20, 110, ...)` twice (`tests.rs:1593-1617`). It matches none of the default profile's per-test overrides (`.config/nextest.toml:94-96`, `:117-123`), so its termination is the default 10 s x 6 = 60 s (`:18`).
- **Measured, not re-run here.** kwavers#1066 records 40.5 to 41.7 s in the parallel `-p kwavers-solver -p kwavers-simulation` run (1243 tests) against 24.157 s before its change, and 13.727 s in CI run 43271418019. A host slowdown of 1.44x or more from the parallel figure terminates the test during a pre-push gate.
- **Target the repository already states.** `.config/nextest.toml:77-78` names an ordinary 30 s performance target; the test exceeds it in the parallel run.
- **Acceptance:** the test's per-test time in the same parallel run, recorded with the command, host class and revision, is under 30 s with no override added, no bound raised and no assertion or grid reduced; the 0.2 ratio assertion and the `residual <= 1e-8 * direct_default` check are unchanged.
- **Next step:** profile one `pml_grazing_reflection_sf` run (cells, steps, per-step cost) and cut the work in the solver step or the test path.
