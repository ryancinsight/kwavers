<a id="kw-test-cpml-thicknesses-budget"></a>

## KW-TEST-CPML-THICKNESSES-BUDGET — `test_cpml_stable_across_thicknesses` runs 45 s against a 10 s slow bound [patch] [perf] — todo

priority: verification; needs: none; scope: `crates/kwavers/tests/cpml_absorption_quality.rs` (`test_cpml_stable_across_thicknesses`, `run_cpml_absorption`) and the CPML and solver step path it exercises

- **Finding.** In the parallel nextest suite `cpml_absorption_quality::test_cpml_stable_across_thicknesses` took 45 s, against the 10 s slow bound and the 60 s termination (`.config/nextest.toml`). It calls `run_cpml_absorption` for the four thicknesses `[6, 8, 10, 12]` in sequence (`cpml_absorption_quality.rs:207`) and asserts finite energy, energy decay and monotone absorption.
- **Outcome:** the unsimplified test (same four thicknesses, same grid and step count) fits the 10 s slow bound under the committed profile, with the production path optimized and neither the bound, the thicknesses nor the workload changed.
- **Next step:** profile one `run_cpml_absorption` call (sampling profiler or `tracing` spans) alone and under parallel load, record the dominant cost against an analytical bound for the grid and step count, then optimize that component, as kwavers#926 did for the adaptive filter.
- **Acceptance:** `cargo nextest run -p kwavers --test cpml_absorption_quality` reports `test_cpml_stable_across_thicknesses` under 10 s with `.config/nextest.toml` and the test body unchanged.
