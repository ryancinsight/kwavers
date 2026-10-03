<a id="kw-sim-test-compile-graph"></a>

## KW-SIM-TEST-COMPILE-GRAPH — Reduce simulation test build latency [patch] [perf] — in-progress

priority: tightening; needs: none; scope: `kwavers-simulation` and `kwavers-python` dependency/feature graphs, `.cargo/config.toml` alias `test-gpu-consumers`, `.github/workflows/compile-timing.yml`

- **Outcome:** reduce the cold compile/link cost of the GPU-enabled simulation and Python test harnesses while retaining every value-semantic test.
- **Entry evidence:** isolated cold runs spent 2m37s and 2m15s compiling the two GPU-enabled graphs against 0.617s and 1.661s of test execution; 431 of 444 Python normal/dev packages are in the simulation graph, so separate invocations rebuild the shared graph.
- **Implemented:** `cargo test-gpu-consumers` in `.cargo/config.toml` runs one Nextest invocation over both packages (the exact 128-ID union of 107 simulation and 21 Python IDs; 127 pass, one declared skip). It is a local gate only; no CI job invokes it.
- **Hosted cold-compile evidence (`compile-timing.yml`, 2026-09-02):** 579 s and 592 s on a 4-core hosted runner against the 181 s acceptance bound (3.2x over). Per the acceptance clause the consolidation is rejected as a CI gate; the alias stays a local instrument.
- **Acceptance:** preserve the 128-ID union, features, assertions, profile, cache policy and timeout bounds; on two fresh four-core Linux runners the combined cold compile must not exceed 181 s, otherwise reject consolidation and use Cargo timings to select one dominant unit.
- **Next step:** `cargo build --timings` over the same graph to attribute the 579 s and select one dominant unit to shrink (a DoR sub-item once collected).
