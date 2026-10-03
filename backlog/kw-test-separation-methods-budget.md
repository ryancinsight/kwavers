<a id="kw-test-separation-methods-budget"></a>

## KW-TEST-SEPARATION-METHODS-BUDGET — `test_all_separation_methods` exceeds the nextest slow bound [patch] — todo

priority: verification; needs: none; scope: `crates/kwavers/tests/clutter_filter_integration.rs` (`test_all_separation_methods`), `crates/kwavers-analysis/src/signal_processing/clutter_filter/adaptive_filter/`

- **Finding.** `clutter_filter_integration::test_all_separation_methods` hit the 60 s nextest termination under host load on 2026-10-02. It builds `generate_fus_data(30, 120, 5.0, 0.5, 0.02, 0.15)` and runs `AdaptiveFilter::filter` once per `SubspaceSeparationMethod` (`FixedRank { clutter_rank: 2 }`, `AdaptiveThreshold { decay_factor: 0.1 }`, `CbrBased { target_cbr_db: 25.0 }`).
- **Second defect in the same test:** all three results are bound to `_result1`..`_result3` and never asserted, so the test passes whenever the three calls return `Ok`. A value-semantic check per method (the separated clutter rank or the rejection it must achieve, with a bound derived from the generated data) belongs in the same change.
- **Outcome:** the unsimplified test (same data size and the same three methods) completes inside the 30 s slow bound under the committed nextest profile, with production code optimized rather than the bound or the workload changed.
- **Next step:** profile the three calls separately (sampling profiler or `tracing` spans around `AdaptiveFilter::filter`) on an unloaded host and under load, record the dominant cost, then optimize that component.
- **Acceptance:** `cargo nextest run -p kwavers --test clutter_filter_integration` passes with `test_all_separation_methods` under 30 s; `.config/nextest.toml` and the generated-data parameters are unchanged; the three methods each carry a value-semantic assertion.
