<a id="kw-gpu-adapter-field-states"></a>

## KW-GPU-ADAPTER-FIELD-STATES — GPU PSTD adapter fields before a batch and after a peak run are unspecified [patch] — todo

priority: verification; needs: none; basis: b3b2c3a372c8ceaf6f330637383e4d2cd11c43c4; scope: `crates/kwavers-simulation/src/solver_adapters/gpu_pstd.rs`, `docs/adr/037-gpu-pstd-output-contract.md`

- **Origin.** Found while reading the adapter for #1053; the first verdict there (comment 6810004764, F8) pointed to the peak run.
- **Read at the basis.** `GpuPstdSimulationAdapter::new` fills its pressure and three velocity arrays with zeros (`gpu_pstd.rs` lines 123 to 126), so `pressure_field()` returns zeros before any batch. `run_peak_pressure` (line 153) calls `store_peak_pressure` and not `store_final_fields`, so the four arrays keep what the last `run` stored, or the zeros. The module doc says `pressure_field()` returns the final field "after `run()`", and ADR 037 speaks of the fields after a batch that requests them; neither names these two states. `git grep -n -E "[.]pressure_field\(\)|velocity_fields\(\)" -- crates/kwavers-simulation/src/solver_adapters/gpu_pstd` finds four reads on a new adapter, each of the shape only (`tests.rs` lines 25, 70, 87 and 104), and three reads after `store_final_fields` (lines 130 to 132).
- **Acceptance:** ADR 037 or the Rustdoc of the accessors states what the four fields hold before a batch and after `run_peak_pressure`, and a test without a device asserts each stated value.
- **Next step:** decide between documenting the present behaviour and clearing the fields in `run_peak_pressure`; the second changes what a caller reads after a peak run.
