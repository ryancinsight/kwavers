<a id="kw-sim-pstd-no-sensor-contract"></a>

## KW-SIM-PSTD-NO-SENSOR-CONTRACT — The two PSTD dispatches differ on a request without sensors [patch] — todo

priority: correctness; needs: none; basis: b3b2c3a372c8ceaf6f330637383e4d2cd11c43c4; scope: `crates/kwavers-simulation/src/dispatch/pstd.rs`, `crates/kwavers-simulation/src/dispatch/pstd/gpu.rs`

- **Origin.** Observed in a probe for #1053, on a Windows host with a GPU adapter.
- **Observed.** One request (7×4×3 grid, three steps, one source, no sensor mask) through `SimulationRunner::run`: with `FftBackend::Leto` it returned an error printed as "I/O error"; with `FftBackend::Hephaestus` it returned `Ok` with sensor data of shape `[0, 3]`. A mask with no true element gave the same two results.
- **Read at the basis.** `dispatch/pstd.rs` replaces a missing mask by one that is false everywhere (lines 188 to 191) and returns `KwaversError::Io` with "No sensor data recorded" when the recorder has no pressure view (line 360).
- **Acceptance:** both backends return the same kind of result for a request without sensors, an empty record or one typed `InvalidInput`; the choice is stated in the Rustdoc of `SimulationRunRequest` and one test per backend asserts it.
- **Next step:** read which callers of the runner pass no sensor mask and what they read from the result.
