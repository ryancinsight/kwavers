<a id="kw-fdtd-ordered-indices-ignored"></a>

## KW-FDTD-ORDERED-INDICES-IGNORED — The FDTD dispatch ignores `transducer_ordered_indices` once a recognised recording mode is set, and its Rustdoc is silent on FDTD [minor] — todo

priority: correctness; needs: none; basis: fd072d51936f9446ab1b62877772154aaf609d7a; scope: `crates/kwavers-simulation/src/dispatch/fdtd.rs`, `crates/kwavers-simulation/src/dispatch/fdtd/tests.rs`, `crates/kwavers-simulation/src/types.rs`, `crates/kwavers-receiver/src/recorder/simple/construction.rs`; milestone: M-solvers-core

- **Origin.** Filing sweep. Both PSTD dispatches settle a request's ordered indices (honor or refuse); the FDTD dispatch does neither once a mode is set.
- **Read at the basis.** `dispatch/fdtd.rs:55-66`: when `recording_modes_from_strings(&req.record_modes)` is non-empty (a string among `p_max`, `p_min`, `p_rms`, `p_final`, `all`; others are dropped, `dispatch/shared.rs:57-69`) the recorder is `SensorRecorder::with_modes(req.sensor_mask..)` and `transducer_ordered_indices` is never read; only the `else if` branch (no mode) calls `from_ordered_indices`. A caller's ordered request then returns the mask's x-fastest row order, or `(0, steps)` for `sensor_mask: None`, with no error. Leto PSTD gives the indices precedence (`dispatch/pstd.rs:347-349`; it drops the modes' statistics) and Hephaestus PSTD refuses with `FeatureNotAvailable` (`dispatch/pstd/gpu.rs:108-111`). `types.rs:120-129` documents both PSTD cases and says nothing of FDTD. Reachable from Python: `execute.rs:47-49` takes `record_modes` from the call, `execute.rs:93-96,129-130` always forwards the transducer's ordered indices. `dispatch/fdtd/tests.rs` pins neither.
- **Outcome:** FDTD either honors the indices (one pressure row per index in the supplied order, with `SensorRecorder` gaining a constructor that pairs ordered indices with a recording spec so the modes' statistics are kept) or refuses a `Some` value with `KwaversError::FeatureNotAvailable`; it never ignores them. The `types.rs` Rustdoc states which.
- **Acceptance:** a test in `dispatch/fdtd/tests.rs` runs FDTD with `transducer_ordered_indices: Some(vec![(5, 1, 1), (1, 1, 1)])`, `sensor_mask: None`, and `record_modes: vec!["p_max".to_owned()]`, and asserts either `sensor_data.shape() == [2, steps]` with row 0 equal to the single-index run at `(5, 1, 1)`, or `FeatureNotAvailable`. It fails at the basis (`shape() == [0, steps]`, `Ok`). A second case with no mode pins the ordered record that `fdtd.rs:64-66` already returns.
- **Next step:** read `SensorRecorder::with_spec` and `from_ordered_indices` (`construction.rs:123,200`); choose honor unless the statistics cannot be kept, then add the test first. `KW-SIM-SENSOR-MASK-DOC` also edits `types.rs:111-129`; sequence the Rustdoc after it.
- **Candidate evidence.** On main 45db5f1d0 the three regression cases fail for missing ordered rows, an ignored empty selection, and an accepted out-of-grid coordinate. The candidate passes all three against single-point mask recordings and retains full-grid statistics. It constructs one recorder and skips cloning an overridden mask.
- **Package gates.** The combined receiver and simulation library suites pass
  163/163 tests with zero skips. The removed pressure buffer for a collection of
  S sensors and N samples occupied 8 × S × N bytes, plus its allocation overhead.
  Allocation profiling of representative jobs remains open.
- **Next step:** coordinate the shared request documentation with PR #1104 and
  rerun the exact integrated head before closure.
