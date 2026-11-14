<a id="kw-sim-sensor-mask-doc"></a>

## KW-SIM-SENSOR-MASK-DOC — The Rustdoc of `SimulationRunRequest::sensor_mask` says no sensors are requested by `None`, which is false beside `transducer_ordered_indices` [patch] — todo

priority: correctness; needs: none; basis: 7e9932f677a607cba87406c4adccb0c1460e4f1b; scope: `crates/kwavers-simulation/src/types.rs`, `crates/kwavers-simulation/src/dispatch/pstd.rs`, `crates/kwavers-simulation/src/dispatch/pstd/gpu.rs`

- **Origin.** The judge of #1087.
- **Read at the basis.** `types.rs:111-117` says `None`, and a mask with no true element, request no sensors, the run succeeds, `sensor_data` has shape `(0, steps)`, and "Neither dispatch returns an error for the absence of sensors". With `transducer_ordered_indices` set the Leto PSTD dispatch builds its recorder from those points (`dispatch/pstd.rs:347-349`) and ignores `sensor_mask`, so `sensor_data` has one row per ordered index; the Hephaestus dispatch refuses the request with `FeatureNotAvailable` (`dispatch/pstd/gpu.rs:108-111`). Both contradict the sentence.
- **Outcome:** the Rustdoc states that `transducer_ordered_indices`, when `Some`, takes precedence over `sensor_mask` on the Leto dispatch and is refused by the Hephaestus dispatch, and that the empty-record rule holds only when it is `None`.
- **Acceptance:** a Leto PSTD test with `sensor_mask: None` and one ordered index asserts `sensor_data` has shape `(1, steps)`, and a Hephaestus-dispatch test (adapter-gated, see `KW-GPU-HOST-TESTS-UNRUN`) or the existing refusal test pins the `FeatureNotAvailable` variant; the Rustdoc then matches both.
- **Next step:** find the existing ordered-index and refusal tests in `dispatch/pstd/tests.rs` and `dispatch/pstd/gpu.rs`, then rewrite the sentence.
