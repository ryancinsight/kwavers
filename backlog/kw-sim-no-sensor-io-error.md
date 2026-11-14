<a id="kw-sim-no-sensor-io-error"></a>

## KW-SIM-NO-SENSOR-IO-ERROR — The FDTD dispatch and the Python PSTD checkpoint restore still fail a no-sensor request with an `Io` error [patch] — todo

priority: correctness; needs: none; basis: 7e9932f677a607cba87406c4adccb0c1460e4f1b; scope: `crates/kwavers-simulation/src/dispatch/fdtd.rs`, `crates/kwavers-python/src/simulation_py/solvers/pstd.rs`

- **Origin.** The judge of #1087, which made the PSTD dispatch return an empty record for a request with no sensor.
- **Read at the basis.** `dispatch/fdtd.rs:94-96` maps a `None` from `extract_recorded_sensor_data()` to `KwaversError::Io("No sensor data recorded")`; `simulation_py/solvers/pstd.rs:285-287` maps a `None` from `run_from_checkpoint_loaded` the same way. The PSTD dispatch returns shape `(0, steps)` for that request (`dispatch/pstd.rs:371-377`), and an `Io` error is not a typed answer to "no sensors requested".
- **Outcome:** an FDTD run and a Python PSTD checkpoint restore with no sensor return an empty record, as the PSTD dispatch does, or a typed validation error if the solver contract forbids it; no `Io` error names absent sensors.
- **Acceptance:** one test per path with `sensor_mask: None` asserts `sensor_data` has shape `(0, steps)` (or the typed error's variant) and fails on the current `Io` error.
- **Next step:** read what `extract_recorded_sensor_data` and `run_from_checkpoint_loaded` return for a recorder with no sensor indices.
