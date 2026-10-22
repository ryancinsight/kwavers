<a id="kw-py-thermal-offset"></a>

## KW-PY-THERMAL-OFFSET — PSTD thermal output from Python sits 273.15 K above the initial temperature [patch] — todo

priority: correctness; needs: none; scope: `crates/kwavers-python/src/simulation_py/solvers/result.rs`, the PSTD thermal coupling in `kwavers-solver`, `crates/kwavers-python/tests/validate_thermal_acoustic_coupling.py`

- **Finding (basis `0da4122f3`):** `validate_thermal_acoustic_coupling.py` was run against a freshly built extension. It fails at line 115 with ΔT_peak = 273.150000 °C, both on main and with the KW-POWER-LAW-SWEEP binding. `result.rs:93` subtracts the Kelvin offset once at the Python boundary, so the solver field holds T₀ + 2·273.15, meaning one °C→K conversion too many upstream. The script exits 0 at import when `pykwavers` is missing and is not collected as `test_*`, so no gate runs it.
- **Acceptance:**
  - At t = 0, `thermal_temperature` equals `initial_temperature` exactly.
  - The script's ΔT bound holds.
  - The check runs in the collected pytest suite.
- **Next step:** trace `initial_temperature` from `configuration.rs:57` into the PSTD thermal state and find the second offset.
