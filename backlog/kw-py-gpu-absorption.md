<a id="kw-py-gpu-absorption"></a>

## KW-PY-GPU-ABSORPTION — Python PSTD bindings keep a second absorption construction and the origin-sampled exponent [patch] — todo

priority: tightening; needs: none; scope: crates/kwavers-python/src/simulation_py/gpu/session/absorption.rs, crates/kwavers-python/src/simulation_py/solvers/pstd.rs, crates/kwavers-gpu/src/pstd_gpu/runner/absorption.rs

- **Finding.** `build_absorption_kernels` in the Python GPU session is a second copy of the GPU PSTD power-law operator construction (`|k|^(y−2)`, `|k|^(y−1)`, τ, η). It keeps its own causal-only `y = 1` guard. `kwavers_gpu::pstd_gpu` owns the same construction in `runner/absorption.rs`.
- **Finding.** `solvers/pstd.rs:150-157` still samples the exponent at the origin (`effective_alpha_power`). The simulation dispatch dropped that rule when it began passing `NonlinearConfig.alpha_power` and `power_law_dispersion` through (KW-NEURO-RESIDUALS).
- **Drift already present.** The runner admits `PowerLawDispersion::Omitted` (η = 0, `y = 1` allowed). The Python copy still rejects `y = 1` unconditionally, and the Python `NonlinearConfig` offers no dispersion choice.
- **Next step.**
  - Have the session call the runner's construction, exposed from `kwavers-gpu` at the narrowest visibility the binding needs, and delete the copy.
  - Pass the configured exponent and dispersion in `solvers/pstd.rs`.
  - Add `power_law_dispersion` to the Python `NonlinearConfig` and its `.pyi` stub.
- **Acceptance.** One construction site. A pytest shows a `y = 1` GPU session without dispersion agreeing with the Rust runner on one sensor trace, to the f32 parity bound in `crates/kwavers-gpu/tests/pstd_parity.rs`.
