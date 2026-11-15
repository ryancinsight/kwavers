<a id="kw-pstd-unit-exponent-backends"></a>

## KW-PSTD-UNIT-EXPONENT-BACKENDS — CPU and GPU PSTD disagree on a configured y = 1 under causal dispersion [patch] — todo

priority: correctness; needs: KW-PY-GPU-ABSORPTION; scope: crates/kwavers-solver/src/forward/pstd/physics/absorption/init.rs, crates/kwavers-gpu/src/pstd_gpu/medium.rs, crates/kwavers-python/tests/test_gpu_pstd_unit_exponent.py

- **Finding.** The two backends handle a configured `alpha_power = 1` under `Causal` differently when the medium supplies a non-degenerate exponent:
  - the CPU rejects it, because it validates the configured exponent whenever absorption is active (`init.rs:153`);
  - the GPU accepts it, because it validates the configured exponent only where a voxel falls back to it (`medium.rs:58-77`).

  #966 made `run()` forward the configuration, so the difference is now reachable.
- **Finding.** The run-path pytest compares with an underived `1e-12·peak` (lines 242, 249), although both paths build identical fields. It reuses `ABSORPTION_VISIBILITY` for an unrelated separation. No `run()`-level test covers GPU dispatch.
- **Next step.** Pick one rule: validate the configured exponent only where it is used. Apply it in both backends. Assert exact equality in the pytest. Add a GPU `run()` case.
- **Acceptance.** Both backends accept or reject the same configurations. The pytest has no underived tolerance.
