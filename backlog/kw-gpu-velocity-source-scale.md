<a id="kw-gpu-velocity-source-scale"></a>

## KW-GPU-VELOCITY-SOURCE-SCALE — GPU PSTD adds velocity sources without k-Wave's 2c₀Δt/Δx scale [patch] — todo

priority: correctness; needs: none; scope: crates/kwavers-gpu/src/pstd_gpu, crates/kwavers-python/src/simulation_py/gpu/session/source.rs

- **Finding.** The CPU solver scales additive velocity sources by `2·c₀·Δt/Δx` (`kwavers-solver/src/forward/fdtd/source_handler/scaling.rs:96-101`). The GPU path adds them unscaled at three points: the session upload (`gpu/session/source.rs:76`), the runner copy (`pstd_gpu/runner.rs:350`) and the shader (`pstd.wgsl:503`). At CFL 0.3 the providers therefore differ by 1/0.6 ≈ 1.667. `pstd_parity.rs` uses pressure sources only, so it cannot see this.
- **Next step.** Apply the scale once, in the runner, from the same helper the CPU uses. Extend `pstd_parity.rs` with a velocity-source case.
- **Acceptance.** GPU and CPU velocity-source traces agree without normalisation, within the f32 parity bound. The ratio test passes at CFL 0.2 and at 0.3.
