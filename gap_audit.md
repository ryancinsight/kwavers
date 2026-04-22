# Gap Audit

## Initial Findings (Sprint Start)
- Current structure claims zero circular dependencies, but the user directive overrides: circular dependencies and cross-contamination *are* currently present and must be resolved.
- Required to validate `pykwavers` against `k-wave-python`. Needs formal suite for 1-to-1 parity mapping.
- Deep nested file structures (3-5+ levels) required but may not be fully conforming.
- `pykwavers` needs to solely represent thin `PyO3` wrappings over `kwavers`, any core fixes must be bubbled down to `kwavers` itself.

## Priority Matrix
1. [Highest] Locate and prune circular dependencies and duplicate/inconsistent implementations across core modules.
2. [High] Define the validation suite matrix for Grid, Source, Signal, Sensor, Solver.
3. [Medium] Ensure GPU support is routed correctly via BURN crate.

## Resolved Since Audit Start
- Tone-burst sample count now follows the k-wave-python floor-plus-one endpoint rule.
- `pykwavers.tone_burst()` now defaults to the k-wave Gaussian envelope and matches the vendored reference waveform after flattening the row vector.
- Seismic FWI now uses the physical L2 residual `d_syn - d_obs`, receiver-order time-reversed adjoint injection, discrete objective scaling by `dt`, CFL validation, and a finite-difference second-derivative gradient path with value-based tests.
- Reconstruction FWI now uses sign-correct residuals, `dt`-scaled objectives, validated wavefield timesteps and geometry, checkpointed adjoint replay, and a real encoded-gradient aggregation contract; the fabricated Hessian proxy was removed.
- The acoustic adjoint-state core is now single-sourced for L2 residuals, objective scaling, time reversal, and signed-correlation accumulation across seismic and reconstruction FWI.
- The acoustic GPU compute path now uses valid 8×8×4 workgroups, a 64-byte WGSL uniform layout, direct shape construction on readback, and a fused velocity update that removes three transient gradient arrays.
- The GPU profiling module is now publicly exported as `kwavers::profiling`, the obsolete tracker-injected GPU constructors were removed, and the current FDTD pressure roundtrip path is validated against the updated constructors.
- The beamforming delay-and-sum readback path now reconstructs the 3D volume without a zero-fill pass, handles `device.poll` results explicitly, and the k-space spectrum shift path now uses coercions instead of redundant casts.
- The FDTD solver now reuses staggered divergence scratch state instead of cloning a temporary divergence volume, the scalar dispatcher no longer performs a redundant full-volume zero-fill, and the GPU in-place overwrite path zeros only boundary faces before stencil writes.
- The vendored `k-wave-python` example parity suite is closed for the 2-D FFT line-sensor example: `kspace_line_recon` is implemented natively, the non-square 2-D FFT axis ordering is corrected, and the parity metric now compares reconstructed images with `pearson_r ≈ 0.99999999`.
- `kwavers` no longer contains direct `rustfft` or `FftPlanner` usage in source, tests, or benches; all transform entry points route through Apollo-backed FFT APIs.
- The Apollo GPU FFT backend now dispatches radix stages with explicit pass boundaries and passes the 128³ kwavers parity check under a hybrid absolute/relative spectral metric.
