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

## Outstanding Gaps (Newly Identified 2026-04-22)
- `at_linear_array_transducer` parity blocked: `KWaveArray::add_rect_element` has no per-element rotation parameter, so tilted linear transducer elements cannot be rasterized.
- `at_focused_annular_array_3D` parity blocked: no `Annulus` element shape; concentric ring BLI rasterization unavailable.
- `at_focused_bowl_AS` / `at_circular_piston_AS` parity blocked: no axisymmetric PSTD solver; only 2-D/3-D Cartesian grids are supported.
- `na_controlling_the_pml` parity blocked: pykwavers solver bindings do not expose PML size/alpha/inside controls.
- `checkpointing` parity blocked: no runtime save/resume for PSTD session state (pressure + velocity + time index).

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
- The vendored `k-wave-python` example parity suite is closed for the 3-D planar-sensor time-reversal example: the forward pressure cache is reused, reconstructed fields are cached in the parity artifacts, and the slow k-Wave reconstruction path no longer runs on every test invocation.
- The vendored `k-wave-python` example parity suite is closed for the 3-D circular piston example: the pykwavers path now uses the native `KWaveArray` disc geometry, zeroes the PML halo before source-weight comparison against the padded reference mask, and matches the analytical on-axis steady-state profile with near-unity Pearson correlation.
- The vendored `k-wave-python` example parity suite is closed for the 3-D focused bowl example: the pykwavers path now uses the canonical spiral/BLI bowl source path, compares the physical-interior source weights against the k-wave-python reference, and validates the on-axis waveform parity with cached comparison artifacts.
- The vendored `k-wave-python` example parity suite is closed for the 2-D focussed detector example: the detector-averaged traces for the on-axis and off-axis source placements now match the k-wave-python reference, and the on-axis/off-axis energy ratio is parity-checked explicitly.
- The vendored `k-wave-python` example parity suite is closed for the 2-D sensor directivity modelling example: the source-angle trace matrix now matches the k-wave-python reference, and the derived directivity curve is parity-checked explicitly.
- The vendored `k-wave-python` example parity suite is closed for `at_array_as_sensor`: the Rust arc rasterizer now matches the upstream line-sampled BLI footprint with the correct `bli_tolerance = 0.05`, the parity bootstrap resolves the rebuilt `target/maturin/pykwavers.dll`, and the raw/combined detector comparisons now pass value-based regression checks.
- The vendored `k-wave-python` example parity suite is closed for `at_array_as_source`: the Rust/Python source matrix now preserves the canonical arc ordering, the example compares exact source-mask and distributed source-signal parity, and the p_max/p_rms field outputs now pass value-based regression checks.
- The vendored `k-wave-python` example parity suite is closed for `us_defining_transducer`: the pykwavers scan-line run now uses the k-Wave reference time-step count, the sensor-trace lengths align exactly, and the per-sensor parity report now passes value-based regression checks.
- The vendored `k-wave-python` example parity suite is closed for `ivp_photoacoustic_waveforms`: the cached initial-pressure traces now match the k-Wave and pykwavers outputs, the single-sensor waveform parity is validated directly, and the PASS report metrics remain intact.
- The vendored `k-wave-python` example parity suite is closed for `us_bmode_phased_array`: the quick steering-angle sweep now matches the cached k-Wave and pykwavers scan lines, the fundamental and harmonic B-mode images pass parity checks, and the GPU profile fields remain reportable.
- The vendored `k-wave-python` example parity suite is closed for `sd_focussed_detector_3D`: the per-source traces now match the k-Wave and pykwavers outputs, the on-axis/off-axis directivity ratio is parity-checked explicitly, and the PASS report metrics remain intact.
- `kwavers` no longer contains direct `rustfft` or `FftPlanner` usage in source, tests, or benches; all transform entry points route through Apollo-backed FFT APIs.
- The Apollo GPU FFT backend now dispatches radix stages with explicit pass boundaries and passes the 128³ kwavers parity check under a hybrid absolute/relative spectral metric.
- The neural beamforming layer adaptation path now uses a parameter-dependent calibration objective instead of constant-offset mutation.
- The distributed neural beamforming processor now partitions frame-major RF volumes across healthy processors with zero-copy frame views and deterministic recomposition.
