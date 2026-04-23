# Backlog / Strategy

## Architectural Enhancements
- Restructure into clean Domain/Application/Infrastructure/Presentation bounded contexts.
- Ensure dependency flows are strictly unidirectional (Domain -> App -> Infra/Presentation).
- Review all modules (core, physics, math, domains, simulation, clinical, analysis, solvers).
- BURN crate integration for optimized GPU support.
- Autodiff/PINN implementations for neural network-based physics solving.

## Validation Goals
- Implement automated test scenarios comparing `pykwavers` outputs natively against `k-wave-python` identical scenarios.
- Quantitatively verify sources, signals, grids, sensors, and solvers.
- Closed the vendored `k-wave-python` 2-D FFT line-sensor parity gap in `pykwavers` via native `kspace_line_recon`, the non-square 2-D FFT axis fix, and Python binding export.
- Closed the vendored `k-wave-python` 3-D planar-sensor time-reversal parity gap in `pykwavers` by caching the reconstructed fields and preserving the exact forward pressure/sensor ordering contract.
- Closed the vendored `k-wave-python` 3-D circular piston parity gap in `pykwavers` by using the native `KWaveArray` disc geometry, clipping the PML halo before source-weight comparison against the padded reference mask, and validating the analytical on-axis piston profile.
- Closed the vendored `k-wave-python` 3-D focused bowl parity gap in `pykwavers` by switching the bowl rasterizer to the canonical spiral/BLI source path, reporting the physical-interior source-weight parity, and validating the on-axis waveform comparison.
- Closed the vendored `k-wave-python` 2-D focussed detector parity gap in `pykwavers` by comparing detector-averaged traces for the on-axis and off-axis source cases and validating the directivity-energy ratio.
- Closed the vendored `k-wave-python` 2-D sensor directivity modelling gap in `pykwavers` by comparing the full source-angle trace matrix and the derived directivity curve against the reference example.
- Closed the vendored `k-wave-python` `at_array_as_sensor` gap in `pykwavers` by aligning the arc geometry to the upstream line-sampled BLI footprint, preferring the rebuilt `target/maturin/pykwavers.dll` extension artifact, and validating exact mask parity plus raw/combined detector-matrix comparison.
- Closed the vendored `k-wave-python` `at_array_as_source` gap in `pykwavers` by reusing the canonical arc ordering, comparing exact source-mask and distributed source-signal parity, and validating p_max/p_rms field parity against the rebuilt extension.
- Closed the vendored `k-wave-python` `us_defining_transducer` gap in `pykwavers` by carrying the reference time-step count through the pykwavers scan-line run, aligning the sensor-trace lengths, and validating per-sensor trace metrics with a PASS report.
- Closed the vendored `k-wave-python` `ivp_photoacoustic_waveforms` gap in `pykwavers` by reusing the cached initial-pressure traces, comparing the single-sensor waveform directly, and validating the PASS report metrics.
- Closed the vendored `k-wave-python` `us_bmode_phased_array` gap in `pykwavers` by validating the quick steering-angle sweep against the cached k-Wave and pykwavers scan lines, confirming the fundamental/harmonic B-mode parity, and preserving the existing GPU profile contract.
- Closed the vendored `k-wave-python` `sd_focussed_detector_3D` gap in `pykwavers` by validating per-source trace parity, checking the on-axis/off-axis directivity ratio, and preserving the PASS report contract.
- Keep exact tone-burst regression coverage for the Gaussian default envelope and non-integer sample-count cases.
- Validate the seismic FWI adjoint-state path with receiver-order residual reversal, discrete L2 objective scaling, CFL checks, and finite-difference gradient identities.
- Validate the reconstruction FWI path with sign-correct residuals, `dt`-scaled objectives, checkpointed adjoint replay, timestep validation, and encoded-gradient aggregation.
- Extract and keep the acoustic adjoint-state core as the single source of truth for L2 residuals, objective scaling, time reversal, and signed-correlation accumulation.
- Maintain checkpointed replay regression coverage for reconstruction FWI to preserve exact adjoint-state accumulation with reduced peak memory.
- Validate the acoustic GPU compute path with workgroup sizes that satisfy device invocation limits, matched uniform-buffer layouts, and fused field-update loops that avoid transient gradient volumes.
- Validate the GPU memory-tracking surface through the public `kwavers::profiling` export, direct allocation-guard RAII semantics, and FDTD pressure upload/download roundtrips.
- Remove remaining GPU-adjacent lint noise in beamforming and k-space hot paths by replacing zero-fill readback, eliminating redundant casts, and keeping dispatch/debug metadata on the production path.
- Remove remaining FDTD solver allocation churn by reusing staggered divergence scratch state, eliminating redundant scalar zero-fills, and keeping GPU readback buffers in place.
- Finish the FFT migration by keeping `kwavers` on Apollo-backed transforms only, preserving no direct `rustfft` usage in `kwavers` source, tests, or benches.
- Keep the Apollo GPU FFT backend parity-checked against kwavers examples after the radix-stage dispatch fix and hybrid absolute/relative parity metric.

## Outstanding k-wave-python Parity Gaps
- `at_linear_array_transducer`: requires per-element Euler rotation on `ElementShape::Rect` rasterization; not yet wired in `KWaveArray` or the pykwavers `add_rect_element` binding.
- `at_focused_annular_array_3D`: requires a new `ElementShape::Annulus { inner_d, outer_d, focus }` (or composite `add_annular_array`) with BLI rasterization.
- `at_focused_bowl_AS` and `at_circular_piston_AS`: require an axisymmetric PSTD solver (radial-axial grid + Bessel-based k-space operators).
- `na_controlling_the_pml`: requires PML configuration knobs (`pml_size`, `pml_alpha`, `pml_inside`) on the pykwavers solver binding.
- `checkpointing`: requires a save/resume contract for PSTD session state and re-entry from a prior time step.

## Technical Debt Prevention
- Proactively locate and discard deprecated or duplicate methods, replacing them strictly with unified accessors.
- Remove outdated benchmarking, test data, and logs upon obsolescence.
- Keep the neural beamforming adaptation and distributed execution paths on the canonical SSOT helpers; extend them by refining the shared partition/recomposition logic rather than cloning variant-specific processors.
