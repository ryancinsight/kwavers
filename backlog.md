# Backlog / Strategy

## Architectural Enhancements
- Restructure into clean Domain/Application/Infrastructure/Presentation bounded contexts.
- Ensure dependency flows are strictly unidirectional (Domain -> App -> Infra/Presentation).
- Keep concrete solver assembly in `simulation::solver_factory`, keep `solver::factory` limited to descriptor-based selection policy, and reject domain-layer imports of solver or simulation modules.
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
- Closed the pressure-source ordering contract in `pykwavers` by switching the arc and linear-array source builders to Fortran-order active-cell rows and pinning the exact helper-matrix parity against k-wave-python; the rebuilt `at_linear_array_transducer` example now passes the `p_max` field comparison.
- Closed the vendored `k-wave-python` `us_defining_transducer` gap in `pykwavers` by carrying the reference time-step count through the pykwavers scan-line run, aligning the sensor-trace lengths, and validating per-sensor trace metrics with a PASS report.
- Closed the vendored `k-wave-python` `ivp_photoacoustic_waveforms` gap in `pykwavers` by reusing the cached initial-pressure traces, comparing the single-sensor waveform directly, and validating the PASS report metrics.
- Closed the vendored `k-wave-python` `us_bmode_phased_array` gap in `pykwavers` by validating the quick steering-angle sweep against the cached k-Wave and pykwavers scan lines, confirming the fundamental/harmonic B-mode parity, and preserving the existing GPU profile contract.
- Closed the vendored `k-wave-python` `sd_focussed_detector_3D` gap in `pykwavers` by validating per-source trace parity, checking the on-axis/off-axis directivity ratio, and preserving the PASS report contract.
- Closed the vendored `k-wave-python` `us_bmode_linear_transducer` gap in `pykwavers` by disabling GPU source-kappa correction to match the upstream `NotATransducer.u_mode = "additive-no-correction"` contract, reusing full medium buffers with borrowed `PyReadonlyArray3` uploads, and pinning the PASS report with a cached regression test.
- Reduced the pre-sweep `us_bmode_linear_transducer` GPU hot path by restoring the measured medium-upload timing after cached execution, aggregating per-line GPU timings through a compact tuple summary, and advancing the lateral medium window in place to avoid rebuilding the active slab on every scan line.
- Closed the PSTD checkpointing contract by validating bit-exact save/resume continuation, exact checkpoint file deletion after restore, and the PASS report emitted by `checkpointing_compare.py`.
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
- `at_linear_array_transducer`: closed after switching the parity example to the upstream additive pressure-source mode; the source rows remain Fortran-ordered and the rebuilt extension now matches `p_max` parity.
- `at_focused_annular_array_3D`: requires a new `ElementShape::Annulus { inner_d, outer_d, focus }` (or composite `add_annular_array`) with BLI rasterization.
- `at_focused_bowl_AS` and `at_circular_piston_AS`: closed after fixing the pykwavers sensor reshape to Fortran order, which restored PASS parity on both cached axisymmetric PSTD example comparisons.
- `na_controlling_the_pml`: closed by validating waveform parity across the PML attenuation sweep and exact k-Wave-style save-to-disk HDF5 input-file parity via versioned artifacts in `pykwavers/examples/output/na_controlling_the_pml/hdf5_v1/`.
- `checkpointing`: closed by validating bit-exact save/resume continuation, exact checkpoint file deletion after restore, and the PASS report emitted by `checkpointing_compare.py`.

## Technical Debt Prevention
- Proactively locate and discard deprecated or duplicate methods, replacing them strictly with unified accessors.
- Prefer `..Default::default()` for `FdtdConfig` test/example literals so new defaulted fields remain single-sourced by `FdtdConfig::default()`.
- Keep FWI example and caller synthetic data generation routed through `FwiProcessor::generate_synthetic_data`, the public wrapper over the canonical forward model.
- Remove outdated benchmarking, test data, and logs upon obsolescence.
- Keep the neural beamforming adaptation and distributed execution paths on the canonical SSOT helpers; extend them by refining the shared partition/recomposition logic rather than cloning variant-specific processors.
- Closed the sonoluminescence bremsstrahlung oversized-file gap by splitting constants, Gaunt factors, noble-gas data, plasma state, emission model, field assembly, and value-semantic tests into nested vertical modules below 200 lines each.
- Closed the acoustic conservation oversized-file gap by splitting metrics, energy, mass-continuity, momentum, entropy, intensity, heat-source, validation, and value-semantic tests into nested vertical modules below 150 lines each.
- Closed the sonogenetics channel oversized-file gap by splitting constants, gating parameters, channel identity, open-probability equations, ion-current computation, and value-semantic tests into a nested vertical module tree with unchanged public facade exports.
- Closed the quantum-optics orphan/oversized-file gap by wiring `physics::optics::quantum_optics` into the optics module tree, splitting Einstein coefficients, Gaunt factors, special functions, correction assessment, constants, and tests into nested files, and replacing constant invalid-domain fallbacks with non-finite outputs plus tests.
- Closed the skull aberration oversized-file gap by splitting phase-screen constants, model construction, volumetric phase integration, element correction extraction, aperture maps, and value-semantic tests into nested files below 200 lines each; mismatched element coordinate arrays now return a dimension error instead of panicking inside a `KwaversResult` API.

## AS PSTD FFT Hot-Path Optimization
- Closed the axisymmetric PSTD parity gap: `at_circular_piston_AS` (pearson=0.9907, PASS) and `at_focused_bowl_AS` (pearson=1.0000, PASS) after five physics fixes: one-sided radial PML (`pnz = nz + p`, `pz_embed = 0`), CPML inner-z transparency (`radial_inner_z_transparent`), source injection skip for `rhoy`, `density_scale = 1.0` for AS, and correct embed offset.
- Identified AS PSTD FFT allocation churn: `axisymmetric.rs` calls `fft_2d_array`/`ifft_2d_array` (allocating ~6–8 `Array2<Complex64>` per time step on the `(2·nx)×(4·nz)` WSWA domain) instead of the `forward_into`/`inverse_into` pre-allocated path already used by the 3D PSTD propagators. Pre-allocating expanded buffers (`a_exp`, `uz_exp`, `uz_on_r_exp`) and FFT output buffers in `AsContext`, then routing through `plan.forward_into`/`plan.inverse_into`, eliminates all per-step allocation churn on the AS hot path.

## Sonogenetics Research Modernization
- Closed the bacterial-channel coverage gap in `physics::acoustics::therapy::sonogenetics` by adding `MscLG22N` and `MscS` to the existing `MechanoChannel` abstraction, updating theorem/proof documentation for two-state gating, and preserving one canonical `compute_p_open` dispatch path.
- Closed the sonogenetics channel organization gap by moving the two-state gating theorem, pressure-threshold theorem, channel identity table, canonical parameters, and ion-current theorem into domain-scoped nested files while preserving the single canonical dispatch path.
- Corrected `ion_current` to return injected depolarizing current `g·n·P_open·(E_rev − V_m)`, matching the LIF equation contract while documenting the distinction from electrophysiology outward-current sign.
- Residual performance follow-up: `cargo test -p kwavers --lib` passes but reports `solver::forward::nonlinear::kzk::solver::tests::test_conservation_diagnostics_disable` and `solver::validation::numerical_accuracy::pstd::tests::test_pstd_phase_velocity_accuracy` as running beyond 60 seconds; optimize the real KZK/PSTD paths before treating this as closed performance debt.

## Thermal Property Law Modernization
- Closed the thermal absorption placeholder gap in `physics::thermal::properties`: the previous `1 - 0.02 ΔT` law could become negative during ablation heating. The replacement is a positive exponential soft-tissue law using the same `0.015 1/°C` coefficient as the bioheat absorption model.
- Aligned `sound_speed_vs_temperature` with the generic soft-tissue coefficient `dc/(c dT)=1.6e-3` used in the temperature-dependent medium model and documented the local hyperthermia validity boundary from ultrasound thermometry literature.
- Residual modeling scope: generic scalar functions still use soft-tissue coefficients. Tissue-specific thermal updates should route through table-backed material records or explicit coefficient structs before adding organ-specific behavior.

## Plasmonic Effective-Medium Modernization
- Closed the plasmonic mixture-law placeholder gap in `physics::electromagnetic::plasmonics`: `CouplingModel::None` now evaluates the Maxwell-Garnett dilute-sphere closed form instead of a linear dielectric blend.
- Replaced the linearized `CouplingModel::QuasiStatic` branch with the physical closed-form root of the symmetric Bruggeman equation, preserving the existing coupling-model API while making endpoint and residual identities testable.
- Closed the `MieTheory::gold_in_water` simplified Drude-Lorentz placeholder by routing the gold dielectric closure through Johnson-Christy measured optical constants, affine interpolation over the tabulated wavelength domain, and exact `ε=(n+ik)²` conversion.
- Closed the electromagnetic plasmonic-trait physics/organization gap by moving spheroid depolarization formulas into the nested `traits/plasmonic/geometry.rs` kernel and replacing the prior negative-permittivity resonance expression with a Fröhlich/Drude resonance law plus finite damping.
- Residual modeling scope: additional gold datasets such as Rakic, Olmon, or temperature-dependent Magnozzi/Yakubovsky records should be introduced through an explicit dielectric-data strategy before adding film, size-corrected, or thermal gold models.
