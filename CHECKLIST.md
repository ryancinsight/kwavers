# Project Checklist

## Phase 1: Foundation (0-10%)
- [ ] 100% Audit/Planning/Gap Analysis
- [x] Detect Root/VCS & initialize formal artifacts (`checklist.md`, `backlog.md`, `gap_audit.md`)
- [x] Verify `k-wave` and `k-wave-python` present in `external/`
- [x] Check existing modules for circular dependencies and cross-contamination (Solvers, Domains, Simulation, Clinical, Analysis, Physics, Math, Core)

## Phase 2: Execution & Restructuring (10-50%)
- [x] Eliminate circular dependencies & cross-contaminations
- [x] Replace the domain-owned solver factory with the `simulation::solver_factory` assembly boundary and descriptor-only `solver::factory` selection policy
- [x] Add architecture regression coverage for domain-to-solver/simulation imports and solver-factory-to-domain/simulation imports
- [x] Consolidate FDTD test/example struct literals onto `FdtdConfig::default()` for geometry and future defaulted fields
- [x] Add `FwiProcessor::generate_synthetic_data` as the public synthetic-data SSOT over the canonical FWI forward model
- [x] Remove PSTD absorption visibility and unreachable-code diagnostics without changing fractional-Laplacian formulas
- [x] Clean codebase: Remove dead/deprecated code, resolve all warnings, avoid build logs
- [x] Establish deep nested file structure (3-5+ levels) with parent/child hierarchies
- [x] Enforce Single Source of Truth via shared accessors

## Phase 3: Component Validation vs k-Wave (50%+)
- [x] Grids: Implement in `kwavers`, wrap in `pykwavers`, validate vs `k-wave`
- [x] Sources: Implement in `kwavers`, wrap in `pykwavers`, validate vs `k-wave`
- [x] Signals: Implement in `kwavers`, wrap in `pykwavers`, validate vs `k-wave`
- [x] Sensors: Implement in `kwavers`, wrap in `pykwavers`, validate vs `k-wave`
- [x] Solvers: Implement in `kwavers` (using BURN for GPU/Autodiff for PINN), wrap in `pykwavers`, validate vs `k-wave`
- [x] FWI physics: implement acoustic L2 objective, receiver-order adjoint injection, CFL validation, and second-derivative gradient tests
- [x] Reconstruction FWI: implement sign-correct residuals, `dt`-scaled objective, checkpointed replay adjoint accumulation, timestep validation, and encoded-gradient aggregation
- [x] Shared acoustic adjoint-state core: consolidate L2 residuals, objective scaling, time reversal, and signed-correlation accumulation across FWI paths
- [x] GPU acoustic field path: enforce workgroup limits, correct uniform layout, and fuse velocity updates to remove temporary gradient volumes
- [x] GPU allocation tracking: publish `kwavers::profiling`, validate guard-based RAII budgets, and verify FDTD pressure roundtrip with the current constructors
- [x] Beamforming/k-space cleanup: remove zero-fill readback, fix `device.poll` handling, and eliminate redundant spectrum-shape casts
- [x] FDTD scratch reuse: keep staggered divergence in solver-owned scratch state, drop redundant scalar zero-fills, and preserve in-place GPU overwrite semantics
- [x] k-wave-python example parity: complete the 2-D FFT line-sensor comparison and finish any missing CPML handling required for lower-dimensional embeddings
- [x] k-wave-python example parity: complete the 3-D planar-sensor time-reversal comparison, including cached reconstructed fields for the slow k-Wave reconstruction path
- [x] k-wave-python example parity: complete the 3-D circular piston comparison with exact weighted source masks and analytical steady-state validation
- [x] k-wave-python example parity: complete the 3-D focused bowl comparison with physical-interior source-weight parity and on-axis waveform validation
- [x] k-wave-python example parity: complete the 2-D focussed detector comparison with detector-averaged trace parity and directivity-energy validation
- [x] k-wave-python example parity: complete the 2-D sensor directivity modelling comparison with source-angle trace parity and directivity curve validation
- [x] k-wave-python example parity: complete the `at_array_as_sensor` comparison with exact arc mask parity, raw detector-matrix comparison, and combined arc-trace validation
- [x] k-wave-python example parity: complete the `at_array_as_source` comparison with exact source-mask parity, distributed source-signal parity, and p_max/p_rms field validation
- [x] pressure-source ordering contract: emit Fortran-order active-cell rows for arc and linear-array pressure builders and pin exact helper-matrix parity against k-wave-python
- [x] k-wave-python example parity: complete the `us_defining_transducer` comparison with exact time-step alignment, per-sensor trace parity, and report-backed PASS validation
- [x] k-wave-python example parity: complete the `ivp_photoacoustic_waveforms` comparison with cached initial-pressure trace parity and PASS report validation
- [x] k-wave-python example parity: complete the `us_bmode_phased_array` comparison with quick-sweep scan-line parity, B-mode image parity, and PASS report validation
- [x] k-wave-python example parity: complete the `sd_focussed_detector_3D` comparison with per-source trace parity, directivity ratio validation, and PASS report validation
- [x] k-wave-python example parity: complete the `us_bmode_linear_transducer` comparison with GPU source-mode alignment, quick raw scan-line parity, and PASS report validation
- [x] GPU PSTD sweep optimization: split per-line medium upload timing, compact GPU profile aggregation, and advance the lateral medium window in place before the 16-line sweep
- [x] FFT migration: remove direct `rustfft` usage from `kwavers` source, tests, and benches; route all transform calls through Apollo-backed FFT APIs
- [x] Apollo GPU FFT parity: validate the 128³ GPU FFT case after correcting radix-stage dispatch and using a hybrid absolute/relative error metric
- [x] Neural layer adaptation: replace constant-offset mutation with the exact scalar calibration step on the layer parameters
- [x] Distributed neural beamforming: chunk frame-major RF volumes across healthy processors, reuse frame views, and recombine deterministically
- [x] k-wave-python example parity: `at_linear_array_transducer` — parity closed with additive pressure-source mode, Fortran-order source rows, rebuilt extension cache v5, and validated `p_max` field parity
- [x] k-wave-python example parity: `at_focused_annular_array_3D` — add `ElementShape::Annulus` with BLI rasterization, expose in pykwavers, validate concentric-ring focused field (pearson=0.907 PASS; per-element drive pearson=0.907 PASS)
- [x] k-wave-python example parity: `at_circular_piston_AS` — axisymmetric PSTD parity PASS with Fortran-order sensor reshaping, Pierce analytical validation, and cached figure/report artifacts
- [x] k-wave-python example parity: `at_focused_bowl_AS` — axisymmetric PSTD parity PASS with Fortran-order sensor reshaping, O'Neil analytical comparison, and cached figure/report artifacts
- [x] k-wave-python example parity: `na_controlling_the_pml` — waveform parity PASS across the PML attenuation sweep and exact k-Wave-style save-to-disk HDF5 input-file parity PASS via `na_controlling_the_pml_compare.py`
- [x] PSTD checkpointing: exact save/resume contract with binary KWCP state, file deletion after restore, bit-exact continuation, and PASS report validation via `checkpointing_compare.py`
- [x] Compare script standardization: normalize `parity_status:` key across all 22 example compare scripts; add NPZ caching to `us_bmode_phased_array_tiny_compare.py`; fix `sys.exit` → `raise SystemExit`
- [x] Workspace clippy hygiene: eliminate all warnings across `kwavers`, `pykwavers`, `apollo-fft`, `apollo-fft-wgpu`, `ritk-core`, `ritk-model`, `ritk-registration`; replace approximate π/√2 literals with `std::f{32,64}::consts`
- [x] k-wave-python example parity: `tvsp_homogeneous_medium_monopole` — 2-D time-varying pressure source parity PASS (pearson=0.9996, rms_ratio=1.027) via `tvsp_homogeneous_medium_monopole_compare.py`; pre-filtered signal shared across both engines; NPZ caching
- [x] k-wave-python example parity: `ivp_homogeneous_medium` — 2-D initial-pressure (two discs) with 50-pt Cartesian sensor parity PASS (pearson=0.9977, psnr=38.7 dB) via `ivp_homogeneous_medium_compare.py`; C→Fortran sensor-row permutation to align k-wave and pykwavers traversal orders
