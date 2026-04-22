# Project Checklist

## Phase 1: Foundation (0-10%)
- [ ] 100% Audit/Planning/Gap Analysis
- [x] Detect Root/VCS & initialize formal artifacts (`checklist.md`, `backlog.md`, `gap_audit.md`)
- [x] Verify `k-wave` and `k-wave-python` present in `external/`
- [x] Check existing modules for circular dependencies and cross-contamination (Solvers, Domains, Simulation, Clinical, Analysis, Physics, Math, Core)

## Phase 2: Execution & Restructuring (10-50%)
- [x] Eliminate circular dependencies & cross-contaminations
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
- [x] FFT migration: remove direct `rustfft` usage from `kwavers` source, tests, and benches; route all transform calls through Apollo-backed FFT APIs
- [x] Apollo GPU FFT parity: validate the 128³ GPU FFT case after correcting radix-stage dispatch and using a hybrid absolute/relative error metric
