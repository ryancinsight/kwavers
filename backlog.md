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

## Technical Debt Prevention
- Proactively locate and discard deprecated or duplicate methods, replacing them strictly with unified accessors.
- Remove outdated benchmarking, test data, and logs upon obsolescence.
