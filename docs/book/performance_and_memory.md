# Performance and Memory

## Scope

Performance covers allocation control, scratch reuse, GPU staging, FFT planning, cache locality, SIMD/SWAR strategy, Rayon parallelism, WGPU dispatch, benchmark instrumentation, and memory budgets. Code ownership maps to `kwavers::profiling`, `kwavers::gpu`, `kwavers::solver`, `kwavers::analysis::performance`, and Apollo.

## Theorem: Scratch Reuse Lowers Peak Allocation

If a loop of `T` steps allocates a temporary array of size `N` at each step, then replacing it with one persistent scratch array reduces loop allocation count from `T` arrays to `1` array without changing arithmetic results.

### Proof Sketch

The scratch buffer is overwritten before each use and is not part of the mathematical state after the step. Reusing the same storage preserves the data-dependency graph while eliminating repeated allocation.

## Algorithm: Performance Validation

1. Measure time and memory on real 1-D, 2-D, and 3-D workloads.
2. Attribute time to propagation, FFT, source injection, boundary update, sensor sampling, and transfer stages.
3. Optimize real hot paths: scratch reuse, in-place kernels, tiling, and backend dispatch.
4. Re-run correctness tests and benchmark comparisons after every optimization.

## Implementation Targets

- Keep performance changes value-identical to reference outputs.
- Prefer `*_into` APIs and caller-owned buffers for hot paths.
- Keep GPU readback and staging explicit in benchmark reports.

## Research Anchors

- k-Wave 2-D acceleration/treatment-planning context: https://www.iccs-meeting.org/archive/iccs2025/papers/159080065.pdf
- Apollo transform backend: `apollo/`
