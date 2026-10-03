<a id="kw-gpu-062"></a>

## KW-GPU-062 — GPU PSTD peak-pressure output [major] — review

priority: feature; needs: none; scope: `crates/kwavers-gpu/src/pstd_gpu/` and its WGSL shader ABI, `crates/kwavers-simulation/src/solver_adapters/gpu_pstd.rs`, `crates/kwavers-math/src/fft/mod.rs`

- **Acceptance:** the provider accumulates `max_t |p|` on the GPU for every voxel, transfers exactly that volume when requested, and never labels a final pressure frame a peak envelope. The output request supports final, peak or both without allocating a peak volume for a sensor-only run. Source, lossless-absorption and heterogeneous-nonlinearity choices match the CPU PSTD contract with no host fallback. The reference FFT runs directly on the shared Leto/Eunomia complex type.
- **Decision:** [ADR-040](docs/adr/040-gpu-pstd-peak-pressure-output.md).
- **Evidence:** the simulation adapter requests the explicit peak output, retains it apart from final fields, shares the direct runner's weighted pressure-source schedule, and rejects unsampled `Source` objects and unsupported velocity-source assembly. Warning-denied all-feature Clippy and the WGPU-featured Nextest lane (259/259, including the heterogeneous CPU/GPU contract and real peak-envelope runs) pass.
- **Closure gate:** a replacement hosted matrix; re-validate against the default branch and close if the peak output path and its tests are present.
- **External requirement:** the private full-wave consumer owns its explicit peak-pressure regression; its inaccessible checkout does not block this repository's delivery.
