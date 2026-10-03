<a id="kw-gpu-060"></a>

## KW-GPU-060 — Hephaestus backend-kernel ownership [major] — review

priority: architecture; needs: none; scope: `crates/kwavers-gpu/src/backend/`, `crates/kwavers-gpu/src/pstd_gpu/`, ADR 039

- **Outcome:** concrete WGPU buffer allocation, pipeline execution and shader dispatch sit behind a Hephaestus-owned provider trait, so WGPU and CUDA implement one operation contract with no algorithm call-site branches. `WgpuComputeProvider` uses Hephaestus typed transfer plus `binary_elementwise_into` and `WgslMultiStorageKernel`; the local buffer/pipeline managers and their unsafe device-pointer ownership are deleted; Leto stays only at the host-array boundary.
- **Decision:** [ADR 039](docs/adr/039-hephaestus-backend-kernel-ownership.md).
- **Acceptance:** `wgpu::Buffer`, `wgpu::ComputePipeline` and `GpuProviderContext<WgpuDevice>` signatures are confined to the WGPU provider implementation; CUDA compute is exposed only for operations with real CUDA kernels and value-semantic differential tests; WGPU value regressions keep exact multiplication and the affine spatial derivatives.
- **Evidence:** offline GPU and CUDA-provider compilation; warning-denied Clippy for both feature sets; GPU backend Nextest 45/45 and CUDA-provider backend Nextest 50/50 (the WGPU cases run on a real adapter). Thermal-acoustic, FDTD pressure, PSTD state/run/medium-update, multi-GPU context and acquisition paths are provider-generic over `GpuDeviceProvider`; WGPU is the only real implementation.
- **Residuals (from the removed narrative, re-validate at next touch):** a real CUDA spatial-derivative kernel and a WGPU-vs-CUDA differential suite do not exist, so CUDA stays outside `GpuComputeProvider`; Apollo has no CUDA FFT provider (upstream, with Hephaestus); Hermes needs a public ternary accumulation slice facade; PINN GPU training through Coeus on Hephaestus provider traits; `kwavers-solver` direct Rayon/ndarray-parallel holdouts and a top-level dev `tokio`; the `async-runtime,gpu` stream test target needs `kwavers_analysis::visualization::stream` restored or the stale test deleted; broad `kwavers-solver --features gpu --all-targets` clippy was blocked by test-target lint debt.
- **Next step:** re-validate each residual against the default branch and split the live ones into DoR items.
