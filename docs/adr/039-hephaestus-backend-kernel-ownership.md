# ADR 039: Move backend kernel ownership to Hephaestus

- **Status:** Accepted
- **Date:** 2026-07-17
- **Change class:** [major]

## Context

`kwavers-gpu::backend` owned a second WGPU buffer pool, host transfer path,
pipeline cache, and bind-group dispatcher for elementwise multiplication and
spatial derivatives.  It duplicated the typed allocation, transfer, pooling,
and WGSL multi-storage dispatch that `hephaestus-wgpu` already owns.  The local
manager also retained a non-owning raw device pointer and an unsafe `Send`
implementation.

Leto is not a GPU provider.  It remains the dense host-array representation at
the Kwavers solver interface.  Hephaestus owns the device lifetime, typed
buffers, transfer, synchronization, and kernel dispatch below that boundary.

## Decision

Delete the public backend buffer-manager and pipeline-manager modules.  Make
the WGPU provider upload dense `leto::Array3<f32>` slices into
`hephaestus_wgpu::WgpuBuffer<f32>`, dispatch multiplication through
Hephaestus' `binary_elementwise_into`, and dispatch the spatial derivative
through `WgslMultiStorageKernel`.  The derivative WGSL source is reduced to
its one live operation and its binding declaration is owned by Hephaestus.

No compatibility re-export remains.  `kwavers-gpu` advances from 3.0.0 to
4.0.0 because the obsolete public manager types and modules are removed.

## Consequences

The WGPU implementation remains the only concrete WGSL kernel provider, while
CUDA remains limited to the real Hephaestus elementwise operation it implements.
Both implementations continue to use Leto only at host-array boundaries.

## Verification

- Compile-time: provider operations bind to `ComputeDevice`,
  `MultiStorageDevice`, and `MultiStorageKernel`.
- Value semantics: exact multiply and affine-derivative regressions execute
  through the WGPU provider on the available adapter: 45/45 backend tests pass.
- CUDA-provider type and operation-family verification: 50/50 backend tests
  pass without fabricating a CUDA spatial derivative.
- `cargo clippy --offline -p kwavers-gpu --all-targets --features gpu -- -D
  warnings` and the corresponding `cuda-provider` invocation pass.
- `cargo doc --offline --no-deps -p kwavers-gpu` and
  `cargo test --offline -p kwavers-gpu --doc` pass.
- Structural audit: no `WgpuBackendBufferManager` or `WgpuPipelineManager`
  source reference remains after migration.
