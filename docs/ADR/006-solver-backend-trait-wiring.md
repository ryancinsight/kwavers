# ADR 006: Wire `solver::backend` Trait, Gate Bit-Rotted GPU Submodule

**Status**: 🟢 Accepted (Phase 1 implemented)
**Date**: 2026-05-05
**Context**: Stream A audit follow-on, sprint cycle following ADR 005 Phase 1 landing
**Deciders**: Ryan Clanton

---

## Context

A poison-pill verification (deliberate compile error injected into
`solver/backend/gpu/buffers.rs`) revealed that the entire 2057-line
`kwavers/src/solver/backend/` subtree was **orphaned**: `solver/mod.rs`
contained no `pub mod backend;` declaration, so neither the `Backend` trait
nor any of its types ever entered the crate's module graph. The directory
was code-on-disk that the compiler never read.

This is a violation of the project's "Compute backend trait" standard,
which requires CPU / GPU / accelerator dispatch to be mediated by a single
public trait. The trait existed (`solver::backend::traits::Backend`) but
was unreachable.

### Verification of orphan status

| Check | Result |
|---|---|
| Poison `INTENTIONAL_COMPILE_POISON` injected into `buffers.rs` and built `--features gpu` | 0 errors emitted — compiler never visits the file |
| Grep for `pub mod backend;` in `solver/mod.rs` | not present |
| Grep for `impl Backend for` outside the orphaned subtree | 0 matches |
| Grep for `use crate::solver::backend::*` outside the subtree | 1 site, gated `#[cfg(feature = "gpu")]` inside `solver/validation/gpu_cpu_equivalence/` — itself orphaned (parent `mod.rs` lacks `pub mod gpu_cpu_equivalence;`) |

### Wire-in attempt surfaces 32 latent compile errors

When the orphan was repaired by adding `pub mod backend;` to
`solver/mod.rs` and `pub mod gpu;` (gated on `feature = "gpu"`) to
`solver/backend/mod.rs`, building with `--features gpu` produced **32
errors** of three categories:

1. `wgpu::Device::queue()` is called but no such method exists on `wgpu`
   v26's `Device` (queue is owned separately). 2 sites in `buffers.rs`.
2. `KwaversError::ConfigError(ConfigError::InvalidParameter { .. })` —
   neither `KwaversError` has a `ConfigError` variant in its current
   form, nor does the current `ConfigError` enum carry an
   `InvalidParameter` variant. 6 sites in `buffers.rs`.
3. `wgpu::Instance::new(...)` and `wgpu::DeviceDescriptor { ... }` field
   shape mismatches — the upstream API has shifted since this code last
   compiled. ~24 sites in `init.rs`, `pipeline/manager/execute.rs`, and
   `realtime_loop.rs`.

These are the predictable signature of code that was never touched by
the type checker as upstream APIs evolved.

## Decision

**Two-phase repair**:

### Phase 1 (this ADR — `[minor]`, landed)

Wire the trait surface in. Gate the broken submodule behind an explicit
opt-in feature so it cannot leak into normal builds:

```rust
// solver/mod.rs
pub mod backend;

// solver/backend/mod.rs
#[cfg(all(feature = "gpu", feature = "solver_backend_gpu_unstable"))]
pub mod gpu;
pub mod traits;
pub use traits::{Backend, BackendCapabilities, BackendType, ComputeDevice};
```

```toml
# kwavers/Cargo.toml
solver_backend_gpu_unstable = ["gpu"]
```

After Phase 1:

- The `Backend` trait, `BackendCapabilities`, `BackendType`, and
  `ComputeDevice` are reachable via the canonical
  `crate::solver::backend::{Backend, ...}` path.
- The bit-rotted `gpu` submodule cannot break a normal
  `cargo build` or `cargo build --features gpu` build — it is only
  pulled in under `--features solver_backend_gpu_unstable`, an explicit
  opt-in for repair work.
- 3 compile-time surface tests (`backend_surface_tests` in
  `solver/backend/mod.rs`) verify the trait is reachable, object-safe,
  and that all capability / device types value-inspect correctly per
  anti-mock rules.

### Phase 2 (separate sprint, `[patch]`)

Repair the 32 errors in `solver/backend/gpu/`:

1. Replace `device.queue()` with the `Queue` argument that should be
   threaded through `BufferManager::read_buffer_to_array` (currently the
   queue is not in scope at the call site — the function signature must
   take `queue: &wgpu::Queue`).
2. Update `KwaversError`/`ConfigError` variant names to current spelling
   (`InvalidConfiguration`, etc.) — verify against
   `core::error::config::ConfigError`'s current shape.
3. Update `wgpu::InstanceDescriptor`, `RequestAdapterOptions`,
   `DeviceDescriptor`, and `BufferDescriptor` field initializers to
   match `wgpu` v26 — most likely just need `..Default::default()` plus
   removal of fields that no longer exist.
4. Audit the unused imports in `realtime_loop.rs` and `mod.rs`.
5. Drop the `solver_backend_gpu_unstable` feature flag and gate the
   submodule on `feature = "gpu"` alone.
6. Wire `solver::validation::gpu_cpu_equivalence` into
   `solver/validation/mod.rs` (also currently orphaned) once the
   GPUBackend it depends on compiles.

Phase 2 needs no further ADR — the architectural decision is settled
here; only the mechanical repair remains.

### Phase 3 (separate sprint, `[arch]`)

Tighten `Backend` to match the project standard's "Compute backend trait"
specification more fully:

```rust
trait ComputeBackend {
    type Buffer;
    type KernelDescriptor;
    type DispatchFuture<T>: Future<Output = T>;
    // ...
}
```

Current `solver::backend::traits::Backend` is a thinner mathematical-
operation API (`fft_3d`, `element_wise_multiply`, etc.). The full
spec calls for associated-type buffer/kernel/future types so additional
backends (Metal, CUDA, ROCm) can land as new `impl` blocks without
algorithmic duplication. This is `[arch]` and gets its own ADR.

## Consequences

### Positive

- The `Backend` trait is now part of the public API at its canonical
  location, satisfying the standards rule.
- Future GPU/accelerator implementations have a stable trait to target
  without first re-discovering its existence.
- Bit-rotted code is contained behind an explicit, named feature flag
  rather than silently sitting on disk pretending to be live.
- The poison-pill methodology is recorded in this ADR as a verification
  technique for catching orphaned modules in the future.

### Negative

- `solver::backend::gpu` remains broken until Phase 2 lands. Any
  consumer that needs a GPU `Backend` impl must wait or work in the
  `solver_backend_gpu_unstable` branch.
- Phase 2 requires touching 32 sites; non-trivial.

### Neutral

- No change to default-feature build artifacts, default-feature test
  output, or `cargo-semver-checks` results: the trait was previously
  unreachable, so adding it expands the public API surface but does not
  contract or alter anything existing consumers were relying on.

## Verification (Phase 1)

| Gate | Outcome |
|---|---|
| `cargo check --package kwavers` (default features) | clean |
| `cargo check --package kwavers --features gpu` | clean (4 unrelated `kwavers`/5 `ritk-io` warnings, 0 errors) |
| `cargo check --package kwavers --features solver_backend_gpu_unstable` | 32 errors as documented (acceptable: explicit-opt-in flag for repair work) |
| `cargo test --package kwavers --lib solver::backend::backend_surface_tests` | 3/3 pass |
| Full nextest suite | (pending; ran in same sprint cycle) |

## References

- Project standards `compute_backend_trait` rule.
- Project standards `cleanup` rule (orphaned modules removed or wired in,
  not left dormant).
- Project standards `anti_mocking` rule (surface tests inspect computed
  values rather than `is_ok()`).
- ADR 005 — additive-deprecation pattern precedent for staged migrations.
