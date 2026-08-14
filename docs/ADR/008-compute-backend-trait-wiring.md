# ADR 008 — Wire the `ComputeBackend` trait; gate the GPU submodule

- **Status:** Implemented (Phases 1–3)
- **Date:** 2026-05-05 · **Audited:** 2026-06-03
- **Change class:** [arch]
- **Relates:** paths relocated by the crate split ([ADR 011](011-workspace-crate-split.md))

## Context

A poison-pill check (a deliberate compile error injected into
`solver/backend/gpu/buffers.rs`) revealed the entire ~2,057-line
`solver/backend/` subtree was **orphaned**: `solver/mod.rs` had no
`pub mod backend;`, so the backend trait and its types never entered the module
graph. This violated the project's compute-backend-trait standard (CPU/GPU/
accelerator dispatch must be mediated by one public trait); the trait existed but
was unreachable. The GPU submodule had also bit-rotted against the current wgpu.

## Decision

- **Phase 1** — add `pub mod backend;`; gate the broken GPU submodule behind an
  opt-in `solver_backend_gpu_unstable` feature; add value-inspecting surface tests.
- **Phase 2** — repair the GPU backend against current wgpu and wire CPU/GPU
  equivalence validation.
- **Phase 3** — widen the trait to an associated-type design (rename `Backend` →
  `ComputeBackend`) with a thin mathematical-operation surface.

## Current state (audited 2026-06-03)

Implemented through Phase 3, in `crates/kwavers-solver/src/`:

- **Phase 1.** `lib.rs:5` declares `pub mod backend;`. `backend/mod.rs:23-24` gates
  `gpu` on `all(feature="gpu", feature="solver_backend_gpu_unstable")`; the feature
  is in `Cargo.toml:42`. Surface tests (`backend/mod.rs:31-105`) inspect values.
- **Phase 3.** The trait is renamed: `pub trait ComputeBackend` (`backend/traits.rs:65`),
  re-exported as `pub use traits::{BackendCapabilities, BackendType, ComputeBackend, ComputeDevice};`
  (`backend/mod.rs:29`). The old `Backend` identifier no longer exists.
- **Phase 2 (substantially done).** A real `impl ComputeBackend for GPUBackend`
  exists (`backend/gpu/mod.rs:166`), and `validation/gpu_cpu_equivalence` is wired
  (`validation/mod.rs:3`, re-exported `:30-31`).

Stale residue to clean: `lib.rs:62-63` still carries a "backend … not yet
implemented" comment.

## Consequences

- The backend trait is reachable and has a concrete GPU implementor; CPU/GPU
  equivalence validation is in the graph.
- The `Backend` → `ComputeBackend` rename means any text referring to `Backend`
  is stale; the canonical name is `ComputeBackend`.
- Open: confirm whether the `solver_backend_gpu_unstable` gate is still warranted
  now that a GPU impl compiles, and remove the stale `lib.rs` comment.
- Original ADR cited `kwavers/src/solver/backend/…`; now `crates/kwavers-solver/src/backend/…`
  ([ADR 011](011-workspace-crate-split.md)).
