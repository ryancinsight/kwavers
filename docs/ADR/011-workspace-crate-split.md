# ADR 011 — Workspace crate split of the `kwavers` monolith

- **Status:** Implemented (split complete). **Amended 2026-06-03: the re-export facade was removed — see Amendment below.**
- **Date:** 2026-06-01 · **Audited:** 2026-06-03
- **Change class:** [arch] → [major] post-1.0 (target `4.0.0`)
- **Relates:** extends and partially supersedes [ADR 006](006-workspace-pyo3-bindings-architecture.md); promotes the layering of [ADR 003](003-signal-processing-analysis-layer.md), [ADR 007](007-solver-forward-domain-grouping.md), [ADR 008](008-compute-backend-trait-wiring.md) to crate boundaries

## Context

`kwavers` was a single crate of ~461k LOC across ~3,475 files. Rust compiles a
crate as a unit, so any edit recompiled everything — incremental builds ran 5–8
minutes even for one-line changes, the dominant drag on iteration. The module tree
already formed a clean, acyclic, linear layer DAG with zero upward edges:

```
core → math → domain → physics → solver → analysis → simulation → diagnostics/therapy
```

## Decision

Split the monolith into **one crate per layer plus a thin `kwavers` facade**,
extracting leaf-first (highest fan-in first) for the largest build-time win.

The facade re-exports each layer crate under its original module name
(`pub use kwavers_core as core;`, …), so existing `crate::core::…` /
`kwavers::core::…` paths resolve unchanged — avoiding a 3,377-site path rewrite.
This facade is the **intended end-state** (mirrors how `tokio` re-exports its
sub-crates), not a transitional shim, and is therefore exempt from the anti-shim
rule (which targets intra-crate rename aliases kept to dodge call-site updates).

## Current state (audited 2026-06-03)

Complete. All layer crates are extracted, declared as workspace members
(`Cargo.toml`), and re-exported by the facade (`crates/kwavers/src/lib.rs:60-96`):

| Crate | LOC (approx) |
|-------|------|
| `kwavers-core` | 10.5k |
| `kwavers-math` | 14.7k |
| `kwavers-domain` | 68.1k |
| `kwavers-physics` | 78.1k |
| `kwavers-solver` | 151.7k |
| `kwavers-analysis` | 50.4k |
| `kwavers-simulation` | 11.7k |
| `kwavers-diagnostics` | 21.5k |
| `kwavers-therapy` | 37.7k |
| `kwavers-gpu` | wgpu/WGSL backend (leaf above solver) |

**Divergence from the original plan — `clinical` split into two crates.** The
original DAG listed a single `kwavers-clinical`; it was instead realized as
`kwavers-diagnostics` (imaging) + `kwavers-therapy` (therapy/safety/regulatory/
patient_management). The facade reconstructs the original namespace
(`crates/kwavers/src/lib.rs:106-135`):

```rust
pub mod clinical {
    pub use kwavers_diagnostics as imaging;
    pub use kwavers_therapy::{patient_management, regulatory, safety, therapy};
}
```

The originally cross-cutting `gpu/` and `infrastructure/` modules have since been
fully promoted out of the facade:

- **GPU** → `crates/kwavers-gpu` (a leaf above `solver`, implementing the
  `ComputeBackend`/`FdtdGpuAccelerator` trait surfaces of
  [ADR 008](008-compute-backend-trait-wiring.md)). The facade depends on it and
  re-exports `kwavers::gpu` (feature-gated) and `kwavers::profiling`.
- **I/O** → the output writers (`save_pressure_data`, `save_data_csv`,
  `save_light_data`, `generate_summary`) moved to `kwavers_simulation::io`, where
  run output belongs; the facade re-exports them at the crate root.
- **`architecture/`** (a runtime layer-dependency validator) was deleted — crate
  boundaries now enforce the DAG at compile time, making it redundant.

The facade crate `crates/kwavers/src/` is therefore now **purely thin**: `lib.rs`
(re-exports + `init_logging`/`get_version_info`), `main.rs` (a smoke-test binary),
and `tests.rs` (integration tests over the unified surface). No layer code remains
in it.

The PyO3 bindings (`crates/kwavers-python`) depend only on the `kwavers` facade —
the one-directional dependency of [ADR 006](006-workspace-pyo3-bindings-architecture.md)
is preserved.

## Consequences

- A one-line edit now recompiles the changed layer crate + facade instead of the
  whole monolith.
- Public API is unchanged for facade consumers (`kwavers::…`); the new `kwavers-*`
  crates are additive. Run `cargo-semver-checks` on the facade before the `4.0.0`
  tag.
- Extraction surfaced two reusable lessons for any further crate boundary:
  (1) error `From`/`#[from]` impls referencing foreign crates must become optional
  feature-gated deps to keep the foundation a clean leaf; (2) `pub(crate)` items
  shared across modules via `crate::<layer>::…` must be promoted to `pub` once the
  module boundary becomes a crate boundary.
- The `gpu` feature forwards into the layer crates that expose feature-gated GPU
  paths (`kwavers-core/gpu`, `kwavers-math/gpu`, `kwavers-solver/gpu`, …) alongside
  the concrete `kwavers-gpu/gpu` backend.

## Amendment (2026-06-03) — re-export facade removed

The original decision made `kwavers` a thin **re-export facade** (`pub use
kwavers_core as core;` …) so `kwavers::core::…` paths kept resolving, explicitly
to avoid a mass path rewrite. That facade has now been **removed**: depending on a
single re-export crate re-introduced the indirection the split set out to
eliminate, and the only external consumer (`kwavers-python`) gains nothing from it.

New end-state:

- **No re-export surface.** `crates/kwavers/src/lib.rs` no longer re-exports any layer
  crate. Consumers depend on the layer crates directly (`kwavers_core`,
  `kwavers_domain`, `kwavers_solver`, …).
- **`kwavers` is now a thin top-level *app / integration* crate**: it holds the
  binary (`main.rs`), the cross-cutting integration tests / examples / benches
  (which depend on all layer crates), and two app utilities (`init_logging`,
  `get_version_info`). It is a leaf — nothing depends on it.
- **`kwavers-python` depends on the layer crates directly** (not on `kwavers`);
  its `kwavers::<layer>::…` paths were rewritten to `kwavers_<layer>::…` and its
  `gpu`/`plotting`/`pinn`/`full` features now forward to the layer crates.
- The mass path rewrite the original ADR avoided was executed mechanically:
  `kwavers::<layer>` → `kwavers_<layer>`, with the facade's root re-exports
  (`Grid`, `KwaversResult`, …) mapped to their origin crates and grouped
  `use kwavers::{…}` imports split per origin crate.

Trade-off accepted: there is no longer a unified `kwavers::…` Rust API surface;
external Rust consumers (none known outside this repo) import the layer crates.
This supersedes the "deliberate end-state facade" framing in the Decision above
and the facade dependency recorded in
[ADR 006](006-workspace-pyo3-bindings-architecture.md).
