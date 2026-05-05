# ADR 005: `solver::forward` Module Domain Grouping

**Status**: 🟡 Proposed
**Date**: 2026-05-05
**Context**: Stream A architectural audit, Sprint cycle following GPU-buffer SSOT closure
**Deciders**: Ryan Clanton

---

## Context

`kwavers/src/solver/forward/mod.rs` declares 20 sibling `pub mod` entries with no
domain grouping:

```rust
pub mod acoustic;        pub mod acoustic_ivp;     pub mod bem;
pub mod bubble_dynamics; pub mod coupled;          pub mod elastic;
pub mod elastic_wave;    pub mod fdtd;             pub mod helmholtz;
pub mod hybrid;          pub mod imex;             pub mod nonlinear;
pub mod ode;             pub mod optical;          pub mod plugin_based;
pub mod poroelastic;     pub mod pstd;             pub mod sem;
pub mod thermal;         pub mod thermal_diffusion;
```

This violates the structural rule "file trees should narrow scope by level, use
domain-relevant names" recorded in the project standards under `structure`. The
intended physics domains (acoustic / elastic / thermal / optical / hybrid / ODE /
plugin) are flattened into a single namespace; readers must pattern-match on
module names rather than navigate by domain.

### Internal-path coupling

A grep of `solver::forward::{fdtd|pstd|acoustic|...}` across the workspace
returns **93 source files** that consume these paths directly (tests, benchmarks,
factory glue, simulation backends, GPU wrappers, clinical workflows). Each such
path is part of the *internal* public API of the crate — Rust treats every
`pub mod` reachable from `lib.rs` as a stable surface for downstream consumers
inside and outside the crate.

A flat-to-nested move (e.g., `solver::forward::fdtd` → `solver::forward::acoustic::fdtd`)
breaks every one of these 93 paths simultaneously.

## Decision

**Defer the topology change pending a deprecation window.** Implement domain
grouping as an *additive* layer rather than a *destructive* move:

1. **Phase 1 (this ADR — minor):** Introduce thin re-export modules at
   `solver::forward::{acoustic, elastic, thermal, optical, hybrid_models, ode_methods, plugin}`
   that *re-export* the existing flat modules without moving any source files.
   Example: `solver::forward::acoustic` becomes a `pub mod acoustic` whose body is
   solely `pub use super::{fdtd, pstd, acoustic_ivp, nonlinear, helmholtz};`.
   This keeps every existing `solver::forward::fdtd::*` path live while exposing
   the domain-grouped path `solver::forward::acoustic::fdtd::*`.
2. **Phase 2 (separate ADR — major):** Once internal callers have migrated
   to the grouped paths (tracked in a follow-up checklist item per call site),
   mark the flat re-exports `#[deprecated]` for one minor release.
3. **Phase 3 (separate ADR — major):** Remove the flat re-exports; `solver::forward`
   exposes only domain-grouped paths and the canonical struct re-exports
   (`FdtdSolver`, `PSTDSolver`, etc.) at the top level.

### Domain mapping

The flat module names `acoustic`, `elastic`, `thermal`, `optical` are already
taken by existing concrete solver modules (e.g., `solver::forward::acoustic`
hosts `AcousticWavePlugin`; `solver::forward::elastic` hosts elastic SWE). To
avoid path collision in Phase 1, group modules use a `_solvers` suffix and
**include the same-named flat module as a group member**:

| Domain group         | Members                                                                  |
|----------------------|--------------------------------------------------------------------------|
| `acoustic_solvers`   | `acoustic`, `acoustic_ivp`, `fdtd`, `pstd`, `nonlinear`, `helmholtz`     |
| `elastic_solvers`    | `elastic`, `elastic_wave`, `poroelastic`, `sem`                          |
| `thermal_solvers`    | `thermal`, `thermal_diffusion`                                           |
| `optical_solvers`    | `optical`                                                                |
| `boundary_element`   | `bem`                                                                    |
| `hybrid_models`      | `hybrid`, `coupled`                                                      |
| `ode_methods`        | `ode`, `imex`                                                            |
| `multiphysics_bubble`| `bubble_dynamics`                                                        |
| `plugin`             | `plugin_based`                                                           |

## Consequences

### Positive

- Domain navigation works immediately for new readers (Phase 1).
- Existing call sites continue to compile unchanged (no breaking churn).
- Migration to grouped paths can proceed call-site-by-call-site under the
  user's `prefer the smallest reversible change` policy.
- `cargo-semver-checks` reports zero regressions for Phase 1.

### Negative

- During Phases 1 and 2, two paths to the same module exist
  (`solver::forward::fdtd` and `solver::forward::acoustic::fdtd`). This is a
  documented temporary SSOT relaxation, with the Phase 3 termination explicit.
- Deprecation lint noise during Phase 2 once `#[deprecated]` is applied.

### Neutral

- No physics or numerics change.
- No public-API surface contraction in Phase 1; surface *expands* additively.

## Out of scope

- Moving source files. All `kwavers/src/solver/forward/<flat>/...` paths remain.
- Modifying any `pub use` re-export at the `solver::forward` top level
  (`FdtdSolver`, `PSTDSolver`, `BemSolver`, etc. continue to be exported there).
- Updating the 93 internal call sites — that work is tracked as a separate
  per-domain checklist after this ADR lands.

## Verification plan

For Phase 1:
- `cargo check --package kwavers` clean.
- All existing tests pass without modification.
- `cargo-semver-checks` reports no breaking changes against the prior tag.
- A new doc test in `solver/forward/mod.rs` exercises both old and new paths
  on the same item to prove they resolve to the same type.

## References

- Project standards `structure` rule (file tree narrows scope by level).
- Project versioning policy: tag `[arch]` items require ADR + deprecation
  period; this ADR splits the work into one `[minor]` and two future `[major]`
  cycles to honor that gate.
- ADR 004 (Domain Material Property SSOT Pattern) — precedent for additive
  consolidation under a deprecation window.
