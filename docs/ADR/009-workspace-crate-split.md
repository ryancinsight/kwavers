# ADR 009 — Workspace crate split of the `kwavers` monolith

- **Status:** Accepted — in progress. `kwavers-core` extracted & verified
  (2026-06-01); remaining crates pending.
- **Date:** 2026-06-01
- **Change class:** [arch] → implies [major] post-1.0 (target `4.0.0`)
- **Supersedes/relates:** layering established in ADR 003/005/006; this ADR
  promotes the existing in-crate module layers to workspace crates.

## Context

`kwavers` is a single crate of **461,571 LOC across 3,475 files**. Rust's unit of
compilation is the crate, so any edit recompiles the whole crate; observed
incremental builds are **5–8 minutes** even for one-line changes. This is the
dominant drag on iteration speed.

The module tree already forms clean architectural layers. After the dependency
cleanup recorded in `backlog.md` (2026-06-01), the **full module DAG is acyclic
and linear**, verified by `grep` of cross-layer `use` edges (non-test):

```
core → math → domain → physics → solver → analysis → simulation → clinical
```

Every layer has **0 upward library edges**. `pykwavers` (PyO3) and `gaia`,
`apollo`, `ritk` are already separate workspace crates.

## Decision

Split the `kwavers` crate into one crate per layer, plus a thin `kwavers` facade.
Extract **leaf-first** (most-depended-on layers first) for the largest
incremental-build win.

### Target crate DAG

| Crate | From module | Depends on (internal) |
|-------|-------------|-----------------------|
| `kwavers-core` | `core/` | — (external only) |
| `kwavers-math` | `math/` | core |
| `kwavers-domain` | `domain/` | core, math |
| `kwavers-physics` | `physics/` | core, math, domain |
| `kwavers-solver` | `solver/` | core, math, domain, physics |
| `kwavers-analysis` | `analysis/` | core, math, domain, solver |
| `kwavers-simulation` | `simulation/` | …, solver, physics |
| `kwavers-clinical` | `clinical/` | all of the above |
| `kwavers` (facade) | `lib.rs`, `gpu/`, `infrastructure/` | re-exports all |
| `pykwavers` | (existing) | `kwavers` facade |

`gpu/` and `infrastructure/` are cross-cutting; they stay in the facade crate
initially and may be promoted later (`kwavers-gpu` behind `ComputeBackend`).

### Extraction strategy — facade re-export (NOT a path rewrite)

`crate::core::…` appears **3,377 times** crate-wide (the error type
`KwaversError`/`KwaversResult` and the constants are used everywhere). Rewriting
all 3,377 to `kwavers_core::…` is high-churn, high-risk, and offers no benefit.

Instead, the `kwavers` facade crate re-exports each extracted crate under its
original module name:

```rust
// kwavers/src/lib.rs
pub use kwavers_core as core;     // crate::core::X still resolves, unchanged
pub use kwavers_math as math;
// …
```

This is the **deliberate end-state facade**, not a transitional shim: `kwavers`
is by design the unified public surface (mirrors how `tokio` re-exports its
sub-crates). It is therefore **exempt from the SSOT anti-shim rule** in
`CLAUDE.md` (which targets intra-crate rename aliases kept to avoid updating
callers). The facade is recorded here as the intended architecture, not an
avoidance of work.

Per-crate edits at extraction are limited to:
1. self-referential paths *inside* the moved layer (`crate::<layer>::X` →
   `crate::X`) — e.g. core has **60** such refs (mechanical `sed`);
2. the new crate's `Cargo.toml`;
3. one facade `pub use` line + one workspace-member entry.

`kwavers-core` external deps (from `use` audit): `libc, log, ndarray, rayon,
serde, thiserror` (verify `num-traits` at extraction). Workspace-level dep
versions are inherited via `[workspace.dependencies]`.

## Extraction order (each a verifiable commit; build must stay green)

1. `kwavers-core` (zero internal deps; biggest fan-in → biggest build-time ROI)
2. `kwavers-math`
3. `kwavers-domain`
4. `kwavers-physics`, then `kwavers-solver`
5. `kwavers-analysis`, `kwavers-simulation`, `kwavers-clinical`

Stop-the-line gate after each: `cargo build` + the layer's tests green before the
next extraction.

## Versioning

[arch] post-1.0 ⇒ **`4.0.0`**. The public API is re-exported unchanged through the
facade, so `cargo-semver-checks` should report no surface regression for facade
consumers; the new `kwavers-*` crates are additive. The CLD-13 `PressureFieldSeries`
breaking change (already in Unreleased) also lands in 4.0.0. Migration note: external
users keep importing from `kwavers::…`; only direct-source consumers (none known
outside this repo) would see new crate names.

## Risks & mitigations

- **Hidden cycle exposed at crate boundary** (module cycle legal in one crate,
  illegal across crates). *Mitigation:* the DAG check is already 0-upward; extract
  leaf-first so each step only depends on already-extracted crates.
- **Missing `Cargo.toml` dep** → compile error. *Mitigation:* `use`-audit per
  crate before extraction; iterate on the (fast, isolated) new crate's build.
- **`#[cfg(test)]` dev-dependency cycles** (e.g. physics tests use a solver
  plugin). *Mitigation:* permitted by Cargo; declare as `[dev-dependencies]`.
- **`gpu`/`infrastructure` cross-cutting refs.** *Mitigation:* keep in the facade
  crate for phase 1; promote later only if a clean boundary emerges.

## Verification

- After each extraction: `cargo build --workspace` green; extracted layer's tests
  pass; `cargo doc --no-deps` clean.
- End state: measure cold vs warm incremental rebuild time for a one-line edit in
  `clinical` — expected to drop from whole-crate (5–8 min) to the changed crate +
  facade only.
- `cargo-semver-checks` on the `kwavers` facade before the 4.0.0 tag.

## Implementation notes

### `kwavers-core` (first extraction, 2026-06-01)

Extracting `core` exposed real foundation-layer debt the single-crate build had
hidden: `KwaversError` is coupled to **5 higher-layer crates** via `From`/`#[from]`
(`wgpu`, `flume`, `ritk_registration`, `nifti`, `anyhow`). The orphan rule pins
these conversions to the `KwaversError`-owning crate, so they cannot move out.
Resolution that keeps the foundation a clean leaf:
- `anyhow` → normal dep (foundation-appropriate `Other` aggregation variant);
- `wgpu`/`flume`/`ritk-registration`/`nifti` → **optional deps behind features**
  (`gpu`/`channels`/`registration`/`nifti`); the `#[from]`/`From` impls are
  `#[cfg(feature=…)]`-gated. Default `kwavers-core` = clean leaf; the `kwavers`
  facade enables `channels`+`registration` always (those deps were unconditional
  in the monolith) and forwards `gpu`/`nifti`. Behaviour is byte-identical to the
  pre-split feature matrix.
- Also needed `log`'s `std` feature (for `SetLoggerError: std::error::Error`).

Mechanics: `git mv kwavers/src/core → crates/kwavers-core/src`, `mod.rs`→`lib.rs`,
60 internal `crate::core::` → `crate::`, facade `pub use kwavers_core as core`.
Verified: kwavers-core compiles default + all-features; full `kwavers` build green.

**Expect the same `From`-coupling pattern at the next boundaries** (e.g. domain
error conversions): audit each layer's error `From` impls before extraction and
apply the optional-feature-gate pattern.

### `kwavers-math` (second extraction, 2026-06-01)

No error-coupling (clean). Surfaced a second recurring pattern: **`pub(crate)`
items shared cross-module via `crate::math::…` become invisible across the crate
boundary** and must be promoted to `pub`. Found: `geometry::{distance3,
normalize3, orthogonal_basis_from_normal3}` (caught as an unused-import warning on
the now-dead-within-crate re-export) and `StaggeredGridOperator::{dx,dy,dz}`
fields (E0616 in `kwavers-solver`'s velocity updater). Also needed `log` + `rand`
deps (used via inline `log::…!`/`rand::random` macros with no `use`, so the
`use`-audit missed them).

**Pre-extraction audit checklist (apply to domain/physics/solver/…):**
1. error `From`/`#[from]` impls referencing foreign crates → optional+feature-gate;
2. `grep "pub(crate)"` items referenced externally via `crate::<layer>::…` → `pub`;
3. inline crate-path refs without `use` (`log::`, `rand::`, `rayon::`) → add deps;
4. doctests using the old `kwavers::<layer>::…` path → `<new_crate>::…`.

## Alternatives considered

- **Full path rewrite (`kwavers_core::` everywhere).** Rejected: 3,377+ edits,
  high regression risk, no benefit over the facade.
- **Do nothing.** Rejected: the 5–8 min build tax compounds on every task.
- **Single split (lib vs everything).** Rejected: doesn't help incremental builds
  meaningfully; the win comes from isolating the high-fan-in foundation layers.
