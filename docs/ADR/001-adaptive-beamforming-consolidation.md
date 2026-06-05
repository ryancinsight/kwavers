# ADR 001 — Adaptive-beamforming consolidation

- **Status:** Implemented (clean removal)
- **Date:** 2025-11-03 · **Audited:** 2026-06-03
- **Change class:** [patch]
- **Relates:** subsumed by [ADR 002](002-sensor-array-processing-consolidation.md); placement finalized by [ADR 003](003-signal-processing-analysis-layer.md)

## Context

The adaptive-beamforming module carried a monolithic `algorithms_old.rs`
(2,193 lines, ~4× the 500-line target) that duplicated every algorithm already
refactored into dedicated submodules: `MinimumVariance`/MVDR, `MUSIC`,
`EigenspaceMV`, `RobustCapon`, `CovarianceTaper`, `DelayAndSum`. The duplicate
implementations had to be kept in sync, doubled the test surface, and risked
behavioral divergence between the two copies.

## Decision

Remove `algorithms_old.rs` entirely and keep exactly one implementation per
algorithm in its dedicated submodule. The duplicate was dead-end code, so the
transition was a clean break — no `#[deprecated]` shim or feature-flag bridge.

## Current state (audited 2026-06-03)

Done. `algorithms_old.rs` and the entire `adaptive_beamforming/` module no
longer exist anywhere in the tree. Following [ADR 003](003-signal-processing-analysis-layer.md),
the canonical implementations now live in the analysis crate, each as a single
authoritative file:

- `MinimumVariance` (MVDR) — `crates/kwavers-analysis/src/signal_processing/beamforming/adaptive/mvdr/mod.rs:55`
- `MUSIC` — `crates/kwavers-analysis/src/signal_processing/beamforming/adaptive/subspace/music.rs:26`
- `EigenspaceMV` — `crates/kwavers-analysis/src/signal_processing/beamforming/adaptive/subspace/esmv.rs:27`

All files are within the 500-line target; no duplicate algorithm structs remain.

## Consequences

- Single source of truth per algorithm; halved maintenance and test surface.
- The original ADR adopted a phased "Option 3: migrate with deprecation"; the
  realized route was a more aggressive clean removal, which is preferred under
  the anti-shim policy.
- This decision is a precursor to the broader array-processing unification in
  [ADR 002](002-sensor-array-processing-consolidation.md).
