# ADR 026: Photoacoustic forward-pipeline consolidation (PLC-1)

- Status: Accepted
- Date: 2026-06-19
- Change class: [arch]
- Supersedes the in-`kwavers-simulation` half of DEBT-3 (gap_audit.md).

## Context

The 2026-06-19 coverage/placement audit (PLC-1) found photoacoustic (PA) code
spread across five locations. A consumer analysis shows these are mostly **layered
concerns**, not duplicates:

| Location | Role | Status |
|----------|------|--------|
| `kwavers-physics::photoacoustics` | physics (governing eqns, Grüneisen, confinement, validity) | canonical (physics layer) |
| `kwavers-physics::analytical::photoacoustics` | closed-form formulas (absorption, Grüneisen, sphere pressure, unmixing) — drives the PyO3 bindings | canonical (analytical) |
| `kwavers-imaging::photoacoustic` | imaging data model (config, materials, scenario, state, `PressureFieldSeries`) | canonical (imaging layer) |
| `kwavers-solver::inverse::reconstruction::photoacoustic` | inversion (time-reversal, FBP, iterative) | canonical (solver layer) |
| `kwavers-simulation::modalities::photoacoustic` | **forward simulator** `PhotoacousticSimulator` (optics→acoustics→reconstruction) | **canonical forward pipeline** |
| `kwavers-simulation::photoacoustics` | parallel forward pipeline: `PhotoacousticOrchestrator` + `PhotoacousticRunner` + `vertical/{optical,source,acoustic,reconstruction}` | **dead duplicate** |

The genuine duplication is the **two parallel forward pipelines inside
`kwavers-simulation`** (the original DEBT-3).

### Evidence (consumer analysis, grep over the whole repo)

- `modalities::photoacoustic::PhotoacousticSimulator` is **live**: consumed by
  `examples/photoacoustic_imaging.rs`, the `photoacoustic_proptest`,
  `photoacoustic_validation`, and `ultrasound_physics_validation` test suites, and
  re-exported from `kwavers-simulation::lib`.
- `kwavers-simulation::photoacoustics` (orchestrator/runner/vertical, ~1325 LOC)
  is referenced **only** by its own internal files and a single
  `pub use photoacoustics::PhotoacousticRunner;` re-export in `lib.rs`. That
  re-export is consumed by **nothing** — no example, test, binding, or sibling
  crate references `PhotoacousticRunner` or `PhotoacousticOrchestrator`.
- The live `modalities` pipeline has **zero** dependency on `photoacoustics/`.
- The PyO3 PA bindings use `kwavers_physics::analytical::photoacoustics`, an
  unrelated module.

## Decision

The `kwavers-simulation::modalities::photoacoustic` `PhotoacousticSimulator` is the
**single canonical forward PA pipeline**. The parallel
`kwavers-simulation::photoacoustics` subtree (orchestrator + runner + `vertical/`)
is **dead duplicate code** and is **removed** in full, together with its `pub mod`
and `pub use` in `lib.rs`. No capability is merged: the dead pipeline has no
consumers and no test coverage, so there is nothing validated to preserve (its
history remains in git if a specific routine is ever needed).

The other four locations are correct layering and are left in place.

## Consequences

- ~1325 LOC of unmaintained parallel code removed; one forward PA pipeline remains.
- No behavioral change: the removed code was unreachable from any consumer.
- Verification: `kwavers-simulation` and the facade build clean; the PA example and
  the three PA test suites (which exercise the canonical pipeline) still pass.

## Residual (tracked, not in this change)

- PLC-2 CEUS consolidation (perfusion/microbubble physics duplicated
  `kwavers-imaging` vs `kwavers-physics`).
- PLC-3 remainder: `ceus/microbubble` vs `therapy/microbubble` dedup; therapy
  subtree living in `kwavers-physics` vs `kwavers-therapy` layering.
