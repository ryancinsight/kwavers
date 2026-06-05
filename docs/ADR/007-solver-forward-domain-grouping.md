# ADR 007 — `solver::forward` domain grouping

- **Status:** Phase 1 implemented; Phase 2 (deprecate flat paths) open
- **Date:** 2026-05-05 · **Audited:** 2026-06-03
- **Change class:** [arch]
- **Relates:** paths relocated by the crate split ([ADR 011](011-workspace-crate-split.md))

## Context

`solver/forward/mod.rs` declared 20 sibling `pub mod` entries (acoustic, bem,
bubble_dynamics, elastic, fdtd, helmholtz, hybrid, imex, nonlinear, ode, optical,
poroelastic, pstd, sem, thermal, …) with no domain grouping. The physics domains
(acoustic / elastic / thermal / optical / hybrid / ode / plugin) were flattened
into one namespace, so readers had to pattern-match on names instead of navigating
by domain — a violation of the deep-vertical-hierarchy structural rule.

## Decision

Introduce thin domain-group re-export modules (`acoustic_solvers`,
`elastic_solvers`, `thermal_solvers`, `optical_solvers`, `boundary_element`,
`hybrid_models`, `ode_methods`, `multiphysics_bubble`, `plugin`) that re-export
the existing leaf modules, **additively** — flat paths stay live (Phase 1). Phase 2
marks the flat module paths `#[deprecated]`; Phase 3 performs the destructive move.

## Current state (audited 2026-06-03)

Phase 1 done, in `crates/kwavers-solver/src/forward/mod.rs`:

- All 9 domain-group shells exist as pure `pub use` modules
  (`acoustic_solvers` `:88-94`, `elastic_solvers` `:98-101`, `thermal_solvers`
  `:104-107`, `optical_solvers` `:110-112`, `boundary_element` `:115-117`,
  `hybrid_models` `:121-124`, `ode_methods` `:127-129`, `multiphysics_bubble`
  `:132-134`, `plugin` `:137-139`).
- `path_equivalence_tests` (`:151-229`) prove flat ≡ grouped resolution at compile
  time (the ADR's verification requirement).
- The flat module list is now **16, not 20**: `acoustic`, `elastic_wave`, and
  `sem` were removed/consolidated by the elastic-as-PSTD-plugin merge documented in
  the `mod.rs` header (`:35-41`), which post-dates the original member table.
- **Partial Phase 3:** the deprecated top-level flat *item* re-exports
  (`forward::FdtdSolver`, …) were removed 2026-06-02 (`mod.rs:145-149`), but the
  flat *module* paths (`forward::fdtd`) remain live and `lib.rs:32-35,56-59` still
  re-exports `FdtdSolver`/`PSTDSolver`/`fdtd`/`pstd` at the crate root.
- **Phase 2 not done:** no `#[deprecated]` markers on the flat module paths.

## Consequences

- Navigation by domain is available now without breaking any caller.
- Open: apply Phase 2 `#[deprecated]` to flat module paths, then complete the
  Phase 3 move and drop the flat paths.
- Original ADR cited `kwavers/src/solver/forward/mod.rs`; the file is now at
  `crates/kwavers-solver/src/forward/mod.rs` ([ADR 011](011-workspace-crate-split.md)).
