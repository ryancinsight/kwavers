# ADR 009 — pykwavers elastic-wave bindings

- **Status:** Implemented (Phases A.1–A.4); A.3.5 stress-tensor source deferred
- **Date:** 2026-05-05 · **Audited:** 2026-06-03
- **Change class:** [minor]
- **Relates:** binding layer per [ADR 006](006-workspace-pyo3-bindings-architecture.md); paths per [ADR 011](011-workspace-crate-split.md)

## Context

k-Wave ships four EWP (Elastic Wave Propagation) examples exercising
`pstdElastic2D`/`3D`: layered-medium IVP, plane-wave Kelvin-Voigt absorption,
shear-wave Snell's law, and a 3-D focused velocity source. The Rust core already
had the elastic SWE capability (`ElasticWaveSolver`/`Config`/`Field`, velocity-
Verlet, PML), but it was not exposed to Python, so the examples could not be
replicated side-by-side against k-wave-python.

## Decision

Expose the elastic solver through the PyO3 bindings in phases:

- **A.1** elastic medium constructor
- **A.2** `SolverType::Elastic` + initial-displacement (IVP) source
- **A.3** velocity-source mask · **A.3.5** stress-tensor source
- **A.4** heterogeneous elastic medium + the four EWP comparison scripts

The binding stays conversion-only and dispatches to the Rust solver through the
facade (per [ADR 006](006-workspace-pyo3-bindings-architecture.md)).

## Current state (audited 2026-06-03)

A.1–A.4 landed; only A.3.5 is open. In `crates/kwavers-python/src/`:

- **A.1** `Medium.elastic(c_compression, c_shear, density, grid)` — `medium_py/mod.rs:159`.
- **A.2** `SolverType::Elastic` — `solver_type_bindings.rs:45`;
  `Source.from_initial_displacement` — `source_py/elastic.rs:92`. Dispatch reaches
  the real `ElasticWaveSolver` via `simulation_py/mod.rs:664` →
  `crates/kwavers-simulation/src/dispatch/elastic.rs`.
- **A.3** `Source.from_elastic_velocity_source(mask, ux, uy, uz, mode)` —
  `source_py/elastic.rs:24` (gained an `additive`/`dirichlet` mode arg).
- **A.3.5 — not landed.** `from_elastic_stress_source` exists only in prose; the
  shear-wave Snell's-law script uses `from_initial_displacement` as a workaround.
- **A.4** `Medium.elastic_heterogeneous(c_compression, c_shear, density, reference_frequency)`
  — `medium_py/mod.rs:186`. All four EWP compare scripts exist under
  `crates/kwavers-python/examples/` (layered 358 L, 3D 277 L, snells_law 312 L,
  plane_wave_absorption 341 L).

A `SolverType::ElasticPSTD` variant (`solver_type_bindings.rs:56`) was added beyond
the original single-engine assumption.

## Consequences

- The k-Wave EWP family is replicable from pykwavers end-to-end.
- Open: implement A.3.5 stress-tensor source; consider a follow-up ADR for the
  `ElasticPSTD` engine introduced outside this roadmap.
- The Rust elastic core moved to `crates/kwavers-solver/src/forward/elastic/swe/`
  and `crates/kwavers-domain` (medium); bindings are split across
  `crates/kwavers-python/src/{medium_py,source_py,solver_type_bindings,…}`
  ([ADR 011](011-workspace-crate-split.md)).
