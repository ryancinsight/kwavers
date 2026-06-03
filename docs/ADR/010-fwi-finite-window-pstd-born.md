# ADR 010 — Finite-window PSTD Born forward boundary

- **Status:** Implemented (and extended beyond the original scope)
- **Date:** 2026-05-24 · **Audited:** 2026-06-03
- **Change class:** [minor]
- **Relates:** paths per [ADR 011](011-workspace-crate-split.md)

## Context

The Ali 2025 reduced breast-UST probe separates two contracts: stationary
frequency-domain Helmholtz/CBS operators, and finite-window PSTD acquisition with
source differencing, k-space source correction, and trailing-cycle demodulation.
`PstdSpectralConvergentBornOperator` shares the homogeneous PSTD modal symbol but
remains a stationary CBS operator; applying finite-window source/bin transfer to
only the stationary source projection over-amplified heterogeneous scattering
increments by ≈ 985–989× on the determined (4,4,3) probe.

## Decision

Add a solver-owned forward diagnostic that implements the first-order finite-window
PSTD Born recurrence directly:

```text
p0[n+1] = (2 − θ²)·p0[n] − p0[n−1] + S[n]      (background)
ps[n+1] = (2 − θ²)·ps[n] − ps[n−1] + χ·(…)     (scattered, χ = (s² − s0²)/s0²)
```

Expose it as `solver::inverse::fwi::frequency_domain::simulate_pstd_finite_window_born_observation`,
with a conversion-only PyO3 wrapper. The adjoint gradient was explicitly *not*
part of this decision.

## Current state (audited 2026-06-03)

Implemented and since extended. In
`crates/kwavers-solver/src/inverse/fwi/frequency_domain/finite_window.rs`:

- `simulate_pstd_finite_window_born_observation` — `:72` (recurrence at `:181`,
  `chi` at `:181/754`).
- PyO3 wrapper `simulate_breast_fwi_pstd_finite_window_born_observation` —
  `crates/kwavers-python/src/breast_fwi_bindings/finite_window.rs:28`
  (GIL released via `py.detach`, `:52`).
- **Beyond the original scope:** the discrete adjoint gradient
  `finite_window_pstd_born_gradient` (`:246`) now exists — contradicting this
  ADR's original "adjoint not implemented" consequence — as does a second-order
  Born term `simulate_pstd_finite_window_born_second_order_observation` (`:547`,
  PyO3 wrapper at `:79`).
- Tests: `…/frequency_domain/tests/{inversion,gradient_fd}.rs` and
  `crates/kwavers/tests/pstd_finite_window_born.rs`.

## Consequences

- The finite-window acquisition is modeled by its own recurrence rather than
  abusing the stationary CBS operator; the ≈985× over-amplification is resolved.
- The adjoint gradient and second-order term are now landed; a follow-up note
  should document the adjoint-gradient theorem behind `finite_window_pstd_born_gradient`.
- Residual risk: re-verify the Ali 2025 reproduction
  (`crates/kwavers-python/examples/replicate_ali2025_breast_fwi.py`).
- Module path is still `solver::inverse::fwi::frequency_domain::…` via the facade;
  physical location is `crates/kwavers-solver/…` ([ADR 011](011-workspace-crate-split.md)).
