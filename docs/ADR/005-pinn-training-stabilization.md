# ADR 005 — PINN training stabilization

- **Status:** Implemented (with one documented limitation)
- **Date:** 2026-02-02 · **Audited:** 2026-06-03
- **Change class:** [minor]
- **Relates:** paths relocated by the crate split ([ADR 011](011-workspace-crate-split.md))

## Context

The 3-D wave-equation PINN exhibited boundary-condition loss explosion during
training (BC loss ≈ 0.038 → ≈ 1.7×10³¹), causing gradient blow-up and NaN/Inf,
discovered by the BC validation suite (`tests/pinn_bc_validation.rs`). Root causes:
unbounded gradients, a fixed learning rate too large for BC-gradient magnitudes,
and loss components differing by orders of magnitude so one term dominated.

## Decision

Stabilize training with three mechanisms, plus a configurable gradient-norm bound:

1. **Adaptive learning-rate scheduling** — decay on plateau toward a floor.
2. **EMA loss-component normalization** — each loss term divided by an
   exponential-moving-average scale so data/PDE/BC/IC contribute comparably.
3. **NaN/Inf early stopping** — abort with a typed error on divergence.

Gradient clipping was specified as a `max_grad_norm` config field but **deferred**
in application because the Burn version in use does not expose the gradient
introspection needed to clip.

## Current state (audited 2026-06-03)

Implemented as decided. Code lives in
`crates/kwavers-solver/src/inverse/pinn/ml/burn_wave_equation_3d/`:

- Adaptive LR scheduler — `solver/training.rs:111-115` (config), `:199-218` (decay);
  optimizer recreated per step at `:184`.
- EMA loss normalization — `LossScales` in `solver/losses.rs:20-37`, applied
  `L/(scale+ε)` at `:153-163`.
- NaN/Inf early stop — `solver/training.rs:157-171` (returns
  `KwaversError::InvalidInput("Training diverged…")`).
- `max_grad_norm = 1.0` — `config/mod.rs:97,108,151` — **config field only, not
  applied**: `solver/training.rs:189-196` marks clipping `KNOWN_LIMITATION`
  (Burn API).

The BC validation test still exists at `crates/kwavers/tests/pinn_bc_validation.rs`
(facade crate, gated `#[cfg(feature = "pinn")]`). No explicit Xavier/He
initialization is implemented — the network uses Burn's default `LinearConfig`
init (`network/core.rs:86,94,103`).

## Consequences

- Training is stable for the BC-validation workloads; the three pillars are live.
- `max_grad_norm` remains decorative until the Burn gradient-introspection API is
  available — tracked as open Future Work (P1).
- Original ADR cited `src/solver/inverse/pinn/ml/burn_wave_equation_3d/{solver,config}.rs`;
  those files are now split into `solver/{training,losses,core}.rs` and
  `config/mod.rs` under `crates/kwavers-solver/…` ([ADR 011](011-workspace-crate-split.md)).
