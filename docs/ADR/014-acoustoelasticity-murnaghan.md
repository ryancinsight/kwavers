# ADR 014 — Acousto-elasticity: Murnaghan Stress-Dependent Wave Speed + Pre-Stress Inversion

**Status:** Accepted
**Change class:** [major] (new constitutive relation + inversion); scope split below
**Date:** 2026-06-06

## Context

Elastography §11.9 documents nonlinear elastography via the **Murnaghan** third-order elastic
constants `(l, m, n)`: a static pre-stress `σ₀` shifts the shear-wave speed, and measuring that
shift over a load cycle (e.g. cardiac pulsation) recovers the in-situ stress or the nonlinear
constants (Algorithm "Pre-Stress Estimation"). The book audit found this **not implemented**
(the nonlinear elastic solver uses hyperelastic Neo-Hookean / Mooney-Rivlin / Ogden — a
different formulation; no Murnaghan / acousto-elastic path exists).

## Decision

Implement the **analytical acousto-elastic relation and its inversion** in
`kwavers_physics::analytical::elastography` (alongside the existing `shear_wave_speed` /
Voigt helpers):

- `acoustoelastic_sensitivity(m, n, lambda, mu)` → `A = (m+n) / (2(λ+μ))`.
- `acoustoelastic_shear_speed(mu, lambda, m, n, rho, sigma0)` → `c_S = √((μ + A·σ₀)/ρ)`
  (first-order Hughes–Kelly / Murnaghan relation, §11.9).
- `estimate_prestress(c_s, c_s0, rho, A)` and `estimate_prestress_sequence(...)` →
  `σ₀ = ρ(c_S² − c_S0²)/A` (Algorithm: slope inversion of `ρc_S²` vs `σ₀`).

**Scope split:** the *closed-form* relation + inversion (what §11.9 actually documents and what
the cardiac-cycle algorithm needs) is implemented now. A full **3rd-order elastic-wave PDE
solver** (Murnaghan terms in the time-domain elastic update) is a separate, larger item and is
explicitly deferred — it is not required by the documented relation/algorithm.

## Alternatives considered

- Full 3rd-order elastic FDTD now — large, and not needed to deliver the §11.9 capability;
  deferred behind its own ADR if/when a nonlinear forward field is required.
- Putting it in the inverse-elastography solver — the relation is analytical and PyO3-friendly,
  so it belongs with the other analytical elastography helpers.

## Verification

- Forward at `σ₀ = 0` equals `√(μ/ρ)` (consistency with `shear_wave_speed`).
- Forward is monotonic in `σ₀` and matches the closed form.
- Round-trip: `estimate_prestress(acoustoelastic_shear_speed(σ₀)) == σ₀`.
- `acoustoelastic_sensitivity` equals the algebraic `(m+n)/(2(λ+μ))`.

## Consequences

- Delivers quantitative pre-stress estimation from shear-speed-vs-load data.
- First-order relation; second-order `O(σ₀²)` terms and a full nonlinear forward field remain.
