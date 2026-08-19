# ADR 032: Cavitation-cloud refinements — dp/dt coupling, R(t) shielding, interface-instability diagnostic, sparse coupling solver

- Status: Accepted
- Date: 2026-06-20
- Change class: [major] (four opt-in refinements / diagnostics; CLD-1 frontier)
- Builds on: ADR 027–031.

## Context

Four items remained open on CLD-1 after ADR 031. Each is implemented opt-in (default
behaviour unchanged) and validated; two are honest *refinements with caveats*, one is
a *diagnostic* (the full nonlinear instability is research-grade), one is *numerics*.

## Decisions

### 1. `dp/dt` coupling (opt-in: `couple_pressure_rate`)
The Keller-Miksis acceleration depends on the driving pressure **and its rate**
`dp/dt` (the acoustic-radiation term). The Pass-2 *integration* already uses
`dp_dt = (p_total − p_prev)/dt` (coupling included, finite-difference). The gap was
the **source strengths** `S_j = R_j²R̈_j + 2R_jṘ_j²` evaluating `R̈` at `dp/dt = 0`.
When enabled, the per-cell finite-difference rate `(driving_j − prev_total_j)/dt`
(lagged) is threaded into the source/affine acceleration evaluations. Because `R̈` is
affine in `p` *and* `dp/dt`, the direct solve's slope `d_j = ∂R̈/∂p` is unchanged and
the rate term is absorbed into the constant `c_j` — so the linear system is still
exact. The rate itself is explicit (lagged), documented.

### 2. `R(t)`-dependent shielding (opt-in: `shielding_radius_dependent`)
`shielded_pressure` already forms the void fraction `β = n·(4/3)π R(t)³` from the
instantaneous radius, but the Commander-Prosperetti attenuation used the *equilibrium*
`R0` for the resonance (CP is a linear theory about `R0`). When enabled, the
instantaneous per-cell `R(t)` is used for the resonance too — a **quasi-static**
extension (valid for slow `R` variation; large-amplitude scattering is nonlinear and
beyond CP, documented).

### 3. Cloud-interface (RT/RM) instability **diagnostic**
`interface_instability` returns the **linear** growth rates at the cloud edge from
the void-fraction interface and the local acceleration / velocity jump:
- Rayleigh-Taylor: `σ_RT = sqrt(A·k·a)` for Atwood number `A`, perturbation
  wavenumber `k`, interface acceleration `a` (real growth only when the heavy fluid
  is accelerated into the light one, `A·a > 0`).
- Richtmyer-Meshkov (impulsive): amplitude rate `ȧ = k·Δv·a₀·A` (Richtmyer 1960).
`A` is computed from the effective (Wood) mixture densities across the interface.
This is a **diagnostic** (growth rates), not a nonlinear interface simulation — the
full RT/RM evolution (mushrooms, mixing) remains out of scope and is stated as such.

### 4. Sparse / matrix-free coupling solver (`CouplingScheme::ImplicitIterative`)
The dense direct solve (ADR 031) is `O(active³)` time and `O(active²)` memory — it
does not scale to large active clouds. `ImplicitIterative` solves the same system
`(I − D·G)·S = e` with the validated matrix-free `solve_lsqr_matfree`: a
`MatFreeOperator` applies `M·x = x − D·(G·x)` (and `Mᵀ = I − G·D`, `G` symmetric) by
computing `G_ab = ρ/d_ab` on the fly from cell positions within the cutoff — `O(active)`
memory, `O(active·neighbours)` per matvec. For moderate active counts the dense
`ImplicitDirect` remains; for large counts use `ImplicitIterative`.

## Validation

1. `couple_pressure_rate` changes the source strength when `dp/dt ≠ 0`; off ⇒ identical
   to before.
2. `shielding_radius_dependent`: attenuation differs when `R ≠ R0`; equals the
   `R0` result when `R = R0`.
3. `interface_instability`: `σ_RT = sqrt(A·k·a)` matches the closed form; stable
   (zero growth) when the light fluid is on top (`A·a ≤ 0`); RM rate matches
   `k·Δv·a₀·A`.
4. `ImplicitIterative` matches `ImplicitDirect` (and a converged fixed point) on a
   moderate cloud to the solver tolerance.

All four are opt-in; defaults reproduce ADR 027–031 exactly.

## Residual (CLD-1)

Nonlinear RT/RM interface evolution and its feedback on collapse; fully implicit
`dp/dt` coupling; nonlinear (large-amplitude) cloud scattering; experimental/k-Wave
end-to-end erosion comparison.
