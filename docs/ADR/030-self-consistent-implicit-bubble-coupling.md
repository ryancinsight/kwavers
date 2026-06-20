# ADR 030: Self-consistent (implicit) inter-bubble coupling

- Status: Accepted
- Date: 2026-06-19
- Change class: [major] (opt-in numerical-scheme upgrade to the collective coupling)
- Builds on: ADR 028 (explicit inter-bubble coupling), ADR 029 (shielding).

## Context

ADR 028 couples bubbles **explicitly**: each step's source strengths
`S_j = R_j² R̈_j + 2 R_j Ṙ_j²` are evaluated at the *previous* total pressure
(lagged), then each bubble is driven by `p_ext + Σ_{k≠j}(ρ/d_jk)S_k`. The lag is an
`O(dt)` error: at the start of the step a bubble's own radiated pressure and its
neighbours' driving are not mutually consistent. For strong coupling or larger `dt`
this matters.

A fixed point exists and is reachable: the Keller-Miksis acceleration is **affine
in the driving pressure**, `R̈_j = a_j + b_j·p_drive,j`, so the source strength
`S_j = c_j + d_j·p_drive,j` is affine too, and `p_drive,j = p_ext,j +
Σ_{k≠j}(ρ/d_jk)S_k` closes a linear system `(I − D·G)·S = c + D·p_ext`. The
explicit scheme is its first Jacobi iterate from a lagged guess; iterating to
convergence gives the self-consistent (implicit) solution.

## Decision

Add an **opt-in self-consistent coupling** computed by fixed-point iteration on the
coupling field (reusing the canonical Keller-Miksis acceleration each iterate, so no
hand-derived `a_j/b_j`):

1. Collect active cells `(position, R, Ṙ)`.
2. Initialise the per-cell coupling pressure `p_couple = 0`.
3. **Iterate** up to `coupling_max_iterations`:
   - `S_a = R_a² · R̈_a + 2 R_a Ṙ_a²`, with `R̈_a = KM_accel(state_a, p_ext_a +
     p_couple_a)` (the current self-consistent driving);
   - `p_couple_a ← Σ_{b≠a, d_ab≤r_cut}(ρ/d_ab)·S_b`;
   - stop when `max_a |Δp_couple_a| < coupling_tolerance·(|p_ext|‖∞ + p0)`.
4. Drive each bubble with the converged `p_ext + p_couple` and integrate.

`CloudParameters` gains `implicit_coupling` (opt-in, default `false`),
`coupling_max_iterations`, `coupling_tolerance`. When off, the explicit (ADR 028)
scheme is used unchanged. When coupling is disabled, or a single active cell, both
reduce **exactly** to ADR 027/029.

## Why this is correct (validation plan)

- **Self-consistency**: at the returned field, recomputing `p_couple` from the final
  `S` reproduces it within tolerance (the fixed-point equation holds) —
  value-semantic.
- **Convergence**: the per-iterate residual decreases monotonically toward the
  tolerance for weak/moderate coupling.
- **Explicit limit**: for weak coupling the implicit result ≈ the explicit
  (one-iterate) result; the difference grows with coupling strength.
- **Reduction**: coupling-off or single cell ⇒ `p_couple = 0` ⇒ exactly ADR 027/029
  (keystone + prior coupling/shielding tests still hold).

## Consequences

- Removes the `O(dt)` lag error in the coupling for moderate strengths; the leading
  collective interaction is now self-consistent within each step. Reuses the
  validated KM acceleration (SSOT).
- **Cost**: `O(iterations · active²)` per step (vs `O(active²)` explicit); opt-in,
  so default workflows are unaffected. Strong coupling beyond the contraction regime
  may not converge in the plain fixed point — capped by `coupling_max_iterations`
  (best effort; under-relaxation / direct linear solve is the documented next step).

## Residual (still open — CLD-1)

Direct (non-iterative) linear solve / under-relaxation for the strong-coupling
(non-contractive) regime; coupling of `dp/dt` (only the instantaneous pressure is
made self-consistent); `R(t)`-dependent shielding; cloud-interface instabilities;
experimental/k-Wave erosion comparison.
