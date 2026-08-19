# ADR 028: Inter-bubble acoustic coupling in the cavitation cloud

- Status: Accepted
- Date: 2026-06-19
- Change class: [major] (collective dynamics; behavioral change for multi-cell clouds)
- Builds on: ADR 027 (per-cell time-resolved bubble dynamics). Advances CLD-1 from
  *independent* oscillators toward *collective* cloud dynamics.

## Context

After ADR 027 each cloud cell is a real, time-resolved Keller-Miksis bubble driven
by its **local external** pressure — but the cells are independent. The leading
collective effect is **acoustic coupling**: a pulsating bubble `j` radiates a
pressure that adds to the driving pressure seen by its neighbours. In the
incompressible (near-field) limit a bubble of radius `R_j(t)` radiates, at distance
`d`,

```text
p_rad(d, t) = (ρ_L / d) · d/dt(R_j² Ṙ_j) = (ρ_L / d) · (R_j² R̈_j + 2 R_j Ṙ_j²) .
```

This is the basis of secondary Bjerknes forces and coupled-bubble resonance
(Mettin et al. 1997; Ida 2002; Bremond et al. 2006). The total driving pressure on
bubble `i` becomes `p_ext(x_i) + Σ_{j≠i} (ρ_L/d_ij)·S_j`, with source strength
`S_j = R_j² R̈_j + 2 R_j Ṙ_j²`.

## Decision

Add **explicit (lagged) pairwise acoustic coupling** among active cloud cells:

1. **`bubble_radiated_pressure(ρ, d, R, Ṙ, R̈)` = `(ρ/d)(R² R̈ + 2 R Ṙ²)`** — a
   free function (the incompressible radiated near-field pressure), unit-tested
   against the closed form.
2. **Two-pass `evolve_cloud`**:
   - *Pass 1 (source strengths):* for each active cell `j`, compute
     `S_j = R_j² a_j + 2 R_j Ṙ_j²`, where `a_j` is the Keller-Miksis acceleration
     evaluated at the **previous total** driving pressure (fully explicit ⇒ no
     implicit all-bubble solve; stable for acoustic-resolution `dt`).
   - *Pass 2 (integrate):* for each active cell `i`,
     `p_total_i = p_ext_i + Σ_{j≠i, d_ij≤r_cut} (ρ/d_ij)·S_j`, then advance the cell
     with the adaptive Keller-Miksis integrator under `p_total_i` and
     `dp/dt = (p_total_i − prev_total_i)/dt`.
3. **Geometry**: cell `i` sits at `(i·dx, j·dy, k·dz)` from `CloudParameters::
   cell_spacing`; an `interaction_radius` cutoff bounds the `O(active²)` sum.
4. **Toggle**: `coupling_enabled` — **opt-in, default `false`** (revised from the
   initial draft after verification). The coupling sum is `O(active²)` per step and
   amplifies the drive into the stiff violent-collapse regime, where the adaptive
   integrator burns its full sub-step budget per cell; enabling it by default made
   the full-grid lithotripsy orchestrator sim exceed the 60 s test ceiling. So it
   is opt-in: existing workflows keep the fast, validated ADR-027 per-cell path, and
   collective studies enable coupling on tractable problem sizes (and set a finite
   `interaction_radius`). With it off, or with a single active cell, the coupling
   sum is empty and the model reduces **exactly** to ADR 027.

`CloudParameters` gains `cell_spacing`, `coupling_enabled`, `interaction_radius`
(Default-constructed; sole call site unaffected). The per-cell previous-pressure
state becomes the previous *total* driving pressure.

## Why this is correct (validation plan)

- **Reduces exactly to ADR 027**: a single active cell (or `coupling_enabled=false`)
  ⇒ empty coupling sum ⇒ the per-cell keystone (cloud cell == standalone
  Keller-Miksis) still holds, re-tested.
- **Radiated-pressure closed form**: `bubble_radiated_pressure` equals
  `(ρ/d)(R²R̈+2RṘ²)` for known inputs (value-semantic).
- **`1/d` scaling**: the coupling pressure a fixed source induces on a probe halves
  when the separation doubles.
- **Symmetry**: two identical bubbles induce equal coupling on each other.
- **Off ⇒ on differs only via the coupling term**: disabling coupling recovers the
  uncoupled trajectory.

## Consequences

- Multi-cell clouds now carry the leading collective (acoustic-coupling) effect —
  the dominant inter-bubble interaction and the mechanism of secondary Bjerknes
  forces. Reuses the validated Keller-Miksis solver (SSOT).
- **Cost**: `O(active² )` per step (bounded by `interaction_radius`); documented.

## Residual (still open — CLD-1)

Self-consistent (implicit) coupling; cloud-scale energy focusing / shielding where
the cloud edge screens the interior (Maeda & Colonius 2018); Rayleigh-Taylor /
Richtmyer-Meshkov cloud-interface instabilities; compressible (retarded-time)
coupling. The coupling here is incompressible and explicit — the leading, not the
complete, collective model.
