# ADR 029: Cloud-scale acoustic shielding of the cavitation cloud

- Status: Accepted
- Date: 2026-06-19
- Change class: [major] (collective dynamics; opt-in behavioral change)
- Builds on: ADR 027 (per-cell time-resolved dynamics) and ADR 028 (inter-bubble
  acoustic coupling). Advances CLD-1 toward cloud-scale collective behaviour.

## Context

ADR 028 gave each bubble its neighbours' radiated pressure, but every cell still
sees the **full** incident external pressure. In a dense cloud this is wrong: the
peripheral bubbles scatter and absorb the incident wave, so bubbles deep in the
cloud experience a **reduced, screened** driving field — the cloud edge *shields*
the interior (Maeda & Colonius 2018; Wang & Brennen 1994). This screening, with
the ADR-028 inward coupling, is what produces the cloud-scale collective response
(energy focusing toward the interior on collapse).

kwavers already has the bubbly-medium acoustics: `commander_prosperetti_attenuation`
(Commander & Prosperetti 1989) gives the resonant-scattering amplitude attenuation
`α(f, β)` [Np/m] of a monodisperse bubble cloud of void fraction `β`.

## Decision

Add **directional cloud-scale shielding**: screen the incident pressure as it
penetrates the cloud, using the validated Commander–Prosperetti attenuation.

1. **Void fraction per cell**: `β = n·(4/3)π R³` from the seeded number density `n`
   (`density_field`) and the current representative radius `R` (`radius_field`).
2. **Per-cell attenuation**: `α = commander_prosperetti_attenuation(f_drive, β, R0,
   c_L, ρ_L, μ_L, p0, γ)` (reuses the validated model; linearized about `R0`).
3. **Beer–Lambert screening** along the configured incident axis: for a wave
   entering one face, the field driving the cell at column-depth `k` is
   `p_eff = p_ext · exp(−τ_k)`, where the optical depth `τ_k = Σ_{m<k} α_m·Δs +
   ½ α_k·Δs` accumulates the attenuation of all bubbles between the entry face and
   the cell centre. This is applied per column independently (`O(N)`).
4. The screened field replaces the external pressure in `evolve_cloud`; the
   ADR-028 inter-bubble coupling is then added on top.
5. **Toggle**: `shielding_enabled` (opt-in, default `false`) + `incident_axis`
   (0/1/2) + `incident_from_high` (entry face). Off, or with zero void fraction,
   `τ ≡ 0` ⇒ `p_eff = p_ext` ⇒ reduces **exactly** to ADR 027/028.

`CloudParameters` gains `shielding_enabled`, `incident_axis`, `incident_from_high`.

## Why this is correct (validation plan)

- **Reduces exactly to ADR 027/028**: zero void fraction (no nuclei) or
  `shielding_enabled=false` ⇒ `α=0` ⇒ `p_eff=p_ext`; the keystone (cell ==
  standalone Keller-Miksis) and coupling tests still hold.
- **Beer–Lambert decay**: a uniform-void-fraction column screens the field as
  `p_eff(k)/p_ext = exp(−α·Δs·(k+½))` — value-semantic against the closed form.
- **Edge vs interior**: the entry-face cell is unscreened; interior cells are
  monotonically more screened.
- **Monotone in void fraction**: a denser cloud (larger `β`) screens the interior
  more strongly (larger `α`).

## Consequences

- Dense clouds now screen their interior — the second cloud-scale collective effect
  (with ADR-028 coupling). `O(N)` per step (a prefix sum per column), much cheaper
  than the `O(active²)` coupling. Reuses the validated Commander–Prosperetti model
  (SSOT, no cloned attenuation).
- Opt-in (default off) to preserve existing workflows/results; enabled per study.

## Residual (still open — CLD-1)

The attenuation is linearized about `R0` (not the instantaneous `R(t)`); the
screening is along a single configured axis (not a full multi-directional / scattered
field); cloud-interface instabilities and a self-consistent (implicit) coupled
shielding+collapse solve remain open, as does an experimental/k-Wave erosion
comparison. This is the leading screening model, not the complete collective solve.
