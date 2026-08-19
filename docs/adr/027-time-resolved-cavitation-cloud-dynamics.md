# ADR 027: Time-resolved per-cell bubble dynamics for the cavitation cloud

- Status: Accepted
- Date: 2026-06-19
- Change class: [major] (behavioral change to lithotripsy/histotripsy cloud erosion)
- Builds on: CLD-1 (gap_audit) — the prior increment replaced the static-R₀ erosion
  proxy with a per-call Gilmore `inertial_collapse_energy` sized from the field's
  *peak* rarefactional pressure. This ADR removes the remaining snapshot limitation.

## Context

The cavitation cloud (`kwavers-therapy::...::lithotripsy::cavitation_cloud`) tracked
only a per-cell bubble *number density* with ad-hoc growth/collapse rates; the
previous increment computed a physics-based collapse energy but from a single
field-peak pressure per `evolve_cloud` call — so the bubble dynamics were not
*localized* and not *time-resolved* (no per-cell R(t), no history). `evolve_cloud`
receives an instantaneous pressure field each call, so any real bubble history must
live in per-cell state carried across calls.

kwavers already has the engine: `KellerMiksisModel` (instantaneous `p_acoustic` +
`dp_dt`) and `integrate_bubble_dynamics_adaptive`, which advances a `BubbleState`
by `dt` with adaptive sub-stepping that resolves violent collapse.

## Decision

Make each cloud cell a **real, independent bubble oscillator** with time-resolved
state `(R, Ṙ)` integrated across `evolve_cloud` calls by the canonical adaptive
Keller-Miksis integrator under the **local** instantaneous pressure:

1. **Per-cell state** (lightweight `Array3<f64>`): `radius_field`, `velocity_field`,
   `r_max_field` (running max since last collapse), and `prev_pressure` (for the
   `dp_dt` finite difference across calls). No heavy `BubbleState` is stored — a
   `BubbleState` is reconstructed per step from `(R, Ṙ)` + shared `BubbleParameters`.
2. **Pure-mechanical bubble**: `use_thermal_effects = false`,
   `use_mass_transfer = false`, so the cell state is exactly `(R, Ṙ)` (a 2nd-order
   ODE) — deterministic and reconstructible.
3. **Per step, per active cell**: `dp_dt = (p_local − prev_p_local)/dt`; advance the
   cell via `integrate_bubble_dynamics_adaptive(solver, &mut state, p_local, dp_dt,
   dt, time)`; update `r_max`.
4. **Erosion = event-based per real collapse**: when a cell completes a collapse
   (wall velocity transitions `Ṙ < 0 → Ṙ ≥ 0` at the radius minimum), deposit
   `density · (4/3)π r_max³ (p₀ − p_v) · erosion_efficiency` and reset `r_max` to
   the current radius. Each genuine inertial collapse erodes, scaled by the *actual*
   maximum radius the bubble reached locally.
5. **Density** is the seeded bubble number density (from `initialize_cloud`); it
   modulates erosion magnitude. The ad-hoc density growth/collapse *rates* are
   removed (the real R(t) supersedes them).

The public API (`new`, `evolve_cloud`, `initialize_cloud`) is preserved; new
accessors expose `radius_field`. Behavior changes (real per-cell dynamics) — hence
[major].

## Why this is correct (validation plan)

- **Keystone differential test**: a 1-cell cloud advanced through a pressure
  sequence reproduces, bit-for-bit, the standalone
  `integrate_bubble_dynamics_adaptive` called directly with the same `(p, dp_dt, dt)`
  sequence — i.e. the cloud cell *is* a real Keller-Miksis bubble, by construction
  (no fabricated dynamics).
- **Rayleigh collapse**: a bubble grown to R_max under tension collapses on the
  Rayleigh timescale `τ ≈ 0.915 R_max √(ρ/Δp)` (order-of-magnitude check).
- **Erosion localizes** where the local collapse is strongest (deeper local
  rarefaction → larger local R_max → more erosion in that cell).
- **Bounds / stability**: `R > 0` preserved; non-finite states are caught.

## Consequences

- Cloud erosion is now driven by genuine localized, time-resolved single-bubble
  dynamics — the strongest single-bubble fidelity available, reusing the validated
  solver (SSOT, no cloned ODE).
- **Cost**: per-active-cell adaptive ODE per step (O(active cells × substeps)).
  Acceptable for the lithotripsy cadence; documented. Only cells with bubbles
  (`density > 0`) are integrated.

## Residual (still open — collective / research frontier, tracked in CLD-1)

Inter-bubble acoustic coupling and emission back-reaction; cloud-scale energy
focusing (Maeda & Colonius 2018); shock-bubble Richtmyer-Meshkov / Rayleigh-Taylor
cloud instabilities; inter-phase mass transfer. The cells remain *independent*
oscillators — collective cloud collapse is not modeled and is not claimed accurate.
