# ADR 019 — Marchenko redatuming + Wasserstein objective (prior-less FWI)

**Status:** Accepted (kernel + objective delivered; quantitative Marchenko validation = staged milestone)
**Change class:** [minor] (additive module + objective; experimental redatum gated by docs)
**Date:** 2026-06-08
**Builds on:** existing Wasserstein misfit (`MisfitType::Wasserstein`); ADR 016 (exact SA engine, candidate oracle).

## Context

The bleeding-edge "prior-less" FWI vision combines two methods that fix the two
dominant FWI failure modes:

- **Marchenko redatuming (physics):** retrieves the focusing functions `f1±` and
  Green's functions `G±` between the surface and a virtual subsurface point from
  the surface reflection response `R(t)` and a single kinematic input (the one-way
  time `t_d`), *without the overburden velocity model*. It uses internal multiples
  correctly, so the redatumed wavefield is free of overburden reverberation
  artefacts — unlike correlation/back-propagation redatuming.
- **Wasserstein / optimal transport (mathematics):** an OT misfit is convex in
  time shift, so the gradient points to the true model regardless of initial
  phase — defeating cycle-skipping. (Already in kwavers and verified:
  `wasserstein_is_convex_in_shift_on_positive_distribution`.)

Combined objective: `J(m) = W_p(G⁻_obs, G⁻_mod(m))` — compare *redatumed* Green's
functions with an OT distance.

## Decision

Add `inverse::marchenko` with:

- **Verified operators** — causal windowed convolution `conv_causal`, correlation
  `corr_causal`, and the truncation window `apply_window` (unit-tested exactly).
- **1-D iterative redatuming** `redatum` — the fixed point
  ```text
  f1⁻ = θ (R * f1⁺),   f1⁺ = f1d⁺ + θ (R ⋆ f1⁻),   G⁻ = (R * f1⁺) − f1⁻
  ```
  with `f1d⁺(t)=δ(t+t_d)` and `θ` muting `|t| ≥ t_d−ε`. **Marked experimental:**
  the iterative *structure* follows Wapenaar et al. (2014) / Thorbecke et al.
  (2017), but the focusing **amplitude/window convention is not yet validated
  against an independent layered-medium reference** (the window/coda geometry is
  convention-sensitive — discovered during implementation: a naïve symmetric
  window can window the coda update to zero). Tests assert only verified
  properties (operator correctness, finite/post-focal Green's function).
- **Combined objective** `marchenko_wasserstein_misfit(R_obs, R_mod, cfg)` —
  composes `redatum` with the validated 1-Wasserstein misfit; tested well-posed
  (self-distance 0, positive for differing data) independently of redatum's
  quantitative status.

## Alternatives considered

- **Inline Marchenko into FWI now.** Premature — the redatuming convention must be
  reference-validated first; shipping unvalidated focusing as "working" would
  violate the integrity bar.
- **Frequency-domain Marchenko.** Time-domain iterative is the standard, simplest
  correct 1-D form and reuses the trace conventions already in the codebase.

## Validation plan (staged milestones)

1. **Quantitative 1-D `redatum` validation** `[major]` — the SA-engine oracle is
   **built** (`marchenko::oracle_tests::redatum_matches_engine_green_function`,
   `#[ignore]`d as the acceptance target): a 1-D layered medium with a transparent
   sponge surface yields `R` (source+receiver at the surface, direct muted) and
   the true `G⁻` (virtual source at `z_f`). **Empirical finding (2026-06-08):**
   `corr(Marchenko,true) = corr(naive,true) ≈ 0.14` — the iteration does not yet
   engage (coda ≈ 0 ⇒ Marchenko == naive). Root-caused blockers to fix before
   un-ignoring:
   - **Window/record geometry:** with focal `t_d` deep, the inter-interface
     internal-multiple period `2(z_f−z_i)/c` can land *beyond `nt`* (off-record)
     or map *outside* the `|t| < t_d−ε` focusing window, so no in-window coda
     forms. The experiment must be designed (interface depths, `t_d`, record
     length, window taper) so the relevant multiples are captured and in-window.
   - **Convolution/correlation convention** of the `f1⁺` update (both `R*` and
     `R⋆` send the single in-window update to the window edge for the tested
     geometry — needs the correct paired operators per Wapenaar 2014 eqs. 9–10).
   - **`G⁻` time-referencing** on the symmetric axis vs the surface recording, and
     **amplitude normalisation** by the direct transmission `T_d` (the comparison
     is scale-invariant cosine, so amplitude is secondary to timing/shape).
   Acceptance: `corr(Marchenko,true) > 0.85` and `> corr(naive,true)`.
2. **Multidimensional Marchenko** `[major]` — extend to multi-trace acquisition
   (R as a t-x matrix), up/down decomposition, point-spread-function handling.
3. **Marchenko–Wasserstein FWI loop** `[major]` — drive a model update from
   `∂J/∂m` of the combined objective (layer-stripping / bottom-up), reusing the
   exact SA-engine gradient infrastructure (ADR 016).

## Consequences

- New module `inverse::marchenko` (operators, experimental `redatum`,
  `marchenko_wasserstein_misfit`, `redatum_naive`).
- The "taming the math" half (Wasserstein) is production-ready; the "taming the
  physics" half (Marchenko) ships as a documented, honestly-scoped kernel pending
  reference validation — not claimed as quantitatively correct.
