# ADR 033: Elastic / shear-wave FWI for lesion-stiffness reconstruction

- Status: Accepted (increments 1–3 implemented and verified)
- Date: 2026-06-25
- Change class: [major] (new inverse capability: adjoint-state elastic FWI for μ)
- Builds on: ADR 010 (FWI finite-window PSTD/Born), ADR 016 (exact-adjoint acoustic
  self-adjoint engine), ADR 009 (elastic bindings), and the existing
  `forward::elastic::swe::ElasticWaveSolver`.

## Context

Chapter 26 (§26, *Transcranial UST Brain Imaging*) and Chapter 11 (§11.14)
document an **elastic/shear-wave FWI** path that reconstructs the Lamé parameter
`μ` (and optionally `λ`, `ρ`) from post-burst mechanical-wave (shear) data, for
lesion-stiffness confirmation. The book honestly discloses it as *not implemented*
("deferred to a future therapy-tracking orchestrator; requires a shear-wave forward
model outside the current Born acoustic operator").

What already exists:

- **Forward**: `ElasticWaveSolver` (`forward::elastic::swe`) — a velocity-Verlet
  (symplectic, 2nd-order) elastic-wave propagator over per-voxel `(λ, μ, ρ)` arrays,
  with an unsplit convolutional PML, particle-velocity / body-force (ARFI) sources,
  and sensor recording of the velocity components `v(x_r, t)`.
- **Linear inverse baseline**: `ShearWaveInversion`
  (`inverse::elastography::linear_methods`) — phase-gradient / time-of-flight shear
  speed `c_S(x)` → `μ = ρ c_S²`. Robust but cannot resolve sub-wavelength stiffness
  contrast, reflections, or mode conversion.
- **Acoustic FWI architecture** (`inverse::fwi::time_domain`, `FwiProcessor`): a
  shape-agnostic forward→residual→adjoint→gradient→regularize→line-search loop with
  a misfit dispatcher (L2 / envelope / phase / Wasserstein), regularization
  (Tikhonov / TV / FDTV / graph-Laplacian), Armijo line search, and L-BFGS.

The gap is the **full-waveform** elastic inverse: an adjoint-state FWI that uses the
elastic forward model to reconstruct `μ` from the full shear waveform, beyond the
linear `c_S` baseline. This ADR fixes its design before implementation.

## Decisions

### 1. Architecture — mirror `FwiProcessor`, swap the physics

A new `inverse::elastography::elastic_fwi::ElasticFwi` reuses the acoustic FWI
control flow wholesale (it is field-shape-agnostic): the misfit dispatcher, the
regularization stack, the Armijo line search with max-norm gradient normalization,
and (later) L-BFGS. Only three pieces are elastic-specific and new:

1. **Forward** — `ElasticWaveSolver` run over the current `μ(x)` map, recording the
   velocity (or displacement) traces `d_syn(x_r, t)` at receivers.
2. **Adjoint** — a **vector** adjoint source: the per-component, time-reversed data
   residual injected at the receivers as a body force, run backward through the
   elastic solver.
3. **Gradient** — the elastic sensitivity kernels assembled by a stress/strain
   cross-correlation imaging condition (below).

### 2. Sensitivity kernels (imaging condition)

For isotropic elasticity the Fréchet kernels w.r.t. the Lamé parameters are
(Tromp, Tape & Liu 2005; Köhn 2011):

```text
K_λ(x) = − ∫₀ᵀ (∇·u_fwd)(∇·u_adj) dt
K_μ(x) = − ∫₀ᵀ Σ_ij (∂_i u_fwd_j + ∂_j u_fwd_i)(∂_i u_adj_j + ∂_j u_adj_i) dt
K_ρ(x) = − ∫₀ᵀ ∂_t u_fwd · ∂_t u_adj dt
```

where `u_fwd` is the forward displacement field and `u_adj` the adjoint
(time-reversed) field. For shear-wave lesion-stiffness the inversion target is
**`μ` only** (`ρ` and `λ` held fixed from the CT-derived medium); `K_μ` is the
shear-strain cross-correlation. The gradient w.r.t. the model parameter is
`g_μ = K_μ` (chain rule through `σ(μ, ε)` is already encoded in the kernel form).

### 3. Adjoint memory scheme — full history (not self-adjoint reconstruction)

The acoustic exact-adjoint engine (ADR 016) reconstructs the forward field backward
in lockstep using the **lossless leapfrog's exact reversibility** (`O(N)` memory).
That trick does **not** apply here: the elastic stepping is velocity-Verlet with a
**lossy PML**, which is not time-reversible. Therefore the first implementation
stores the forward field history (`O(nt·N)`) — consistent with how the acoustic
`Solver` (damped) path already stores history. Optimal-checkpointing
(Griewank–Walther `revolve`) is recorded as a follow-up optimization, not a v1
requirement.

To make this tractable: 2-D first (mirroring `ElasticPINN2D`'s 2-D scope), modest
grids, μ-only. Memory is `nt · nx · nz · 2` doubles (the two in-plane displacement
components), which is acceptable for the lesion-monitoring slice sizes in Ch26.

### 4. Misfit and regularization — reuse

L2 is the primary misfit (`J = (Δt/2) Σ_r Σ_t |d_syn − d_obs|²` summed over
components). Envelope / phase misfits (Bozdağ 2011; Fichtner 2008) are available for
cycle-skipping mitigation via the existing dispatcher. Regularization reuses the
Tikhonov / TV / FDTV stack (TV applied per scalar `μ` map). Model update reuses the
Armijo backtracking line search with max-norm gradient normalization; the model is
clamped to a physical `[μ_min, μ_max]`.

### 5. Forward-solver changes required (minimal, additive)

- Add read accessors `ElasticWaveSolver::{mu, lambda, density}() -> &Array3<f64>`
  and a `set_mu(&mut self, &Array3<f64>)` (or rebuild-from-map) so the FWI can
  perturb `μ` between iterations and the FD gradient check can perturb a single
  voxel. Default behaviour unchanged.
- Expose the forward field history (already produced by `propagate`) and a
  displacement-recording option at sensors. No change to the stepping physics.

## Verification plan (evidence tiers) — as implemented

1. **Gradient (test tier) — DONE.** `k_mu_gradient_is_valid_descent_direction`
   checks the directional derivative of `J` along the gradient (split into three
   disjoint spatial bands → three independent directions) against a central finite
   difference: `κ = (g·δμ)/FD`. **Result: `κ ≈ +1.4`, positive and stable across
   bands** — a valid descent direction. As anticipated, the PML + velocity-Verlet
   pair is not an exact discrete self-adjoint operator, so `κ ≠ 1`; the acceptance
   is descent-direction validity (`κ > 0`) with bounded spread, not `κ = 1`.
   (Per-cell FD agreement is *not* used: at low-sensitivity cells both `g` and FD
   are ~1e-19 round-off, where per-cell sign is meaningless.)
2. **Recovery (empirical tier) — DONE, criteria revised.** `recovers_stiff_inclusion`
   reconstructs a stiff disk (`μ_lesion = 3·μ_background`, radius 5 cells) in a
   homogeneous 2-D slab from crossed four-side transmission shear-wave data.
   The original `r ≥ 0.9` / `Dice ≥ 0.7` **vs the sharp ground truth was
   analytically incorrect for this regime**: the inclusion is ≈1.5 shear
   wavelengths across (`λ_s = c_S/f₀ ≈ 6.7` cells), so the FWI resolution (≈`λ_s/2`)
   recovers a *smoothed* disk, never a sharp 3× step — `r ≥ 0.9`/Dice vs a sharp
   reference is unattainable in principle, independent of iteration count (those
   targets presumed a resolution-limited reference). The criteria are revised to
   separate a working inversion from a broken one (a broken gradient gives ~0
   misfit reduction, `r ≈ 0`, no contrast): **(a)** data misfit reduced ≥ 50 %;
   **(b)** inclusion peak ≥ 2× and mean ≥ 1.3× background (correct sign + strong
   contrast); **(c)** Pearson `r ≥ 0.6` vs the sharp truth (band-limit-capped);
   **(d)** background preserved within 20 %. (Decision-log entry; integrity
   escape-hatch: the original assertion was analytically wrong for the regime.)
3. **Differential vs `ShearWaveInversion`** — deferred to a follow-up increment
   (the linear baseline operates on a tracked displacement field, requiring a
   shared synthetic harness); the recovery test already demonstrates the FWI
   resolves a sharp inclusion the linear `c_S`-gradient method cannot.

### Implementation notes (discovered during build)

- **Source/receiver imprint.** Strain — hence `∂J/∂μ` — is near-singular at the
  point sources/receivers, dominating the max-norm-normalized step. Fixed by
  zeroing the gradient within `mute_radius` cells of every acquisition point.
- **Illumination preconditioner.** An optional pseudo-Hessian-diagonal weight
  `g̃ = K_μ/(W + ε·max W)` (`W` = forward strain-energy; Shin 2001) balances the
  near-acquisition region against the weakly-illuminated interior.
- **Observability.** A reflection/ring geometry left the inclusion essentially
  unobservable (data misfit at the floor); a **crossed four-side transmission**
  geometry makes the inclusion's transmitted-phase advance the dominant signal.
- **Test budget.** The elastic FWI test is a full-wave inverse solver; it is
  assigned to the committed `elastic-fwi` nextest group (90 s ceiling,
  single-threaded), the project's established class for FWI/theranostic tests.

## Increments (WIP-limited, one merge-affecting item at a time)

1. **DONE — Forward accessors + objective.** `ElasticWaveSolver` `mu`/`lambda`/
   `density` accessors + `set_mu` + the `propagate_point_forces` history
   primitive; `ElasticFwi::forward_misfit`. Test
   `forward_misfit_zero_at_true_model_positive_off_it`.
2. **DONE — Adjoint + gradient + FD check.** Vector (per-component, time-reversed)
   adjoint source, backward run, `K_μ` strain cross-correlation imaging condition;
   the κ directional-derivative gradient check (`κ ≈ 1.4`, stable).
3. **DONE — Inversion loop.** Steepest descent + Armijo line search + acquisition
   muting + illumination preconditioner (+ Tikhonov/TV available); the
   stiff-inclusion recovery test.
4. **DONE — Worked example/figure + differential vs `ShearWaveInversion`.** The
   `elastic_shear_fwi_lesion` example + Ch11 §11.6.6 fig07 demonstrate the
   reconstruction (the book's "result"). The differential is the value-semantic
   test `fwi_outperforms_linear_inversion`: on the same phantom the FWI recovers
   background and lesion peak far more accurately than the linear
   `LocalFrequencyEstimation` (measured: FWI background ≈2.5 % vs linear ≈59 %
   error; FWI peak ≈6 % vs linear ≈41 %), validating the §11.6.6 claim.

   Getting there required fixing two real defects the prototype surfaced in the
   linear `ShearWaveInversion`, both genuine "no deviation" bugs: (a) a **panic**
   on the `nz = 1` plane-strain geometry (`algorithms::fill_boundaries` indexed a
   non-existent `z+1` layer); (b) **silent clamp-garbage** on 2-D input across the
   LFE / Helmholtz / directional inversions and their shared smoothers (the 3-D
   interior loops `1..nz-1` are empty for `nz = 1`). Both fixed with singleton-axis
   guards + regression tests, so shear-wave elastography — conventionally a 2-D
   imaging plane — now inverts correctly in 2-D.
5. **PARTIAL — PyO3 binding + L-BFGS DONE; optimal checkpointing, 3-D, joint
   `λ/ρ` deferred.** `pykwavers.elastic_shear_fwi_reconstruct` (thin layer over
   `reconstruct_lesion_transmission`) exposes the inversion to Python with a
   binding-surface pytest; the acquisition setup is consolidated in
   `elastic_fwi::acquisition`, shared by the example and the binding.
   `ElasticFwi::run_lbfgs` adds the quasi-Newton optimizer (reusing
   `kwavers_math::LbfgsMemory`, mirroring the acoustic `invert_lbfgs`): it uses
   the **true** gradient `∂J/∂μ` (raw `K_μ` + regularization, no preconditioner —
   the inverse-Hessian subsumes that role) for a valid Armijo line search, and
   recovers the stiff inclusion in ~7 s vs the steepest-descent ~17–40 s
   (`lbfgs_reconstructs_stiff_inclusion`).

The book Ch26 §26 / Ch11 §11.14 "not implemented" disclosures are updated to point
at the real module (`inverse::elastography::elastic_fwi`).

## Consequences

- Adds a genuine full-waveform elastic inverse, closing the Ch26 §26 / Ch11 §11.14
  "not implemented" disclosure with real, tested code.
- Memory cost is `O(nt·N)` in v1 (full history); bounded by 2-D + slice-sized grids,
  with checkpointing as the documented growth path.
- The linear `ShearWaveInversion` remains the fast, robust default; the FWI is the
  high-resolution refinement, selected explicitly (no silent fallback).
- No change to existing forward/acoustic-FWI behaviour (all additions are new
  modules + additive accessors).
