# ADR 033: Elastic / shear-wave FWI for lesion-stiffness reconstruction

- Status: Proposed
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

## Verification plan (evidence tiers)

1. **Exact gradient (type/test tier)** — a finite-difference gradient check
   `κ = (g·δμ)/(ΔJ) ≈ 1` per the acoustic `self_adjoint_gradient_matches_finite_difference_kappa_one`
   template, for ≥3 independent μ-perturbation directions, with Richardson-extrapolated
   central differences. Because the PML+Verlet adjoint is **not** guaranteed discretely
   exact, the acceptance is: descent-direction validity (`g·δμ < 0` aligned with `ΔJ`)
   with `κ` stable across directions; an exact `κ≈1` is a stretch goal contingent on a
   matched discrete adjoint of the Verlet stepper.
2. **Recovery (empirical tier)** — recover a known stiff cylindrical inclusion
   (`μ_lesion = 3·μ_background`) in a homogeneous SWE slab from synthetic ARFI
   shear-wave data; assert the reconstructed `μ` map has Pearson `r ≥ 0.9` and the
   inclusion Dice `≥ 0.7` vs ground truth (mirroring the Ch31 reconstruction-quality
   gates).
3. **Differential (empirical tier)** — on a smooth `c_S` ramp where the linear
   `ShearWaveInversion` is valid, the elastic FWI must agree with it within the SWE
   linear-regime tolerance; on a sharp inclusion the FWI must outperform it
   (sharper edge, smaller bias), demonstrating the value over the linear baseline.

## Increments (WIP-limited, one merge-affecting item at a time)

1. **Forward accessors + objective** — `ElasticWaveSolver` μ/λ/ρ accessors +
   displacement sensor recording; `ElasticFwi::forward_misfit` (forward run → L2
   objective vs observed). Test: objective is 0 at the true model, > 0 off it.
2. **Adjoint + gradient + FD check** — vector adjoint source, backward run, `K_μ`
   imaging condition; the κ FD gradient check. (The correctness anchor.)
3. **Inversion loop** — descent + Armijo line search + Tikhonov/TV; the
   known-inclusion recovery test.
4. **Differential vs `ShearWaveInversion`** + the Ch26/Ch11 doc update (mark the
   capability implemented; add a worked example / figure).
5. (Deferred) PyO3 binding `run_elastic_fwi_*`, checkpointing, 3-D, joint `λ/ρ`.

## Consequences

- Adds a genuine full-waveform elastic inverse, closing the Ch26 §26 / Ch11 §11.14
  "not implemented" disclosure with real, tested code.
- Memory cost is `O(nt·N)` in v1 (full history); bounded by 2-D + slice-sized grids,
  with checkpointing as the documented growth path.
- The linear `ShearWaveInversion` remains the fast, robust default; the FWI is the
  high-resolution refinement, selected explicitly (no silent fallback).
- No change to existing forward/acoustic-FWI behaviour (all additions are new
  modules + additive accessors).
