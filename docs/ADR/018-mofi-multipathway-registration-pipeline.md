# ADR 018 — Multi-pathway skull-registration pipeline (beyond rigid MOFI)

**Status:** Accepted
**Change class:** [minor] (additive: new drivers + public API; no breaking changes)
**Date:** 2026-06-07
**Builds on:** ADR 016 (exact self-adjoint `∂f/∂c`), ADR 017 (rigid MOFI)

## Context

Rigid MOFI (ADR 017) recovers a 3-DOF SE(2) pose, but leaves four practical gaps: it cannot fix
template **shape** error, cannot correct the template's systematic **sound-speed** (CT→speed) error,
**cycle-skips** under large misalignment on a single waveform misfit, and needs a reasonable
**initial pose**. Each gap is covered by a complementary ultrasound-processing pathway already
present (or cheaply expressible) in kwavers.

## Decision

Add four composable pathways and a pipeline that chains them, all sharing the exact self-adjoint
`∂f/∂c` (ADR 016):

1. **Misfit homotopy** (`align_homotopy`, `MofiStage`, `default_homotopy`). Warm-started anneal of
   the data misfit Wasserstein → envelope → L2 (reusing the wired `MisfitFunction` set). Widens the
   capture basin for large pose error. MOFI already routes its objective through `MisfitFunction`, so
   this is the cloned-processor-per-stage outer loop over `align_from`.
2. **Coarse global pose initializer** (`coarse_pose_search`, `CoarseSearchConfig`). Brute-force grid
   search over `(θ, δ)` evaluated by cheap sensor-only forwards — global sampling, so it cannot
   cycle-skip — seeding `align_from`. **Must use an arrival-time-sensitive, cycle-skip-robust misfit
   (Wasserstein).** Empirically, the phase-blind *envelope* misfit is rotation-insensitive and seeds
   the wrong angle; OT is correct. (Captured as a regression test insight.)
3. **Joint pose + sound-speed calibration** (`align_with_calibration`, `MofiCalibratedResult`).
   Block-coordinate descent alternating rigid pose alignment and a 1-D optimisation of a global
   contrast scale `α` (model `c = c_bg + α·(c_template − c_bg)`, `∂f/∂α = ⟨∂f/∂c, c_φ − c_bg⟩`).
   Corrects the CT→speed systematic error rigid alignment cannot.
4. **Non-rigid FFD** (`align_nonrigid`, `FfdConfig`, `FfdField`; `nonrigid.rs`). A coarse control
   lattice of displacements, bilinearly interpolated to a dense warp `c_u(x) = c_template(x − u(x))`,
   optimised against the acoustic misfit via the chained gradient
   `∂f/∂u_cp = −Σ_x g(x)·∇c·w_cp(x)` with a bending-energy smoothness penalty. Captures residual
   shape mismatch the rigid stages cannot.

**Pipeline** (`align_pipeline`, `PipelineConfig`, `PipelineResult`): coarse search (robust misfit) →
rigid + calibration (warm-started) → non-rigid FFD on the aligned, calibrated template.

## Alternatives considered

- **Standing up full travel-time tomography** (`sound_speed_shift`/`real_time_sirt`) as the
  initializer. Heavier integration; the coarse-search + robust-misfit pathway delivers the same
  "global, cycle-skip-free pose seed" self-contained. TT-tomography image-to-image init remains a
  future option.
- **Joint 4-parameter (θ, δ, α) gradient optimisation** instead of block coordinate. Mixing radians,
  metres, and a dimensionless scale in one normalised gradient is unit-fragile; block coordinate is
  robust and reuses `align_from` for the pose sub-problem.
- **Cubic B-spline FFD**. Bilinear control-lattice interpolation is simpler, has a clean analytic
  Jacobian, and suffices for the smooth deformations targeted; cubic B-splines are a future upgrade.

## Verification

- `homotopy_recovers_large_rotation` — 28° recovered to <1.5°/<1 mm; never worse than plain L2.
- `coarse_initializer_rescues_far_misalignment` — 45° recovered after coarse (Wasserstein) seed,
  where local-from-identity MOFI lands ~22° off.
- `calibration_recovers_pose_and_speed_error` — `α = 1.25` and pose recovered; beats rigid-only misfit.
- `nonrigid_ffd_recovers_smooth_deformation` — recovers a non-rigid warp (misfit ↓, displacement RMS
  < 1 mm vs ~2.5 mm peak).
- `full_pipeline_recovers_compound_misalignment` — pose + speed + warp compound case: aligned model
  matches truth in the illuminated region (RMS Δc < 40 m/s) and `α` recovered.

## Consequences / scope

- 2-D, single-acquisition (as ADR 017). Each pathway is independently usable and composable.
- The rigid pose under a compound non-rigid warp is **not** unique (pose/FFD trade off); the
  pipeline's quality metric is the aligned **model**, not pose-vs-truth.
- New public API on `inverse::fwi::time_domain`: `mofi_align_homotopy`, `mofi_coarse_pose_search`,
  `mofi_align_with_calibration`, `mofi_align_nonrigid`, `mofi_align_pipeline` (+ config/result types).
