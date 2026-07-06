//! Manifold Optimisation for FWI (MOFI): guidance-free rigid alignment of a
//! sound-speed template to acoustic data (Bates et al., *Ultrasound in Medicine
//! & Biology*, 2026, "Automatic Skull-Template Alignment Without a Guidance
//! Image").
//!
//! # Idea
//! Transcranial FWI needs a CT-derived skull template aligned to the patient.
//! Standard FWI updates the model `c` pixel-wise; MOFI instead reparametrises the
//! model as a **rigid-body (SE(2)) transform** of the template,
//! `φ = {θ, δ₁, δ₂}`, and minimises the *acoustic* misfit over just those three
//! parameters — no MRI guidance image. The chained gradient is
//! ```text
//! ∂f/∂φ = (∂c_φ/∂φ)ᵀ ∂f/∂c
//! ```
//! where `∂f/∂c` is the standard FWI adjoint-state gradient and `∂c_φ/∂φ` is the
//! analytic Jacobian of the rigid reparametrisation ([`transform`]).
//!
//! # Manifold update (paper Appendix A)
//! Updates respect the SE(2) Lie-group geometry: the rotation is updated through
//! the SO(2) log/exp maps (which keeps `θ ∈ [−π, π]` and follows the shortest
//! geodesic), and the translation increment is rotated by the current rotation
//! before being added, `δ^{k+1} = δ^k + R_{θ^k} Δ_δ`. Optimisation uses
//! gradient normalisation and an Armijo line search (the paper found explicit
//! line search most stable).
//!
//! # Exact gradient
//! MOFI's accuracy rests on a faithful `∂f/∂c`. This driver uses the self-adjoint
//! engine ([`super::FwiEngine::SecondOrderSelfAdjoint`], ADR 016), whose gradient
//! is the exact discrete `∂f/∂c` (`κ ≈ 1`); the processor passed to [`align`]
//! must select that engine.
//!
//! Parameters are balanced for the line search by optimising in the scaled space
//! `(L·θ, δ₁, δ₂)` where `L` is the domain half-width — a rotation moves the
//! template edge by `≈ L·dθ`, comparable to a translation `dδ`.

mod nonrigid;
mod transform;

#[cfg(test)]
mod tests;

pub use nonrigid::{align_nonrigid, sample_displacement, FfdBasis, FfdConfig, FfdField, FfdResult};
pub use transform::RigidTransform;

use super::{geometry::FwiGeometry, FwiEngine, FwiProcessor};
use crate::inverse::reconstruction::seismic::MisfitType;
use kwavers_core::error::{KwaversError, KwaversResult, ValidationError};
use kwavers_grid::Grid;
use moirai_parallel::{map_collect_with, Adaptive};
use ndarray::{Array2, Array3};
use transform::{project_gradient, transform_template, transform_with_jacobian, PlaneGeometry};

/// MOFI optimisation settings.
#[derive(Debug, Clone, Copy)]
pub struct MofiConfig {
    /// Maximum outer iterations.
    pub max_iterations: usize,
    /// Initial line-search step in the balanced parameter space [m].
    pub initial_step_m: f64,
    /// Armijo sufficient-decrease constant `c₁ ∈ (0, 1)`.
    pub armijo_c1: f64,
    /// Maximum Armijo backtracking halvings per iteration.
    pub max_line_search: usize,
    /// Sound speed assigned where the transformed template maps outside the grid
    /// (the background medium, e.g. water) [m/s].
    pub background_c: f64,
    /// Relative-misfit-change convergence tolerance.
    pub tolerance: f64,
}

impl Default for MofiConfig {
    fn default() -> Self {
        Self {
            max_iterations: 60,
            initial_step_m: 5e-3,
            armijo_c1: 1e-4,
            max_line_search: 15,
            background_c: 1500.0,
            tolerance: 1e-4,
        }
    }
}

/// Result of a MOFI alignment.
#[derive(Debug, Clone, Copy)]
pub struct MofiResult {
    /// Recovered rigid transform aligning the template to the data.
    pub transform: RigidTransform,
    /// Outer iterations performed.
    pub iterations: usize,
    /// Misfit at `φ = 0` (template untransformed).
    pub initial_misfit: f64,
    /// Misfit at the recovered transform.
    pub final_misfit: f64,
}

/// Apply a rigid SE(2) transform to a model field on `grid`.
///
/// `c_φ(x) = model(T_φ⁻¹ x)` by bilinear resampling about the grid centre;
/// out-of-domain points take `background_c`. This is the same reparametrisation
/// MOFI optimises over ([`align`]); exposed for building misaligned synthetics and
/// for applying a recovered [`RigidTransform`].
///
/// # Example
/// ```no_run
/// use kwavers_solver::inverse::fwi::time_domain::{mofi_transform, RigidTransform};
/// use kwavers_grid::Grid;
/// use ndarray::Array3;
/// let grid = Grid::new(32, 32, 1, 1e-3, 1e-3, 1e-3).unwrap();
/// let template = Array3::from_elem((32, 32, 1), 1500.0);
/// let phi = RigidTransform { theta_rad: 6_f64.to_radians(), delta_x_m: 2e-3, delta_y_m: -1e-3 };
/// let misaligned = mofi_transform(&template, &phi, &grid, 1500.0);
/// assert_eq!(misaligned.dim(), template.dim());
/// ```
#[must_use]
pub fn transform(
    model: &Array3<f64>,
    phi: &RigidTransform,
    grid: &Grid,
    background_c: f64,
) -> Array3<f64> {
    let (nx, ny, _) = grid.dimensions();
    let geom = PlaneGeometry::centered(nx, ny, grid.dx, grid.dy);
    transform_template(model, phi, &geom, background_c)
}

/// SE(2) manifold update (paper Appendix A) in the balanced scaled space.
///
/// `dir_scaled` is a descent direction in `(L·θ, δ₁, δ₂)` space; `step` is the
/// arc length [m]. The rotation increment is `dθ = step·dir₀ / L` (wrapped to
/// `[−π, π]` via the log/exp maps), and the translation increment
/// `(step·dir₁, step·dir₂)` is rotated by the current rotation `R_{θ}`.
fn manifold_update(
    phi: &RigidTransform,
    dir_scaled: [f64; 3],
    step: f64,
    length_scale: f64,
) -> RigidTransform {
    let dtheta = step * dir_scaled[0] / length_scale;
    let theta_sum = phi.theta_rad + dtheta;
    // SO(2) log∘exp = wrap to (−π, π], the shortest-geodesic representative.
    let theta_new = theta_sum.sin().atan2(theta_sum.cos());

    let sx = step * dir_scaled[1];
    let sy = step * dir_scaled[2];
    let (s, c) = phi.theta_rad.sin_cos();
    RigidTransform {
        theta_rad: theta_new,
        delta_x_m: phi.delta_x_m + (c * sx - s * sy),
        delta_y_m: phi.delta_y_m + (s * sx + c * sy),
    }
}

/// One stage of a MOFI misfit homotopy.
///
/// Each stage runs [`align_from`] with the data misfit set to `misfit_type` and
/// an optional zero-phase low-pass corner `band_limit_hz`, warm-started from the
/// previous stage's transform. Annealing from a convex, cycle-skip-robust misfit
/// (Wasserstein/optimal transport, envelope, correlation) toward L2 widens the
/// capture basin for large initial misalignments.
#[derive(Debug, Clone, Copy)]
pub struct MofiStage {
    /// Data misfit functional for this stage.
    pub misfit_type: MisfitType,
    /// Optional low-pass corner [Hz] applied to traces this stage (multiscale).
    pub band_limit_hz: Option<f64>,
    /// Optimisation settings for this stage.
    pub config: MofiConfig,
}

/// A robust-to-precise default homotopy: Wasserstein → envelope → L2.
///
/// Wasserstein (optimal transport) is convex in time shifts (good for large
/// pose error), envelope removes carrier oscillation, and L2 sharpens the final
/// fit. `config` is reused for every stage.
#[must_use]
pub fn default_homotopy(config: MofiConfig) -> [MofiStage; 3] {
    [
        MofiStage {
            misfit_type: MisfitType::Wasserstein,
            band_limit_hz: None,
            config,
        },
        MofiStage {
            misfit_type: MisfitType::Envelope,
            band_limit_hz: None,
            config,
        },
        MofiStage {
            misfit_type: MisfitType::L2Norm,
            band_limit_hz: None,
            config,
        },
    ]
}

/// Align by a warm-started misfit homotopy: run each stage in turn, threading the
/// recovered transform forward.
///
/// `processor` selects the engine (must be [`FwiEngine::SecondOrderSelfAdjoint`])
/// and the *base* configuration; per stage it is cloned with the stage's misfit
/// and band-limit. Returns the final stage's result, with `initial_misfit` taken
/// from the first stage (at `φ = 0`).
/// # Errors
/// - As [`align_from`]; also fails if `stages` is empty.
pub fn align_homotopy(
    processor: &FwiProcessor,
    template: &Array3<f64>,
    observed: &Array2<f64>,
    geometry: &FwiGeometry,
    grid: &Grid,
    stages: &[MofiStage],
) -> KwaversResult<MofiResult> {
    if stages.is_empty() {
        return Err(KwaversError::InvalidInput(
            "MOFI homotopy requires at least one stage".to_owned(),
        ));
    }
    let mut phi = RigidTransform::identity();
    let mut initial_misfit = None;
    let mut last = None;
    for stage in stages {
        let staged = processor
            .clone()
            .with_misfit(stage.misfit_type)
            .with_band_limit(stage.band_limit_hz);
        let result = align_from(
            &staged,
            template,
            observed,
            geometry,
            grid,
            &stage.config,
            phi,
        )?;
        phi = result.transform;
        initial_misfit.get_or_insert(result.initial_misfit);
        last = Some(result);
    }
    let mut result = last.expect("non-empty stages produce a result");
    result.initial_misfit = initial_misfit.expect("first stage sets initial misfit");
    Ok(result)
}

/// Align `template` to `observed` acoustic data by rigid (SE(2)) manifold
/// optimisation of the FWI misfit, starting from the identity transform.
///
/// `processor` must select [`FwiEngine::SecondOrderSelfAdjoint`] so the chained
/// gradient uses the exact `∂f/∂c`. `template` is the reference sound-speed image
/// on the inversion grid; `observed` is the recorded data (recorder/Fortran row
/// order, as produced by the engine's forward model).
/// # Errors
/// - Returns [`KwaversError::InvalidInput`] if the processor is not on the
///   self-adjoint engine, or [`KwaversError::Validation`] on shape/geometry
///   problems; propagates forward/adjoint solve errors.
pub fn align(
    processor: &FwiProcessor,
    template: &Array3<f64>,
    observed: &Array2<f64>,
    geometry: &FwiGeometry,
    grid: &Grid,
    config: &MofiConfig,
) -> KwaversResult<MofiResult> {
    align_from(
        processor,
        template,
        observed,
        geometry,
        grid,
        config,
        RigidTransform::identity(),
    )
}

/// Coarse global pose-search settings.
///
/// The search brute-forces a regular grid over rotation and translation,
/// evaluating the data misfit at each candidate by a cheap sensor-only forward
/// solve. Being a global sample (not a gradient step), it is robust to arbitrarily
/// large misalignment and provides a starting pose inside the local basin for
/// [`align_from`] / [`align_homotopy`].
#[derive(Debug, Clone, Copy)]
pub struct CoarseSearchConfig {
    /// Half-range of the rotation sweep [rad]; candidates span `[−θ_max, θ_max]`.
    pub theta_max_rad: f64,
    /// Number of rotation samples (≥ 1).
    pub theta_steps: usize,
    /// Half-range of each translation axis [m]; candidates span `[−δ_max, δ_max]`.
    pub delta_max_m: f64,
    /// Number of samples per translation axis (≥ 1).
    pub delta_steps: usize,
    /// Background sound speed for out-of-domain template pixels [m/s].
    pub background_c: f64,
}

/// The recommended data misfit for [`coarse_pose_search`] and the first
/// homotopy stage: an arrival-time-sensitive, cycle-skip-robust functional.
///
/// **Pitfall:** envelope- and phase-only misfits are robust to cycle-skipping but
/// *phase-blind*, so they poorly constrain **rotation** — a coarse search driven
/// by the envelope misfit seeds the wrong angle (empirically θ ≈ 60° for a true
/// 45°). [`MisfitType::Wasserstein`] (optimal transport) is convex in time shifts
/// *and* sensitive to arrival times, so its global minimum tracks the true pose.
#[must_use]
pub fn recommended_search_misfit() -> MisfitType {
    MisfitType::Wasserstein
}

fn linspace(half_range: f64, steps: usize) -> Vec<f64> {
    let n = steps.max(1);
    if n == 1 {
        return vec![0.0];
    }
    (0..n)
        .map(|i| -half_range + 2.0 * half_range * (i as f64) / ((n - 1) as f64))
        .collect()
}

/// Brute-force global search for the best coarse rigid pose.
///
/// Returns the `(transform, misfit)` minimising the configured data misfit over
/// the search grid. Pure global sampling — no gradient — so it cannot cycle-skip;
/// use it to initialise [`align_from`] when the misalignment may be large.
/// # Errors
/// - Returns [`KwaversError::InvalidInput`] off the self-adjoint engine;
///   propagates solve errors.
pub fn coarse_pose_search(
    processor: &FwiProcessor,
    template: &Array3<f64>,
    observed: &Array2<f64>,
    geometry: &FwiGeometry,
    grid: &Grid,
    search: &CoarseSearchConfig,
) -> KwaversResult<(RigidTransform, f64)> {
    if processor.engine != FwiEngine::SecondOrderSelfAdjoint {
        return Err(KwaversError::InvalidInput(
            "MOFI coarse search requires FwiEngine::SecondOrderSelfAdjoint".to_owned(),
        ));
    }
    let (nx, ny, _) = grid.dimensions();
    let geom = PlaneGeometry::centered(nx, ny, grid.dx, grid.dy);
    let thetas = linspace(search.theta_max_rad, search.theta_steps);
    let deltas = linspace(search.delta_max_m, search.delta_steps);

    // Enumerate the (θ, δ₁, δ₂) grid; each pose's misfit is an independent
    // sensor-only forward solve (~55 MB/call, documented parallel-safe), so the
    // search runs through the Atlas execution provider. Ordered collection
    // preserves grid order, so reducing with a strict `<` reproduces the serial
    // first-minimum tie-break exactly.
    let mut candidates: Vec<RigidTransform> =
        Vec::with_capacity(thetas.len() * deltas.len() * deltas.len());
    for &theta in &thetas {
        for &dx in &deltas {
            for &dy in &deltas {
                candidates.push(RigidTransform {
                    theta_rad: theta,
                    delta_x_m: dx,
                    delta_y_m: dy,
                });
            }
        }
    }
    let misfits: Vec<KwaversResult<f64>> =
        map_collect_with::<Adaptive, _, _, _>(&candidates, |phi| {
            let model = transform_template(template, phi, &geom, search.background_c);
            let synth = processor.forward_model_sensor_only(&model, geometry, grid)?;
            processor.compute_misfit_objective(observed, &synth)
        });

    let mut best = RigidTransform::identity();
    let mut best_misfit = f64::INFINITY;
    for (phi, misfit) in candidates.iter().zip(misfits) {
        let misfit = misfit?;
        if misfit < best_misfit {
            best_misfit = misfit;
            best = *phi;
        }
    }
    Ok((best, best_misfit))
}

/// Result of a pose + sound-speed-calibration alignment.
#[derive(Debug, Clone, Copy)]
pub struct MofiCalibratedResult {
    /// Recovered rigid transform.
    pub transform: RigidTransform,
    /// Recovered global template sound-speed contrast scale `α`: the calibrated
    /// model is `c = c_bg + α·(c_template − c_bg)` (α = 1 leaves the template
    /// unchanged). Corrects a systematic CT→sound-speed mapping error that rigid
    /// alignment alone cannot.
    pub speed_scale: f64,
    /// Misfit at `φ = 0, α = 1`.
    pub initial_misfit: f64,
    /// Misfit at the recovered `(φ, α)`.
    pub final_misfit: f64,
    /// Outer block-coordinate iterations performed.
    pub outer_iterations: usize,
}

/// Calibrated model `c = c_bg + α·(c_template − c_bg)` (affine contrast scaling).
fn scale_contrast(template: &Array3<f64>, alpha: f64, background: f64) -> Array3<f64> {
    template.mapv(|c| background + alpha * (c - background))
}

/// One-dimensional optimisation of the contrast scale `α` for a fixed pose.
///
/// With the pose fixed, the transformed template `c_φ` is fixed and the model is
/// affine in `α`: `c(α) = c_bg + α·(c_φ − c_bg)`. The forward map is nonlinear in
/// `c`, so `f(α)` is minimised by a 1-D Armijo line search using
/// `∂f/∂α = ⟨∂f/∂c, c_φ − c_bg⟩` (the exact self-adjoint gradient).
#[allow(clippy::too_many_arguments)]
fn optimize_speed_scale(
    processor: &FwiProcessor,
    template: &Array3<f64>,
    pose: RigidTransform,
    observed: &Array2<f64>,
    geom: &PlaneGeometry,
    geometry: &FwiGeometry,
    grid: &Grid,
    alpha0: f64,
    config: &MofiConfig,
) -> KwaversResult<(f64, f64)> {
    let bg = config.background_c;
    let c_phi = transform_template(template, &pose, geom, bg);
    let contrast = c_phi.mapv(|c| c - bg); // c_φ − c_bg (fixed for this pose).
    let model_of = |a: f64| contrast.mapv(|d| bg + a * d);

    let mut alpha = alpha0;
    let synth0 = processor.forward_model_sensor_only(&model_of(alpha), geometry, grid)?;
    let mut f = processor.compute_misfit_objective(observed, &synth0)?;
    let step0 = 0.2_f64; // α is O(1) and dimensionless.

    for _ in 0..10 {
        let model = model_of(alpha);
        let (synth, history) = processor.forward_model(&model, geometry, grid)?;
        f = processor.compute_misfit_objective(observed, &synth)?;
        let residual = processor.compute_adjoint_source(observed, &synth)?;
        let g = processor
            .adjoint_gradient_self_adjoint(&residual, &model, geometry, grid, &history, None)?;
        let g_alpha = (&g * &contrast).sum();
        if g_alpha.abs() <= f64::MIN_POSITIVE {
            break;
        }
        let dir = -g_alpha.signum();
        let gd = g_alpha * dir; // = −|g_alpha| < 0.
        let mut step = step0;
        let mut accepted: Option<(f64, f64)> = None;
        for _ in 0..config.max_line_search {
            let a_trial = (alpha + step * dir).max(1e-3);
            let synth_t =
                processor.forward_model_sensor_only(&model_of(a_trial), geometry, grid)?;
            let f_trial = processor.compute_misfit_objective(observed, &synth_t)?;
            if f_trial <= f + config.armijo_c1 * step * gd {
                accepted = Some((a_trial, f_trial));
                break;
            }
            step *= 0.5;
        }
        let Some((a_new, f_new)) = accepted else {
            break;
        };
        let rel = (f - f_new).abs() / f.max(f64::EPSILON);
        alpha = a_new;
        f = f_new;
        if rel < config.tolerance {
            break;
        }
    }
    Ok((alpha, f))
}

/// Joint pose + sound-speed-calibration alignment by block-coordinate descent.
///
/// Alternates (a) rigid pose alignment of the α-scaled template via [`align_from`]
/// and (b) a 1-D optimisation of the contrast scale `α` at the fixed pose, for
/// `outer_iterations` rounds. This corrects both the template's position/orientation
/// and a systematic over/under-estimate of its sound-speed contrast — the latter a
/// known CT→speed error that pure rigid alignment cannot fix.
/// # Errors
/// - As [`align_from`].
// Driver entry point: the acquisition (processor/template/observed/geometry/grid),
// optimisation config, outer-iteration count, and warm-start pose are all
// independent inputs; bundling them would obscure the call site.
#[allow(clippy::too_many_arguments)]
pub fn align_with_calibration(
    processor: &FwiProcessor,
    template: &Array3<f64>,
    observed: &Array2<f64>,
    geometry: &FwiGeometry,
    grid: &Grid,
    config: &MofiConfig,
    outer_iterations: usize,
    initial: RigidTransform,
) -> KwaversResult<MofiCalibratedResult> {
    if processor.engine != FwiEngine::SecondOrderSelfAdjoint {
        return Err(KwaversError::InvalidInput(
            "MOFI calibration requires FwiEngine::SecondOrderSelfAdjoint".to_owned(),
        ));
    }
    let (nx, ny, _) = grid.dimensions();
    let geom = PlaneGeometry::centered(nx, ny, grid.dx, grid.dy);
    let bg = config.background_c;

    let baseline = processor.forward_model_sensor_only(template, geometry, grid)?;
    let initial_misfit = processor.compute_misfit_objective(observed, &baseline)?;

    let mut pose = initial;
    let mut alpha = 1.0_f64;
    let mut final_misfit = initial_misfit;
    let mut outer = 0usize;
    for _ in 0..outer_iterations.max(1) {
        let cal_template = scale_contrast(template, alpha, bg);
        let pose_res = align_from(
            processor,
            &cal_template,
            observed,
            geometry,
            grid,
            config,
            pose,
        )?;
        pose = pose_res.transform;
        let (alpha_new, f) = optimize_speed_scale(
            processor, template, pose, observed, &geom, geometry, grid, alpha, config,
        )?;
        alpha = alpha_new;
        final_misfit = f;
        outer += 1;
    }

    Ok(MofiCalibratedResult {
        transform: pose,
        speed_scale: alpha,
        initial_misfit,
        final_misfit,
        outer_iterations: outer,
    })
}

/// Full multi-pathway alignment pipeline configuration.
#[derive(Debug, Clone, Copy)]
pub struct PipelineConfig {
    /// Coarse global pose-search grid.
    pub coarse: CoarseSearchConfig,
    /// Misfit used for the coarse search (use an arrival-time-sensitive, cycle-
    /// skip-robust functional such as [`MisfitType::Wasserstein`]).
    pub search_misfit: MisfitType,
    /// Rigid/calibration optimisation settings.
    pub rigid: MofiConfig,
    /// Block-coordinate rounds for pose + speed calibration (0 ⇒ skip calibration,
    /// rigid pose only).
    pub calibration_outer: usize,
    /// Non-rigid FFD refinement settings (`n_ctrl_* < 2` ⇒ skip non-rigid).
    pub ffd: FfdConfig,
}

/// Result of the full alignment pipeline.
#[derive(Debug, Clone)]
pub struct PipelineResult {
    /// Recovered rigid pose.
    pub transform: RigidTransform,
    /// Recovered sound-speed contrast scale (1.0 if calibration skipped).
    pub speed_scale: f64,
    /// Recovered non-rigid deformation (zero lattice if non-rigid skipped).
    pub ffd: FfdField,
    /// Misfit at the unaligned template.
    pub initial_misfit: f64,
    /// Misfit at the fully aligned model.
    pub final_misfit: f64,
}

/// Full multi-pathway skull-alignment pipeline:
/// **coarse global pose search → rigid + sound-speed calibration → non-rigid FFD.**
///
/// Each stage covers the previous stage's blind spot: the robust-misfit global
/// search defeats large-misalignment cycle-skipping and supplies an initial pose;
/// the rigid + calibration stage recovers pose and corrects the template's
/// systematic speed error; the non-rigid stage captures residual shape mismatch.
/// All stages share the exact self-adjoint `∂f/∂c` (ADR 016).
/// # Errors
/// - As the constituent stages; requires [`FwiEngine::SecondOrderSelfAdjoint`].
pub fn align_pipeline(
    processor: &FwiProcessor,
    template: &Array3<f64>,
    observed: &Array2<f64>,
    geometry: &FwiGeometry,
    grid: &Grid,
    config: &PipelineConfig,
) -> KwaversResult<PipelineResult> {
    if processor.engine != FwiEngine::SecondOrderSelfAdjoint {
        return Err(KwaversError::InvalidInput(
            "MOFI pipeline requires FwiEngine::SecondOrderSelfAdjoint".to_owned(),
        ));
    }
    let (nx, ny, _) = grid.dimensions();
    let geom = PlaneGeometry::centered(nx, ny, grid.dx, grid.dy);
    let bg = config.rigid.background_c;

    // Stage 1 — coarse global pose seed via a robust, arrival-time-sensitive misfit.
    let search_proc = processor.clone().with_misfit(config.search_misfit);
    let (seed, _) = coarse_pose_search(
        &search_proc,
        template,
        observed,
        geometry,
        grid,
        &config.coarse,
    )?;

    // Stage 2 — rigid pose (+ optional sound-speed calibration), warm-started.
    let (pose, speed_scale, initial_misfit, rigid_misfit) = if config.calibration_outer == 0 {
        let r = align_from(
            processor,
            template,
            observed,
            geometry,
            grid,
            &config.rigid,
            seed,
        )?;
        (r.transform, 1.0, r.initial_misfit, r.final_misfit)
    } else {
        let r = align_with_calibration(
            processor,
            template,
            observed,
            geometry,
            grid,
            &config.rigid,
            config.calibration_outer,
            seed,
        )?;
        (r.transform, r.speed_scale, r.initial_misfit, r.final_misfit)
    };

    // Stage 3 — non-rigid FFD on the rigidly-aligned, calibrated template.
    if config.ffd.n_ctrl_x >= 2 && config.ffd.n_ctrl_y >= 2 {
        let scaled = scale_contrast(template, speed_scale, bg);
        let aligned = transform_template(&scaled, &pose, &geom, bg);
        let ffd =
            nonrigid::align_nonrigid(processor, &aligned, observed, geometry, grid, &config.ffd)?;
        Ok(PipelineResult {
            transform: pose,
            speed_scale,
            ffd: ffd.field,
            initial_misfit,
            final_misfit: ffd.final_misfit,
        })
    } else {
        Ok(PipelineResult {
            transform: pose,
            speed_scale,
            ffd: FfdField::zeros(2, 2, config.ffd.basis),
            initial_misfit,
            final_misfit: rigid_misfit,
        })
    }
}

/// Align starting from a supplied transform `initial` (warm start).
///
/// Identical to [`align`] but begins the optimisation at `initial` instead of the
/// identity — used by [`align_homotopy`] to thread successive misfit stages and by
/// the coarse-pose initializer.
/// # Errors
/// - As [`align`].
pub fn align_from(
    processor: &FwiProcessor,
    template: &Array3<f64>,
    observed: &Array2<f64>,
    geometry: &FwiGeometry,
    grid: &Grid,
    config: &MofiConfig,
    initial: RigidTransform,
) -> KwaversResult<MofiResult> {
    if processor.engine != FwiEngine::SecondOrderSelfAdjoint {
        return Err(KwaversError::InvalidInput(
            "MOFI requires FwiProcessor::with_engine(FwiEngine::SecondOrderSelfAdjoint) \
             so the chained gradient uses the exact ∂f/∂c"
                .to_owned(),
        ));
    }
    if template.dim() != grid.dimensions() {
        return Err(KwaversError::Validation(
            ValidationError::ConstraintViolation {
                message: format!(
                    "MOFI template shape {:?} must match grid {:?}",
                    template.dim(),
                    grid.dimensions()
                ),
            },
        ));
    }
    geometry.validate(grid, processor.parameters.nt)?;

    let (nx, ny, _) = grid.dimensions();
    let geom = PlaneGeometry::centered(nx, ny, grid.dx, grid.dy);
    let length_scale = 0.5 * ((nx as f64) * grid.dx).max((ny as f64) * grid.dy);
    let bg = config.background_c;

    let mut phi = initial;
    let initial_model = transform_template(template, &phi, &geom, bg);
    let initial_synth = processor.forward_model_sensor_only(&initial_model, geometry, grid)?;
    let initial_misfit = processor.compute_misfit_objective(observed, &initial_synth)?;
    let mut current_misfit = initial_misfit;
    let mut iterations = 0usize;

    for _ in 0..config.max_iterations {
        // Transform + analytic Jacobian at the current φ.
        let jac = transform_with_jacobian(template, &phi, &geom, bg);
        let (synth, history) = processor.forward_model(&jac.model, geometry, grid)?;
        current_misfit = processor.compute_misfit_objective(observed, &synth)?;
        let residual = processor.compute_adjoint_source(observed, &synth)?;
        // Exact ∂f/∂c at the current transformed model (self-adjoint engine).
        let g = processor
            .adjoint_gradient_self_adjoint(&residual, &jac.model, geometry, grid, &history, None)?;
        // Chain to the SE(2) parameters, then balance θ against δ.
        let g_phi = project_gradient(&g, &jac);
        let g_scaled = [g_phi[0] / length_scale, g_phi[1], g_phi[2]];
        let g_norm = (g_scaled[0].powi(2) + g_scaled[1].powi(2) + g_scaled[2].powi(2)).sqrt();
        if g_norm <= f64::MIN_POSITIVE {
            break;
        }
        let dir = [
            -g_scaled[0] / g_norm,
            -g_scaled[1] / g_norm,
            -g_scaled[2] / g_norm,
        ];
        let gd = -g_norm; // directional derivative gᵀd in scaled space (< 0).

        // Armijo backtracking line search.
        let mut step = config.initial_step_m;
        let mut accepted: Option<(RigidTransform, f64)> = None;
        for _ in 0..config.max_line_search {
            let trial = manifold_update(&phi, dir, step, length_scale);
            let trial_model = transform_template(template, &trial, &geom, bg);
            let trial_synth = processor.forward_model_sensor_only(&trial_model, geometry, grid)?;
            let trial_misfit = processor.compute_misfit_objective(observed, &trial_synth)?;
            if trial_misfit <= current_misfit + config.armijo_c1 * step * gd {
                accepted = Some((trial, trial_misfit));
                break;
            }
            step *= 0.5;
        }

        let Some((trial, trial_misfit)) = accepted else {
            break; // line search found no decrease → converged/stalled.
        };
        let rel_change = (current_misfit - trial_misfit).abs() / current_misfit.max(f64::EPSILON);
        phi = trial;
        current_misfit = trial_misfit;
        iterations += 1;
        if rel_change < config.tolerance {
            break;
        }
    }

    Ok(MofiResult {
        transform: phi,
        iterations,
        initial_misfit,
        final_misfit: current_misfit,
    })
}
