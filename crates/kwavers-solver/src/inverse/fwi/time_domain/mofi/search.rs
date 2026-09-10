//! Coarse pose search and the speed-scale refinement beside it.

use super::super::{geometry::FwiGeometry, FwiEngine, FwiProcessor};
use super::transform::{transform_template, PlaneGeometry};
use super::*;
use crate::inverse::reconstruction::seismic::MisfitType;
use kwavers_core::error::{KwaversError, KwaversResult};
use kwavers_grid::Grid;
use leto::{Array2, Array3};
use moirai_parallel::{map_collect_with, Adaptive};

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
/// - Returns [`crate::KwaversError::InvalidInput`] off the self-adjoint engine;
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
        Vec::with_capacity((thetas.len()) * (deltas.len()) * (deltas.len()));
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
/// Calibrated model `c = c_bg + α·(c_template − c_bg)` (affine contrast scaling).
pub(super) fn scale_contrast(template: &Array3<f64>, alpha: f64, background: f64) -> Array3<f64> {
    template.mapv(|c| background + alpha * (c - background))
}
/// One-dimensional optimisation of the contrast scale `α` for a fixed pose.
///
/// With the pose fixed, the transformed template `c_φ` is fixed and the model is
/// affine in `α`: `c(α) = c_bg + α·(c_φ − c_bg)`. The forward map is nonlinear in
/// `c`, so `f(α)` is minimised by a 1-D Armijo line search using
/// `∂f/∂α = ⟨∂f/∂c, c_φ − c_bg⟩` (the exact self-adjoint gradient).
#[allow(clippy::too_many_arguments)]
pub(super) fn optimize_speed_scale(
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
        let g_alpha = (&g * &contrast).iter().sum::<f64>();
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
