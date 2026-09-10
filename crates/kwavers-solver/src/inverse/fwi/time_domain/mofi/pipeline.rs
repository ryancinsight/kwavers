//! The calibrated end-to-end alignments callers actually invoke.

use super::super::{geometry::FwiGeometry, FwiEngine, FwiProcessor};
use super::manifold::manifold_update;
use super::search::{optimize_speed_scale, scale_contrast};
use super::transform::{
    project_gradient, transform_template, transform_with_jacobian, PlaneGeometry,
};
use super::*;
use kwavers_core::error::{KwaversError, KwaversResult, ValidationError};
use kwavers_grid::Grid;
use leto::{Array2, Array3};

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
    let (grid_nx, grid_ny, grid_nz) = grid.dimensions();
    if template.shape() != [grid_nx, grid_ny, grid_nz] {
        return Err(KwaversError::Validation(
            ValidationError::ConstraintViolation {
                message: format!(
                    "MOFI template shape {:?} must match grid {:?}",
                    template.shape(),
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
