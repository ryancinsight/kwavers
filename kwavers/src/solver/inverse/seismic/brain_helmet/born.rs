//! Finite-frequency Born inversion for CT-derived brain speed contrast.

use crate::core::error::KwaversResult;
use ndarray::Array2;

use super::{
    conditioning::{
        apply_sobolev_preconditioner, continuation_rows, enhance_reconstruction,
        stage_iteration_count,
    },
    config::{BrainHelmetFwiConfig, C_BRAIN_REF_M_S},
    linear_algebra::{
        matrix_vector, migration_contrast, normal_equation_diagonal_rows, normalized_gradient_rows,
        objective, objective_rows,
    },
    metrics::{normalized_rmse, pearson, percentile_range},
    medium::AcousticSlice,
    sensitivity::build_sensitivity_matrix,
    transducer::HelmetHemisphereGeometry,
};

/// Quality metrics for the reconstructed brain image.
#[derive(Clone, Debug)]
pub struct BrainHelmetFwiMetrics {
    pub initial_objective: f64,
    pub final_objective: f64,
    pub objective_reduction_fraction: f64,
    pub migration_pearson_correlation: f64,
    pub migration_normalized_rmse: f64,
    pub migration_dynamic_range_m_s: f64,
    pub pearson_correlation: f64,
    pub normalized_rmse: f64,
    pub enhanced_dynamic_range_m_s: f64,
    pub active_voxels: usize,
    pub measurements: usize,
    pub continuation_stages: usize,
    pub target_dynamic_range_m_s: f64,
    pub reconstruction_dynamic_range_m_s: f64,
}

/// Result arrays and diagnostics from the encoded brain FWI run.
#[derive(Clone, Debug)]
pub struct BrainHelmetFwiResult {
    pub ct_hu: Array2<f64>,
    pub target_sound_speed_m_s: Array2<f64>,
    pub initial_sound_speed_m_s: Array2<f64>,
    pub migration_sound_speed_m_s: Array2<f64>,
    pub reconstruction_sound_speed_m_s: Array2<f64>,
    pub enhanced_reconstruction_sound_speed_m_s: Array2<f64>,
    pub brain_mask: Array2<bool>,
    pub skull_mask: Array2<bool>,
    pub synthetic_data: Vec<f64>,
    pub residual_history: Vec<f64>,
    pub metrics: BrainHelmetFwiMetrics,
}

#[derive(Clone, Debug)]
pub(super) struct ActiveVoxel {
    pub(super) ix: usize,
    pub(super) iy: usize,
    pub(super) x_m: f64,
    pub(super) y_m: f64,
    pub(super) z_m: f64,
    target_contrast: f64,
}

/// Reconstruct brain sound-speed contrast from encoded 1024-element data.
pub fn reconstruct_brain_slice(
    medium: &AcousticSlice,
    config: &BrainHelmetFwiConfig,
) -> KwaversResult<BrainHelmetFwiResult> {
    config.validate()?;
    let geometry = HelmetHemisphereGeometry::uniform(config.element_count, config.radius_m)?;
    let active = active_voxels(medium);
    let matrix = build_sensitivity_matrix(medium, config, &geometry, &active);
    let nrows = config.measurement_count();
    let data = matrix_vector(&matrix, nrows, active.len(), |j| active[j].target_contrast);
    let migration_model = migration_contrast(&matrix, &data, nrows, active.len(), config);
    let inversion = invert(
        &matrix,
        &data,
        nrows,
        active.len(),
        config,
        &active,
        medium.sound_speed_m_s.dim(),
    );

    let mut migration = medium.initial_sound_speed_m_s.clone();
    let mut reconstruction = medium.initial_sound_speed_m_s.clone();
    for (idx, voxel) in active.iter().enumerate() {
        migration[[voxel.ix, voxel.iy]] = C_BRAIN_REF_M_S * (1.0 + migration_model[idx]);
        reconstruction[[voxel.ix, voxel.iy]] = C_BRAIN_REF_M_S * (1.0 + inversion.model[idx]);
    }
    let enhanced = enhance_reconstruction(
        &reconstruction,
        &medium.brain_mask,
        config.enhancement_gain,
        C_BRAIN_REF_M_S,
    );

    let target: Vec<f64> = active
        .iter()
        .map(|v| C_BRAIN_REF_M_S * (1.0 + v.target_contrast))
        .collect();
    let recon: Vec<f64> = active
        .iter()
        .enumerate()
        .map(|(idx, _)| C_BRAIN_REF_M_S * (1.0 + inversion.model[idx]))
        .collect();
    let migration_values: Vec<f64> = active
        .iter()
        .enumerate()
        .map(|(idx, _)| C_BRAIN_REF_M_S * (1.0 + migration_model[idx]))
        .collect();
    let enhanced_values: Vec<f64> = active
        .iter()
        .map(|voxel| enhanced[[voxel.ix, voxel.iy]])
        .collect();

    let initial_objective = inversion.history.first().copied().unwrap_or(0.0);
    let final_objective = inversion
        .history
        .last()
        .copied()
        .unwrap_or(initial_objective);
    let objective_reduction_fraction = if initial_objective > 0.0 {
        ((initial_objective - final_objective) / initial_objective).clamp(0.0, 1.0)
    } else {
        0.0
    };

    Ok(BrainHelmetFwiResult {
        ct_hu: medium.ct_hu.clone(),
        target_sound_speed_m_s: medium.sound_speed_m_s.clone(),
        initial_sound_speed_m_s: medium.initial_sound_speed_m_s.clone(),
        migration_sound_speed_m_s: migration,
        reconstruction_sound_speed_m_s: reconstruction,
        enhanced_reconstruction_sound_speed_m_s: enhanced,
        brain_mask: medium.brain_mask.clone(),
        skull_mask: medium.skull_mask.clone(),
        synthetic_data: data,
        residual_history: inversion.history,
        metrics: BrainHelmetFwiMetrics {
            initial_objective,
            final_objective,
            objective_reduction_fraction,
            migration_pearson_correlation: pearson(&target, &migration_values),
            migration_normalized_rmse: normalized_rmse(&target, &migration_values),
            migration_dynamic_range_m_s: percentile_range(migration_values),
            pearson_correlation: pearson(&target, &recon),
            normalized_rmse: normalized_rmse(&target, &recon),
            enhanced_dynamic_range_m_s: percentile_range(enhanced_values),
            active_voxels: active.len(),
            measurements: nrows,
            continuation_stages: inversion.stages,
            target_dynamic_range_m_s: percentile_range(target),
            reconstruction_dynamic_range_m_s: percentile_range(recon),
        },
    })
}

#[derive(Clone, Debug)]
struct InversionState {
    model: Vec<f64>,
    history: Vec<f64>,
    stages: usize,
}

fn active_voxels(medium: &AcousticSlice) -> Vec<ActiveVoxel> {
    let (nx, ny) = medium.sound_speed_m_s.dim();
    let cx = (nx - 1) as f64 / 2.0;
    let cy = (ny - 1) as f64 / 2.0;
    let mut active = Vec::new();
    for ix in 0..nx {
        for iy in 0..ny {
            if medium.brain_mask[[ix, iy]] {
                let speed = medium.sound_speed_m_s[[ix, iy]];
                active.push(ActiveVoxel {
                    ix,
                    iy,
                    x_m: (ix as f64 - cx) * medium.spacing_m,
                    y_m: (iy as f64 - cy) * medium.spacing_m,
                    z_m: medium.slice_offset_m,
                    target_contrast: speed / C_BRAIN_REF_M_S - 1.0,
                });
            }
        }
    }
    active
}

fn invert(
    matrix: &[f64],
    data: &[f64],
    nrows: usize,
    ncols: usize,
    config: &BrainHelmetFwiConfig,
    active: &[ActiveVoxel],
    shape: (usize, usize),
) -> InversionState {
    let mut model = vec![0.0; ncols];
    let stages = continuation_rows(config, nrows);

    let mut history = Vec::with_capacity(config.iterations + 1);
    let mut current_objective =
        objective(matrix, data, &model, nrows, ncols, config.regularization);
    history.push(current_objective);

    let mut completed = 0;
    for (stage_idx, rows) in stages.iter().enumerate() {
        let stage_iterations = stage_iteration_count(config.iterations, stages.len(), stage_idx);
        if stage_iterations == 0 {
            continue;
        }
        let diag = normal_equation_diagonal_rows(matrix, rows, ncols, config.regularization);
        let mut stage_objective =
            objective_rows(matrix, data, &model, rows, ncols, config.regularization);

        for _ in 0..stage_iterations {
            let mut gradient =
                normalized_gradient_rows(matrix, data, &model, rows, ncols, &diag, config);
            apply_sobolev_preconditioner(&mut gradient, active, shape, config);
            let mut step = config.relaxation;
            let mut accepted = model.clone();
            let mut accepted_stage_objective = stage_objective;

            while step >= 1.0e-10 {
                let mut trial = model.clone();
                for col in 0..ncols {
                    trial[col] = (trial[col] + step * gradient[col])
                        .clamp(config.contrast_min, config.contrast_max);
                }
                let trial_stage_objective =
                    objective_rows(matrix, data, &trial, rows, ncols, config.regularization);
                if trial_stage_objective <= stage_objective {
                    accepted = trial;
                    accepted_stage_objective = trial_stage_objective;
                    break;
                }
                step *= 0.5;
            }

            model = accepted;
            stage_objective = accepted_stage_objective;
            current_objective =
                objective(matrix, data, &model, nrows, ncols, config.regularization);
            history.push(current_objective);
            completed += 1;
        }
    }

    if completed == 0 {
        history.push(current_objective);
    }

    InversionState {
        model,
        history,
        stages: stages.len(),
    }
}

