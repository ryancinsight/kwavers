//! Matrix-free 3-D Born inversion for CT-derived brain speed contrast.

use crate::core::error::KwaversResult;
use ndarray::Array3;

use super::{
    conditioning::{continuation_rows, stage_iteration_count},
    config::{BrainHelmetFwiConfig, C_BRAIN_REF_M_S},
    transducer::HelmetHemisphereGeometry,
    volume::AcousticVolume,
    volume_operator::{VolumeOperator, VolumeVoxel},
    volume_regularization::{
        edge_preserving_penalty, edge_preserving_projection, enhance_reconstruction_volume,
    },
};

/// Quality metrics for the reconstructed 3-D brain volume.
#[derive(Clone, Debug)]
pub struct BrainHelmetFwiVolumeMetrics {
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

/// Result arrays and diagnostics from a 3-D encoded helmet inversion.
#[derive(Clone, Debug)]
pub struct BrainHelmetFwiVolumeResult {
    pub ct_hu: Array3<f64>,
    pub target_sound_speed_m_s: Array3<f64>,
    pub initial_sound_speed_m_s: Array3<f64>,
    pub migration_sound_speed_m_s: Array3<f64>,
    pub reconstruction_sound_speed_m_s: Array3<f64>,
    pub enhanced_reconstruction_sound_speed_m_s: Array3<f64>,
    pub brain_mask: Array3<bool>,
    pub skull_mask: Array3<bool>,
    pub synthetic_data: Vec<f64>,
    pub residual_history: Vec<f64>,
    pub metrics: BrainHelmetFwiVolumeMetrics,
}

/// Reconstruct brain sound-speed contrast as one coupled 3-D volume.
pub fn reconstruct_brain_volume(
    medium: &AcousticVolume,
    config: &BrainHelmetFwiConfig,
) -> KwaversResult<BrainHelmetFwiVolumeResult> {
    config.validate()?;
    let geometry = HelmetHemisphereGeometry::uniform(config.element_count, config.radius_m)?;
    let receiver_indices = geometry.receiver_indices(&config.receiver_offsets);
    let active = active_voxels(medium);
    let operator = VolumeOperator::new(
        geometry,
        receiver_indices,
        &active,
        medium.spacing_m * medium.spacing_m * medium.spacing_m,
        config,
    );
    let nrows = config.measurement_count();
    let all_rows: Vec<usize> = (0..nrows).collect();
    let row_norms = operator.row_norms();
    let data = operator.data_from_target(&row_norms);
    let migration_model = operator.migration(&data, &all_rows, &row_norms, config);
    let inversion = invert(
        &operator,
        &data,
        &row_norms,
        config,
        &active,
        medium.sound_speed_m_s.dim(),
    );

    let migration = fill_sound_speed_volume(medium, &active, &migration_model);
    let reconstruction = fill_sound_speed_volume(medium, &active, &inversion.model);
    let enhanced = enhance_reconstruction_volume(
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
        .map(|voxel| enhanced[[voxel.ix, voxel.iy, voxel.iz]])
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

    Ok(BrainHelmetFwiVolumeResult {
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
        metrics: BrainHelmetFwiVolumeMetrics {
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

fn active_voxels(medium: &AcousticVolume) -> Vec<VolumeVoxel> {
    let (nx, ny, nz) = medium.sound_speed_m_s.dim();
    let cx = (nx - 1) as f64 / 2.0;
    let cy = (ny - 1) as f64 / 2.0;
    let cz = (nz - 1) as f64 / 2.0;
    let mut active = Vec::new();
    for ix in 0..nx {
        for iy in 0..ny {
            for iz in 0..nz {
                if medium.brain_mask[[ix, iy, iz]] {
                    let speed = medium.sound_speed_m_s[[ix, iy, iz]];
                    active.push(VolumeVoxel {
                        ix,
                        iy,
                        iz,
                        x_m: (ix as f64 - cx) * medium.spacing_m,
                        y_m: (iy as f64 - cy) * medium.spacing_m,
                        z_m: (iz as f64 - cz) * medium.spacing_m,
                        target_contrast: speed / C_BRAIN_REF_M_S - 1.0,
                        attenuation_np_per_m_mhz: medium.attenuation_np_per_m_mhz[[ix, iy, iz]],
                    });
                }
            }
        }
    }
    active
}

fn invert(
    operator: &VolumeOperator<'_>,
    data: &[f64],
    row_norms: &[f64],
    config: &BrainHelmetFwiConfig,
    active: &[VolumeVoxel],
    shape: (usize, usize, usize),
) -> InversionState {
    let mut model = vec![0.0; active.len()];
    let stages = continuation_rows(config, data.len());
    let all_rows: Vec<usize> = (0..data.len()).collect();
    let mut history = vec![composite_objective(
        operator, data, &model, &all_rows, row_norms, config, active, shape,
    )];

    for (stage_idx, rows) in stages.iter().enumerate() {
        let stage_iterations = stage_iteration_count(config.iterations, stages.len(), stage_idx);
        if stage_iterations == 0 {
            continue;
        }
        let diagonal = operator.diagonal(rows, row_norms, config);
        StagePcgContext {
            operator,
            data,
            row_norms,
            config,
            active,
            shape,
            rows,
            all_rows: &all_rows,
            diagonal: &diagonal,
        }
        .solve(stage_iterations, &mut model, &mut history);
        let current_objective = composite_objective(
            operator, data, &model, &all_rows, row_norms, config, active, shape,
        );
        if let Some(projected) = edge_preserving_projection(&model, active, shape, config) {
            let projected_objective = composite_objective(
                operator, data, &projected, &all_rows, row_norms, config, active, shape,
            );
            if projected_objective <= current_objective {
                model = projected;
                history.push(projected_objective);
            }
        }
    }

    InversionState {
        model,
        history,
        stages: stages.len(),
    }
}

struct StagePcgContext<'a> {
    operator: &'a VolumeOperator<'a>,
    data: &'a [f64],
    row_norms: &'a [f64],
    config: &'a BrainHelmetFwiConfig,
    active: &'a [VolumeVoxel],
    shape: (usize, usize, usize),
    rows: &'a [usize],
    all_rows: &'a [usize],
    diagonal: &'a [f64],
}

impl StagePcgContext<'_> {
    fn solve(&self, stage_iterations: usize, model: &mut Vec<f64>, history: &mut Vec<f64>) {
        let mut residual = self.operator.normal_residual(
            self.data,
            model,
            self.rows,
            self.row_norms,
            self.config.regularization,
        );
        let mut z = self.precondition(&residual);
        let mut direction = z.clone();
        let mut rz_old = dot(&residual, &z);
        let mut stage_objective = composite_objective(
            self.operator,
            self.data,
            model,
            self.rows,
            self.row_norms,
            self.config,
            self.active,
            self.shape,
        );
        if rz_old <= 0.0 || !rz_old.is_finite() {
            return;
        }

        for _ in 0..stage_iterations {
            let normal_direction = self.operator.apply_normal(
                &direction,
                self.rows,
                self.row_norms,
                self.config.regularization,
            );
            let denom = dot(&direction, &normal_direction);
            if denom <= 0.0 || !denom.is_finite() {
                break;
            }

            let mut alpha = self.config.relaxation.min(1.0) * rz_old / denom;
            let mut accepted_model = model.clone();
            let mut accepted_objective = stage_objective;
            while alpha >= 1.0e-12 {
                let mut trial = model.clone();
                for (value, direction_value) in trial.iter_mut().zip(&direction) {
                    *value = (*value + alpha * direction_value)
                        .clamp(self.config.contrast_min, self.config.contrast_max);
                }
                let trial_objective = composite_objective(
                    self.operator,
                    self.data,
                    &trial,
                    self.rows,
                    self.row_norms,
                    self.config,
                    self.active,
                    self.shape,
                );
                if trial_objective <= stage_objective {
                    accepted_model = trial;
                    accepted_objective = trial_objective;
                    break;
                }
                alpha *= 0.5;
            }

            if accepted_objective == stage_objective {
                break;
            }

            for (residual_value, normal_value) in residual.iter_mut().zip(normal_direction) {
                *residual_value -= alpha * normal_value;
            }
            *model = accepted_model;
            stage_objective = accepted_objective;
            history.push(composite_objective(
                self.operator,
                self.data,
                model,
                self.all_rows,
                self.row_norms,
                self.config,
                self.active,
                self.shape,
            ));

            z = self.precondition(&residual);
            let rz_new = dot(&residual, &z);
            if rz_new <= 1.0e-24 || !rz_new.is_finite() {
                break;
            }
            let beta = rz_new / rz_old;
            for (direction_value, z_value) in direction.iter_mut().zip(&z) {
                *direction_value = z_value + beta * *direction_value;
            }
            rz_old = rz_new;
        }
    }

    fn precondition(&self, residual: &[f64]) -> Vec<f64> {
        let mut out: Vec<f64> = residual
            .iter()
            .zip(self.diagonal)
            .map(|(value, diag)| value / diag.max(1.0e-12))
            .collect();
        apply_sobolev_preconditioner_3d(&mut out, self.active, self.shape, self.config);
        out
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(av, bv)| av * bv).sum()
}

fn composite_objective(
    operator: &VolumeOperator<'_>,
    data: &[f64],
    model: &[f64],
    rows: &[usize],
    row_norms: &[f64],
    config: &BrainHelmetFwiConfig,
    active: &[VolumeVoxel],
    shape: (usize, usize, usize),
) -> f64 {
    operator.objective(data, model, rows, row_norms, config.regularization)
        + edge_preserving_penalty(model, active, shape, config)
}

fn apply_sobolev_preconditioner_3d(
    gradient: &mut [f64],
    active: &[VolumeVoxel],
    shape: (usize, usize, usize),
    config: &BrainHelmetFwiConfig,
) {
    if config.sobolev_radius_voxels == 0 || config.sobolev_weight == 0.0 {
        return;
    }
    let smoothed = smooth_active_values_3d(gradient, active, shape, config.sobolev_radius_voxels);
    for (value, smooth) in gradient.iter_mut().zip(smoothed) {
        *value = (1.0 - config.sobolev_weight) * *value + config.sobolev_weight * smooth;
    }
}

fn smooth_active_values_3d(
    values: &[f64],
    active: &[VolumeVoxel],
    shape: (usize, usize, usize),
    radius: usize,
) -> Vec<f64> {
    let mut index = Array3::<isize>::from_elem(shape, -1);
    for (col, voxel) in active.iter().enumerate() {
        index[[voxel.ix, voxel.iy, voxel.iz]] = col as isize;
    }
    let (nx, ny, nz) = shape;
    let mut out = vec![0.0; values.len()];
    for (col, voxel) in active.iter().enumerate() {
        let x0 = voxel.ix.saturating_sub(radius);
        let x1 = (voxel.ix + radius).min(nx - 1);
        let y0 = voxel.iy.saturating_sub(radius);
        let y1 = (voxel.iy + radius).min(ny - 1);
        let z0 = voxel.iz.saturating_sub(radius);
        let z1 = (voxel.iz + radius).min(nz - 1);
        let mut sum = 0.0;
        let mut count = 0.0;
        for ix in x0..=x1 {
            for iy in y0..=y1 {
                for iz in z0..=z1 {
                    let neighbor = index[[ix, iy, iz]];
                    if neighbor >= 0 {
                        sum += values[neighbor as usize];
                        count += 1.0;
                    }
                }
            }
        }
        out[col] = if count > 0.0 {
            sum / count
        } else {
            values[col]
        };
    }
    out
}

fn fill_sound_speed_volume(
    medium: &AcousticVolume,
    active: &[VolumeVoxel],
    contrast: &[f64],
) -> Array3<f64> {
    let mut out = medium.initial_sound_speed_m_s.clone();
    for (voxel, value) in active.iter().zip(contrast) {
        out[[voxel.ix, voxel.iy, voxel.iz]] = C_BRAIN_REF_M_S * (1.0 + value);
    }
    out
}

fn pearson(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() || a.len() < 2 {
        return 0.0;
    }
    let ma = a.iter().sum::<f64>() / a.len() as f64;
    let mb = b.iter().sum::<f64>() / b.len() as f64;
    let mut num = 0.0;
    let mut da = 0.0;
    let mut db = 0.0;
    for (&av, &bv) in a.iter().zip(b) {
        let xa = av - ma;
        let xb = bv - mb;
        num += xa * xb;
        da += xa * xa;
        db += xb * xb;
    }
    if da > 0.0 && db > 0.0 {
        num / (da.sqrt() * db.sqrt())
    } else {
        0.0
    }
}

fn normalized_rmse(a: &[f64], b: &[f64]) -> f64 {
    let norm = a.iter().map(|v| v * v).sum::<f64>().sqrt();
    if norm == 0.0 {
        return 0.0;
    }
    let err = a
        .iter()
        .zip(b)
        .map(|(&av, &bv)| {
            let d = av - bv;
            d * d
        })
        .sum::<f64>()
        .sqrt();
    err / norm
}

fn percentile_range(mut values: Vec<f64>) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    values.sort_by(|a, b| a.total_cmp(b));
    let last = values.len() - 1;
    let p05 = values[(0.05 * last as f64).round() as usize];
    let p95 = values[(0.95 * last as f64).round() as usize];
    p95 - p05
}
