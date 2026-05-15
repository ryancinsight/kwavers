//! PCG inversion kernel for the 3-D Born brain-helmet solver.

use ndarray::Array3;

use super::super::{
    conditioning::{continuation_rows, stage_iteration_count},
    config::BrainHelmetFwiConfig,
    volume_operator::{VolumeOperator, VolumeVoxel},
    volume_regularization::{edge_preserving_penalty, edge_preserving_projection},
};

#[derive(Clone, Debug)]
pub(super) struct InversionState {
    pub(super) model: Vec<f64>,
    pub(super) history: Vec<f64>,
    pub(super) stages: usize,
}

pub(super) fn invert(
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
    let regularization = (config, active, shape);
    let mut history = vec![composite_objective(
        operator,
        data,
        &model,
        &all_rows,
        row_norms,
        regularization,
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
        let current_objective =
            composite_objective(operator, data, &model, &all_rows, row_norms, regularization);
        if let Some(projected) = edge_preserving_projection(&model, active, shape, config) {
            let projected_objective = composite_objective(
                operator,
                data,
                &projected,
                &all_rows,
                row_norms,
                regularization,
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
        let regularization = (self.config, self.active, self.shape);
        let mut stage_objective = composite_objective(
            self.operator,
            self.data,
            model,
            self.rows,
            self.row_norms,
            regularization,
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
                    regularization,
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
                regularization,
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
    regularization: (&BrainHelmetFwiConfig, &[VolumeVoxel], (usize, usize, usize)),
) -> f64 {
    let (config, active, shape) = regularization;
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
