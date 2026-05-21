//! PCG inversion kernel for the 3-D Born transcranial focused-bowl solver.

use ndarray::Array3;

use super::super::{
    conditioning::{continuation_rows, stage_iteration_count},
    config::TranscranialUstBornInversionConfig,
    volume_operator::{VolumeOperator, VolumeVoxel},
    volume_regularization::{
        build_active_index, edge_preserving_penalty, edge_preserving_projection,
    },
};

/// Regularization context bundling all parameters needed by `composite_objective`.
///
/// `active_index` is built once in `invert` and reused across the entire
/// inversion loop, avoiding an O(NX·NY·NZ) `Array3` allocation on every
/// line-search trial.
type RegCtx<'a> = (
    &'a TranscranialUstBornInversionConfig,
    &'a [VolumeVoxel],
    (usize, usize, usize),
    &'a Array3<isize>,
);

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
    config: &TranscranialUstBornInversionConfig,
    active: &[VolumeVoxel],
    shape: (usize, usize, usize),
) -> InversionState {
    let mut model = vec![0.0; active.len()];
    let stages = continuation_rows(config, data.len());
    let all_rows: Vec<usize> = (0..data.len()).collect();
    // Build active-index lookup once; passed through every composite_objective call so
    // the O(NX·NY·NZ) Array3 allocation occurs exactly once per inversion.
    let active_index = build_active_index(active, shape);
    let reg: RegCtx<'_> = (config, active, shape, &active_index);
    let mut history = vec![composite_objective(
        operator, data, &model, &all_rows, row_norms, reg,
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
            diagonal: &diagonal,
            active_index: &active_index,
        }
        .solve(stage_iterations, &mut model, &mut history);
        let current_objective =
            composite_objective(operator, data, &model, &all_rows, row_norms, reg);
        if let Some(projected) =
            edge_preserving_projection(&model, active, shape, &active_index, config)
        {
            let projected_objective =
                composite_objective(operator, data, &projected, &all_rows, row_norms, reg);
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
    config: &'a TranscranialUstBornInversionConfig,
    active: &'a [VolumeVoxel],
    shape: (usize, usize, usize),
    rows: &'a [usize],
    diagonal: &'a [f64],
    active_index: &'a Array3<isize>,
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
        let reg: RegCtx<'_> = (self.config, self.active, self.shape, self.active_index);
        let mut stage_objective = composite_objective(
            self.operator,
            self.data,
            model,
            self.rows,
            self.row_norms,
            reg,
        );
        if rz_old <= 0.0 || !rz_old.is_finite() {
            return;
        }

        // Pre-allocate line-search buffers once per `solve` call.
        // At clinical grid sizes (50 K+ active voxels) the original
        // `model.clone()` on every iteration and every backtrack step
        // incurred O(stage_iterations × backtrack_depth) heap allocations,
        // each ~400 KB.  Reusing these two buffers via `copy_from_slice`
        // reduces that to two allocations per `solve` call.
        let ncols = model.len();
        let mut accepted_model = vec![0.0f64; ncols];
        let mut trial = vec![0.0f64; ncols];

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
            accepted_model.copy_from_slice(model);
            let mut accepted_objective = stage_objective;
            while alpha >= 1.0e-12 {
                trial.copy_from_slice(model);
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
                    reg,
                );
                if trial_objective <= stage_objective {
                    accepted_model.copy_from_slice(&trial);
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
            model.copy_from_slice(&accepted_model);
            stage_objective = accepted_objective;
            // Push the accepted stage objective directly; re-evaluating over all_rows
            // costs ~5× more and is not needed for convergence tracking within a stage.
            // `invert` appends an all-rows objective at each stage boundary.
            history.push(accepted_objective);

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
    reg: RegCtx<'_>,
) -> f64 {
    let (config, active, shape, active_index) = reg;
    operator.objective(data, model, rows, row_norms, config.regularization)
        + edge_preserving_penalty(model, active, shape, active_index, config)
}

fn apply_sobolev_preconditioner_3d(
    gradient: &mut [f64],
    active: &[VolumeVoxel],
    shape: (usize, usize, usize),
    config: &TranscranialUstBornInversionConfig,
) {
    if config.sobolev_radius_voxels == 0 || config.sobolev_weight == 0.0 {
        return;
    }
    let smoothed = smooth_active_values_3d(gradient, active, shape, config.sobolev_radius_voxels);
    for (value, smooth) in gradient.iter_mut().zip(smoothed) {
        *value = (1.0 - config.sobolev_weight) * *value + config.sobolev_weight * smooth;
    }
}

/// Separable 3-D box filter over the active-voxel sparse set.
///
/// Three sequential 1-D prefix-sum passes (X → Y → Z) reduce cost from
/// O(N·(2r+1)³) to O(N + 6·NX·NY·NZ).  A parallel count array is filtered
/// identically so inactive positions (zero-padded) do not dilute the average;
/// the gather step divides `sum / count`, preserving the semantics of the
/// original neighbourhood-average over active voxels only.
///
/// Memory layout: flat `[ix * NY * NZ + iy * NZ + iz]` (C row-major).
fn smooth_active_values_3d(
    values: &[f64],
    active: &[VolumeVoxel],
    shape: (usize, usize, usize),
    radius: usize,
) -> Vec<f64> {
    let (nx, ny, nz) = shape;
    let size = nx * ny * nz;

    // Scatter onto dense grids; inactive positions remain 0.
    let mut sum_dense = vec![0.0f64; size];
    let mut cnt_dense = vec![0.0f64; size];
    for (col, voxel) in active.iter().enumerate() {
        let idx = voxel.ix * ny * nz + voxel.iy * nz + voxel.iz;
        sum_dense[idx] = values[col];
        cnt_dense[idx] = 1.0;
    }

    // Three separable 1-D prefix-sum box filter passes.
    box_filter_x(&mut sum_dense, nx, ny, nz, radius);
    box_filter_x(&mut cnt_dense, nx, ny, nz, radius);
    box_filter_y(&mut sum_dense, nx, ny, nz, radius);
    box_filter_y(&mut cnt_dense, nx, ny, nz, radius);
    box_filter_z(&mut sum_dense, nx, ny, nz, radius);
    box_filter_z(&mut cnt_dense, nx, ny, nz, radius);

    // Gather: divide sum by count; fall back to original when count == 0.
    let mut out = vec![0.0f64; values.len()];
    for (col, voxel) in active.iter().enumerate() {
        let idx = voxel.ix * ny * nz + voxel.iy * nz + voxel.iz;
        let c = cnt_dense[idx];
        out[col] = if c > 0.5 {
            sum_dense[idx] / c
        } else {
            values[col]
        };
    }
    out
}

/// In-place 1-D prefix-sum box filter along the X axis.
///
/// For each (iy, iz) line the value at position ix becomes the sum over
/// `[ix.saturating_sub(r), (ix + r).min(nx − 1)]`.
///
/// A single `line` gather buffer and a single `prefix` scratch buffer are
/// allocated once per function call and reused across all (iy, iz) pairs.
fn box_filter_x(data: &mut [f64], nx: usize, ny: usize, nz: usize, r: usize) {
    let stride = ny * nz;
    let mut line = vec![0.0f64; nx];
    let mut prefix = vec![0.0f64; nx + 1];
    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                line[ix] = data[ix * stride + iy * nz + iz];
            }
            apply_box_filter_1d_with_scratch(&mut line, r, &mut prefix);
            for ix in 0..nx {
                data[ix * stride + iy * nz + iz] = line[ix];
            }
        }
    }
}

/// In-place 1-D prefix-sum box filter along the Y axis.
///
/// A single `line` and `prefix` buffer are allocated once per function call
/// and reused across all (ix, iz) pairs.
fn box_filter_y(data: &mut [f64], nx: usize, ny: usize, nz: usize, r: usize) {
    let mut line = vec![0.0f64; ny];
    let mut prefix = vec![0.0f64; ny + 1];
    for ix in 0..nx {
        for iz in 0..nz {
            let base = ix * ny * nz + iz;
            for iy in 0..ny {
                line[iy] = data[base + iy * nz];
            }
            apply_box_filter_1d_with_scratch(&mut line, r, &mut prefix);
            for iy in 0..ny {
                data[base + iy * nz] = line[iy];
            }
        }
    }
}

/// In-place 1-D prefix-sum box filter along the Z axis (contiguous).
///
/// Z lines are contiguous in memory (`[ix * NY * NZ + iy * NZ .. + NZ]`).
/// Uses Rayon `for_each_with` so the `prefix` scratch buffer is cloned once
/// per worker thread rather than once per chunk, bounding allocations to the
/// Rayon thread-pool size instead of `NX × NY`.
fn box_filter_z(data: &mut [f64], _nx: usize, _ny: usize, nz: usize, r: usize) {
    use rayon::prelude::*;
    let scratch_init = vec![0.0f64; nz + 1];
    data.par_chunks_mut(nz)
        .for_each_with(scratch_init, |prefix, line| {
            apply_box_filter_1d_with_scratch(line, r, prefix);
        });
}

/// Replace each element of `line` with the sum over the symmetric window of
/// half-width `r`, computed in O(L) via prefix sums.
///
/// `scratch` must have capacity ≥ `line.len() + 1`; it is resized as needed
/// so the caller may pass a pre-allocated buffer of the right size to avoid
/// repeated heap allocation across calls.
fn apply_box_filter_1d_with_scratch(line: &mut [f64], r: usize, scratch: &mut Vec<f64>) {
    let n = line.len();
    if n == 0 {
        return;
    }
    scratch.resize(n + 1, 0.0);
    scratch[0] = 0.0;
    for i in 0..n {
        scratch[i + 1] = scratch[i] + line[i];
    }
    for i in 0..n {
        let lo = i.saturating_sub(r);
        let hi = (i + r + 1).min(n);
        line[i] = scratch[hi] - scratch[lo];
    }
}
