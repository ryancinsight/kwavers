//! Frequency-continuation and Sobolev-style conditioning for transcranial UST.

use ndarray::Array2;

use super::born::ActiveVoxel;
use crate::solver::inverse::linear_born_inversion::LinearBornInversionConfig;

/// Build low-to-high frequency row schedules.
pub(super) fn continuation_rows(
    config: &LinearBornInversionConfig,
    nrows: usize,
) -> Vec<Vec<usize>> {
    let nf = config.frequencies_hz.len();
    let harmonic_count = config.harmonic_count();
    let stage_count = if config.frequency_continuation { nf } else { 1 };
    let mut stages = Vec::with_capacity(stage_count);
    for stage in 0..stage_count {
        let prefix = if config.frequency_continuation {
            stage + 1
        } else {
            nf
        };
        let rows = (0..nrows)
            .filter(|row| (row / harmonic_count) % nf < prefix)
            .collect();
        stages.push(rows);
    }
    stages
}

pub(super) fn stage_iteration_count(total: usize, stages: usize, stage_idx: usize) -> usize {
    let base = total / stages;
    let extra = usize::from(stage_idx < total % stages);
    base + extra
}

/// Apply a mask-aware Sobolev gradient smoother over active brain voxels.
pub(super) fn apply_sobolev_preconditioner(
    gradient: &mut [f64],
    active: &[ActiveVoxel],
    shape: (usize, usize),
    config: &LinearBornInversionConfig,
) {
    if config.sobolev_radius_voxels == 0 || config.sobolev_weight == 0.0 {
        return;
    }
    let smoothed = smooth_active_values(gradient, active, shape, config.sobolev_radius_voxels);
    for (value, smooth) in gradient.iter_mut().zip(smoothed) {
        *value = (1.0 - config.sobolev_weight) * *value + config.sobolev_weight * smooth;
    }
}

/// Return a structure-enhanced display image derived from the reconstruction.
pub(super) fn enhance_reconstruction(
    reconstruction: &Array2<f64>,
    brain_mask: &Array2<bool>,
    gain: f64,
    c_ref_m_s: f64,
) -> Array2<f64> {
    if gain == 0.0 {
        return reconstruction.clone();
    }
    let (nx, ny) = reconstruction.dim();
    let mut enhanced = reconstruction.clone();
    for ix in 0..nx {
        for iy in 0..ny {
            if !brain_mask[[ix, iy]] {
                continue;
            }
            let x0 = ix.saturating_sub(1);
            let x1 = (ix + 1).min(nx - 1);
            let y0 = iy.saturating_sub(1);
            let y1 = (iy + 1).min(ny - 1);
            let mut sum = 0.0;
            let mut count = 0.0;
            for nx_idx in x0..=x1 {
                for ny_idx in y0..=y1 {
                    if brain_mask[[nx_idx, ny_idx]] {
                        sum += reconstruction[[nx_idx, ny_idx]];
                        count += 1.0;
                    }
                }
            }
            if count > 0.0 {
                let blur = sum / count;
                let high_pass = reconstruction[[ix, iy]] - blur;
                enhanced[[ix, iy]] = (reconstruction[[ix, iy]] + gain * high_pass)
                    .clamp(c_ref_m_s * 0.92, c_ref_m_s * 1.08);
            }
        }
    }
    enhanced
}

/// Separable 2-D box filter over the active-voxel sparse set.
///
/// Two sequential 1-D prefix-sum passes (X → Y) reduce cost from
/// O(N·(2r+1)²) to O(N + 4·NX·NY).  A parallel count array is filtered
/// identically so inactive positions (zero-padded) do not dilute the average;
/// the gather step divides `sum / count`, preserving active-only averaging.
///
/// Memory layout: flat `[ix * NY + iy]` (row-major).
fn smooth_active_values(
    values: &[f64],
    active: &[ActiveVoxel],
    shape: (usize, usize),
    radius: usize,
) -> Vec<f64> {
    let (nx, ny) = shape;
    let size = nx * ny;

    // Scatter onto dense flat grids; inactive positions remain 0.
    let mut sum_dense = vec![0.0f64; size];
    let mut cnt_dense = vec![0.0f64; size];
    for (col, voxel) in active.iter().enumerate() {
        let idx = voxel.ix * ny + voxel.iy;
        sum_dense[idx] = values[col];
        cnt_dense[idx] = 1.0;
    }

    // Two separable 1-D prefix-sum box filter passes.
    box_filter_2d_x(&mut sum_dense, nx, ny, radius);
    box_filter_2d_x(&mut cnt_dense, nx, ny, radius);
    box_filter_2d_y(&mut sum_dense, nx, ny, radius);
    box_filter_2d_y(&mut cnt_dense, nx, ny, radius);

    // Gather: divide sum by count; fall back to original when count == 0.
    let mut out = vec![0.0f64; values.len()];
    for (col, voxel) in active.iter().enumerate() {
        let idx = voxel.ix * ny + voxel.iy;
        let c = cnt_dense[idx];
        out[col] = if c > 0.5 {
            sum_dense[idx] / c
        } else {
            values[col]
        };
    }
    out
}

/// In-place 1-D prefix-sum box filter along X for a flat `[nx × ny]` array.
///
/// For each iy column the X-line is extracted, filtered with a symmetric
/// window of half-width `r`, and written back.  A single `line` and `prefix`
/// buffer are allocated once per call and reused across all iy columns.
fn box_filter_2d_x(data: &mut [f64], nx: usize, ny: usize, r: usize) {
    let mut line = vec![0.0f64; nx];
    let mut prefix = vec![0.0f64; nx + 1];
    for iy in 0..ny {
        for ix in 0..nx {
            line[ix] = data[ix * ny + iy];
        }
        apply_box_filter_1d(&mut line, r, &mut prefix);
        for ix in 0..nx {
            data[ix * ny + iy] = line[ix];
        }
    }
}

/// In-place 1-D prefix-sum box filter along Y (contiguous) for a flat `[nx × ny]` array.
///
/// Y lines are contiguous in memory (`[ix * NY .. ix * NY + NY]`); the filter
/// applies directly to each `ny`-element slice without gather/scatter.
fn box_filter_2d_y(data: &mut [f64], nx: usize, ny: usize, r: usize) {
    let mut prefix = vec![0.0f64; ny + 1];
    for ix in 0..nx {
        let base = ix * ny;
        apply_box_filter_1d(&mut data[base..base + ny], r, &mut prefix);
    }
}

/// Replace each element of `line` with the sum over the symmetric window of
/// half-width `r`, computed in O(L) via prefix sums.
///
/// `scratch` is reused across calls to avoid repeated heap allocation.
fn apply_box_filter_1d(line: &mut [f64], r: usize, scratch: &mut Vec<f64>) {
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
