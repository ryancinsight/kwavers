//! Frequency-continuation and Sobolev-style conditioning for brain FWI.

use ndarray::Array2;

use super::{born::ActiveVoxel, config::BrainHelmetFwiConfig};

/// Build low-to-high frequency row schedules.
pub(super) fn continuation_rows(config: &BrainHelmetFwiConfig, nrows: usize) -> Vec<Vec<usize>> {
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
    config: &BrainHelmetFwiConfig,
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

fn smooth_active_values(
    values: &[f64],
    active: &[ActiveVoxel],
    shape: (usize, usize),
    radius: usize,
) -> Vec<f64> {
    let mut index = Array2::<isize>::from_elem(shape, -1);
    for (col, voxel) in active.iter().enumerate() {
        index[[voxel.ix, voxel.iy]] = col as isize;
    }

    let (nx, ny) = shape;
    let mut out = vec![0.0; values.len()];
    for (col, voxel) in active.iter().enumerate() {
        let x0 = voxel.ix.saturating_sub(radius);
        let x1 = (voxel.ix + radius).min(nx - 1);
        let y0 = voxel.iy.saturating_sub(radius);
        let y1 = (voxel.iy + radius).min(ny - 1);
        let mut sum = 0.0;
        let mut count = 0.0;
        for ix in x0..=x1 {
            for iy in y0..=y1 {
                let neighbor = index[[ix, iy]];
                if neighbor >= 0 {
                    sum += values[neighbor as usize];
                    count += 1.0;
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
