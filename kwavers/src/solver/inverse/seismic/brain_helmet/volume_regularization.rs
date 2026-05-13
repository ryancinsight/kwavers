//! Edge-preserving proximal regularization for 3-D helmet FWI.

use ndarray::Array3;

use super::{config::BrainHelmetFwiConfig, volume_operator::VolumeVoxel};

pub(super) fn edge_preserving_penalty(
    model: &[f64],
    active: &[VolumeVoxel],
    shape: (usize, usize, usize),
    config: &BrainHelmetFwiConfig,
) -> f64 {
    if config.edge_preserving_weight == 0.0 {
        return 0.0;
    }
    let index = active_index(active, shape);
    let epsilon = config.edge_preserving_epsilon;
    let mut penalty = 0.0;
    for (col, voxel) in active.iter().enumerate() {
        for_positive_neighbors(voxel, shape, &index, |neighbor| {
            let diff = model[col] - model[neighbor];
            penalty += (diff * diff + epsilon * epsilon).sqrt() - epsilon;
        });
    }
    config.edge_preserving_weight * penalty
}

pub(super) fn edge_preserving_projection(
    model: &[f64],
    active: &[VolumeVoxel],
    shape: (usize, usize, usize),
    config: &BrainHelmetFwiConfig,
) -> Option<Vec<f64>> {
    if config.edge_preserving_iterations == 0 || config.edge_preserving_step == 0.0 {
        return None;
    }
    let index = active_index(active, shape);
    let mut current = model.to_vec();
    for _ in 0..config.edge_preserving_iterations {
        let mut next = current.clone();
        for (col, voxel) in active.iter().enumerate() {
            let center = current[col];
            let mut weighted_sum = 0.0;
            let mut weight_total = 0.0;
            for_all_neighbors(voxel, shape, &index, |neighbor| {
                let neighbor_value = current[neighbor];
                let diffusivity = charbonnier_diffusivity(
                    neighbor_value - center,
                    config.edge_preserving_epsilon,
                );
                weighted_sum += diffusivity * neighbor_value;
                weight_total += diffusivity;
            });
            if weight_total > 0.0 {
                let local_mean = weighted_sum / weight_total;
                next[col] = ((1.0 - config.edge_preserving_step) * center
                    + config.edge_preserving_step * local_mean)
                    .clamp(config.contrast_min, config.contrast_max);
            }
        }
        current = next;
    }
    Some(current)
}

pub(super) fn enhance_reconstruction_volume(
    reconstruction: &Array3<f64>,
    brain_mask: &Array3<bool>,
    gain: f64,
    c_ref_m_s: f64,
) -> Array3<f64> {
    if gain == 0.0 {
        return reconstruction.clone();
    }
    let (nx, ny, nz) = reconstruction.dim();
    let mut enhanced = reconstruction.clone();
    for ix in 0..nx {
        for iy in 0..ny {
            for iz in 0..nz {
                if !brain_mask[[ix, iy, iz]] {
                    continue;
                }
                let mut sum = 0.0;
                let mut count = 0.0;
                for ax in ix.saturating_sub(1)..=(ix + 1).min(nx - 1) {
                    for ay in iy.saturating_sub(1)..=(iy + 1).min(ny - 1) {
                        for az in iz.saturating_sub(1)..=(iz + 1).min(nz - 1) {
                            if brain_mask[[ax, ay, az]] {
                                sum += reconstruction[[ax, ay, az]];
                                count += 1.0;
                            }
                        }
                    }
                }
                if count > 0.0 {
                    let blur = sum / count;
                    let high_pass = reconstruction[[ix, iy, iz]] - blur;
                    enhanced[[ix, iy, iz]] = (reconstruction[[ix, iy, iz]] + gain * high_pass)
                        .clamp(c_ref_m_s * 0.92, c_ref_m_s * 1.08);
                }
            }
        }
    }
    enhanced
}

fn charbonnier_diffusivity(diff: f64, epsilon: f64) -> f64 {
    1.0 / (1.0 + (diff / epsilon).powi(2)).sqrt()
}

fn active_index(active: &[VolumeVoxel], shape: (usize, usize, usize)) -> Array3<isize> {
    let mut index = Array3::<isize>::from_elem(shape, -1);
    for (col, voxel) in active.iter().enumerate() {
        index[[voxel.ix, voxel.iy, voxel.iz]] = col as isize;
    }
    index
}

fn for_positive_neighbors<F>(
    voxel: &VolumeVoxel,
    shape: (usize, usize, usize),
    index: &Array3<isize>,
    mut f: F,
) where
    F: FnMut(usize),
{
    let (nx, ny, nz) = shape;
    if voxel.ix + 1 < nx {
        visit_neighbor(index[[voxel.ix + 1, voxel.iy, voxel.iz]], &mut f);
    }
    if voxel.iy + 1 < ny {
        visit_neighbor(index[[voxel.ix, voxel.iy + 1, voxel.iz]], &mut f);
    }
    if voxel.iz + 1 < nz {
        visit_neighbor(index[[voxel.ix, voxel.iy, voxel.iz + 1]], &mut f);
    }
}

fn for_all_neighbors<F>(
    voxel: &VolumeVoxel,
    shape: (usize, usize, usize),
    index: &Array3<isize>,
    mut f: F,
) where
    F: FnMut(usize),
{
    let (nx, ny, nz) = shape;
    if voxel.ix > 0 {
        visit_neighbor(index[[voxel.ix - 1, voxel.iy, voxel.iz]], &mut f);
    }
    if voxel.ix + 1 < nx {
        visit_neighbor(index[[voxel.ix + 1, voxel.iy, voxel.iz]], &mut f);
    }
    if voxel.iy > 0 {
        visit_neighbor(index[[voxel.ix, voxel.iy - 1, voxel.iz]], &mut f);
    }
    if voxel.iy + 1 < ny {
        visit_neighbor(index[[voxel.ix, voxel.iy + 1, voxel.iz]], &mut f);
    }
    if voxel.iz > 0 {
        visit_neighbor(index[[voxel.ix, voxel.iy, voxel.iz - 1]], &mut f);
    }
    if voxel.iz + 1 < nz {
        visit_neighbor(index[[voxel.ix, voxel.iy, voxel.iz + 1]], &mut f);
    }
}

fn visit_neighbor<F>(neighbor: isize, f: &mut F)
where
    F: FnMut(usize),
{
    if neighbor >= 0 {
        f(neighbor as usize);
    }
}
