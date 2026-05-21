//! Edge-preserving proximal regularization for 3-D transcranial UST inversion.
//!
//! ## Active-index cache
//!
//! `build_active_index` constructs an `Array3<isize>` that maps each grid
//! position to the column index of the corresponding active voxel (−1 if
//! inactive).  Callers — specifically `pcg::invert` — build this once per
//! inversion and pass it through every `composite_objective` call, avoiding
//! a fresh O(NX·NY·NZ) allocation on every line-search trial.
//!
//! ## Edge-preserving diffusion ping-pong
//!
//! `edge_preserving_projection` now alternates between two pre-allocated
//! buffers with `std::mem::swap`, eliminating the `Vec::clone` that previously
//! occurred on every diffusion iteration.

use ndarray::Array3;

use super::volume_operator::VolumeVoxel;
use crate::solver::inverse::linear_born_inversion::LinearBornInversionConfig;

/// Build the dense active-voxel lookup table once per inversion.
///
/// Returns an `Array3<isize>` of shape `shape` where entry `[ix, iy, iz]`
/// equals the column index of the active voxel at that position, or −1 if the
/// voxel is not in the active set.  Callers must reuse the returned value for
/// the lifetime of the inversion to avoid repeated O(NX·NY·NZ) allocations.
pub(super) fn build_active_index(
    active: &[VolumeVoxel],
    shape: (usize, usize, usize),
) -> Array3<isize> {
    let mut index = Array3::<isize>::from_elem(shape, -1);
    for (col, voxel) in active.iter().enumerate() {
        index[[voxel.ix, voxel.iy, voxel.iz]] = col as isize;
    }
    index
}

/// Charbonnier penalty Σ_{i~j} (sqrt((m_i−m_j)²+ε²) − ε).
///
/// Caller must pass a pre-built `active_index` from `build_active_index` to
/// avoid rebuilding it on every objective evaluation.
pub(super) fn edge_preserving_penalty(
    model: &[f64],
    active: &[VolumeVoxel],
    shape: (usize, usize, usize),
    active_index: &Array3<isize>,
    config: &LinearBornInversionConfig,
) -> f64 {
    if config.edge_preserving_weight == 0.0 {
        return 0.0;
    }
    let epsilon = config.edge_preserving_epsilon;
    let mut penalty = 0.0;
    for (col, voxel) in active.iter().enumerate() {
        for_positive_neighbors(voxel, shape, active_index, |neighbor| {
            let diff = model[col] - model[neighbor];
            penalty += (diff * diff + epsilon * epsilon).sqrt() - epsilon;
        });
    }
    config.edge_preserving_weight * penalty
}

/// Anisotropic diffusion proximal projection (Charbonnier diffusivity).
///
/// Iterates `config.edge_preserving_iterations` Gauss-Seidel steps using a
/// ping-pong between two pre-allocated buffers; no `Vec::clone` is performed
/// inside the loop.
///
/// Caller must pass a pre-built `active_index` from `build_active_index`.
pub(super) fn edge_preserving_projection(
    model: &[f64],
    active: &[VolumeVoxel],
    shape: (usize, usize, usize),
    active_index: &Array3<isize>,
    config: &LinearBornInversionConfig,
) -> Option<Vec<f64>> {
    if config.edge_preserving_iterations == 0 || config.edge_preserving_step == 0.0 {
        return None;
    }
    let n = model.len();
    let mut buf_a = model.to_vec();
    let mut buf_b = vec![0.0f64; n];
    for _ in 0..config.edge_preserving_iterations {
        for (col, voxel) in active.iter().enumerate() {
            let center = buf_a[col];
            let mut weighted_sum = 0.0;
            let mut weight_total = 0.0;
            for_all_neighbors(voxel, shape, active_index, |neighbor| {
                let neighbor_value = buf_a[neighbor];
                let diffusivity = charbonnier_diffusivity(
                    neighbor_value - center,
                    config.edge_preserving_epsilon,
                );
                weighted_sum += diffusivity * neighbor_value;
                weight_total += diffusivity;
            });
            buf_b[col] = if weight_total > 0.0 {
                let local_mean = weighted_sum / weight_total;
                ((1.0 - config.edge_preserving_step) * center
                    + config.edge_preserving_step * local_mean)
                    .clamp(config.contrast_min, config.contrast_max)
            } else {
                center
            };
        }
        std::mem::swap(&mut buf_a, &mut buf_b);
    }
    Some(buf_a)
}

fn charbonnier_diffusivity(diff: f64, epsilon: f64) -> f64 {
    1.0 / (1.0 + (diff / epsilon).powi(2)).sqrt()
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
