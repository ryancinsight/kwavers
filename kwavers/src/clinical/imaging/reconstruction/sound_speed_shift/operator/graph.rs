//! Active-support graph indexing and graph-Laplacian application.

use crate::solver::inverse::same_aperture::ActiveGrid;

use super::SoundSpeedShiftOperator;

impl SoundSpeedShiftOperator {
    pub(in crate::clinical::imaging::reconstruction::sound_speed_shift) fn graph_laplacian_into(
        &self,
        values: &[f64],
        out: &mut [f64],
    ) {
        debug_assert_eq!(values.len(), self.cols());
        debug_assert_eq!(out.len(), self.cols());
        for (row, neighbors) in self.neighbor_indices.iter().enumerate() {
            let center = values[row];
            let mut degree = 0.0;
            let mut sum = 0.0;
            for neighbor in neighbors.iter().flatten() {
                degree += 1.0;
                sum += values[*neighbor];
            }
            out[row] = degree * center - sum;
        }
    }
}

pub(super) fn active_lookup(active: &ActiveGrid, shape: (usize, usize)) -> Vec<Option<usize>> {
    let (nx, ny) = shape;
    let mut lookup = vec![None; nx * ny];
    for (active_idx, (ix, iy)) in active.indices.iter().enumerate() {
        lookup[linear_index(*ix, *iy, ny)] = Some(active_idx);
    }
    lookup
}

pub(super) fn neighbor_indices(
    active: &ActiveGrid,
    shape: (usize, usize),
    lookup: &[Option<usize>],
) -> Vec<[Option<usize>; 4]> {
    let (nx, ny) = shape;
    active
        .indices
        .iter()
        .map(|(ix, iy)| active_neighbors(*ix, *iy, nx, ny, lookup))
        .collect()
}

fn active_neighbors(
    ix: usize,
    iy: usize,
    nx: usize,
    ny: usize,
    lookup: &[Option<usize>],
) -> [Option<usize>; 4] {
    let mut out = [None; 4];
    let mut count = 0;
    for (jx, jy) in lattice_neighbors(ix, iy, nx, ny) {
        if let Some(active_idx) = lookup[linear_index(jx, jy, ny)] {
            out[count] = Some(active_idx);
            count += 1;
        }
    }
    out
}

fn lattice_neighbors(
    ix: usize,
    iy: usize,
    nx: usize,
    ny: usize,
) -> impl Iterator<Item = (usize, usize)> {
    let mut out = [(ix, iy); 4];
    let mut count = 0;
    if ix > 0 {
        out[count] = (ix - 1, iy);
        count += 1;
    }
    if iy > 0 {
        out[count] = (ix, iy - 1);
        count += 1;
    }
    if ix + 1 < nx {
        out[count] = (ix + 1, iy);
        count += 1;
    }
    if iy + 1 < ny {
        out[count] = (ix, iy + 1);
        count += 1;
    }
    out.into_iter().take(count)
}

pub(super) const fn linear_index(ix: usize, iy: usize, ny: usize) -> usize {
    ix * ny + iy
}
