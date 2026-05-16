use ndarray::{Array2, Array3};

use crate::core::error::{KwaversError, KwaversResult};

use super::types::IndexBounds3;

/// Grid center in fractional index coordinates: `((n-1)/2, (m-1)/2)`.
pub fn centered_origin_2d(nx: usize, ny: usize) -> (f64, f64) {
    ((nx - 1) as f64 * 0.5, (ny - 1) as f64 * 0.5)
}

/// 4-connected boundary test for a voxel in a 2-D bool array.
pub fn is_boundary_2d(mask: &Array2<bool>, ix: usize, iy: usize) -> bool {
    let (nx, ny) = mask.dim();
    ix == 0
        || iy == 0
        || ix + 1 == nx
        || iy + 1 == ny
        || !mask[[ix - 1, iy]]
        || !mask[[ix + 1, iy]]
        || !mask[[ix, iy - 1]]
        || !mask[[ix, iy + 1]]
}

/// Compute the 3-D axis-aligned bounding box of active voxels in `mask`.
///
/// Returns `Err` when no active voxel exists.
pub fn active_bounds_3d(mask: &Array3<bool>) -> KwaversResult<IndexBounds3> {
    let mut b = IndexBounds3 {
        x0: usize::MAX,
        x1: 0,
        y0: usize::MAX,
        y1: 0,
        z0: usize::MAX,
        z1: 0,
    };
    let mut any = false;
    for ((ix, iy, iz), &active) in mask.indexed_iter() {
        if active {
            b.x0 = b.x0.min(ix);
            b.x1 = b.x1.max(ix);
            b.y0 = b.y0.min(iy);
            b.y1 = b.y1.max(iy);
            b.z0 = b.z0.min(iz);
            b.z1 = b.z1.max(iz);
            any = true;
        }
    }
    any.then_some(b)
        .ok_or_else(|| KwaversError::InvalidInput("mask contains no active voxels".to_owned()))
}

/// 6-connected boundary test for a voxel in a 3-D bool array.
///
/// Returns `true` when the voxel at `(ix, iy, iz)` is active and at least one
/// 6-connected neighbour is either outside the grid bounds or inactive.
pub fn is_boundary_3d(mask: &Array3<bool>, ix: usize, iy: usize, iz: usize) -> bool {
    let (nx, ny, nz) = mask.dim();
    ix == 0
        || iy == 0
        || iz == 0
        || ix + 1 == nx
        || iy + 1 == ny
        || iz + 1 == nz
        || !mask[[ix - 1, iy, iz]]
        || !mask[[ix + 1, iy, iz]]
        || !mask[[ix, iy - 1, iz]]
        || !mask[[ix, iy + 1, iz]]
        || !mask[[ix, iy, iz - 1]]
        || !mask[[ix, iy, iz + 1]]
}
