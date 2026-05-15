//! Centroid utilities for CT-derived nonlinear 3-D supports.

use ndarray::Array3;

use super::super::types::GridIndex;

pub(crate) fn centroid_index(mask: &Array3<bool>) -> Option<GridIndex> {
    centroid_float(mask, None).map(|center| GridIndex {
        x: center[0].round() as usize,
        y: center[1].round() as usize,
        z: center[2].round() as usize,
    })
}

/// Unscaled grid-index centroid of active voxels in `mask`.
///
/// When `z_range` is `Some((lo, hi))`, only voxels with `lo <= iz <= hi` are
/// included.  Returns `None` when no voxels match.
pub(crate) fn centroid_float(
    mask: &Array3<bool>,
    z_range: Option<(usize, usize)>,
) -> Option<[f64; 3]> {
    let mut sum = [0.0; 3];
    let mut count = 0.0;
    for ((ix, iy, iz), active) in mask.indexed_iter() {
        let in_range = z_range.map_or(true, |(lo, hi)| iz >= lo && iz <= hi);
        if *active && in_range {
            sum[0] += ix as f64;
            sum[1] += iy as f64;
            sum[2] += iz as f64;
            count += 1.0;
        }
    }
    (count > 0.0).then_some([sum[0] / count, sum[1] / count, sum[2] / count])
}
