//! Geometric path-length voxel sensitivity.

use super::super::ray::segment_grid_lengths;
use super::{active_column, path::PathSegment};

pub(super) fn entries(
    path: &[PathSegment],
    active_lookup: &[Option<usize>],
    shape: (usize, usize),
    spacing_m: f64,
) -> Vec<(usize, f64)> {
    let (_, ny) = shape;
    let mut out = Vec::new();
    for segment in path {
        out.extend(
            segment_grid_lengths(segment.start, segment.end, shape, spacing_m)
                .into_iter()
                .filter_map(|((ix, iy), length)| active_column(active_lookup, ix, iy, ny, length)),
        );
    }
    out
}
