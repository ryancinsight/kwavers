//! Boundary extraction helpers for placement visualization.

use leto::{Array2, Array3};

use super::super::geometry::{is_boundary_2d, is_boundary_3d};
use super::{centered_origin_2d, distance_3d, Point3};

pub(super) fn surface_points_3d(
    mask: &Array3<bool>,
    sx: f64,
    sy: f64,
    sz: f64,
    z_range: Option<std::ops::RangeInclusive<usize>>,
    stride: usize,
) -> Vec<Point3> {
    let [nx, ny, nz] = mask.shape();
    let mut points = Vec::new();
    for ix in (0..nx).step_by(stride) {
        for iy in (0..ny).step_by(stride) {
            for iz in (0..nz).step_by(stride) {
                if mask[[ix, iy, iz]]
                    && z_range.as_ref().is_none_or(|range| range.contains(&iz))
                    && is_boundary_3d(mask, ix, iy, iz)
                {
                    points.push(Point3 {
                        x_m: (ix as f64 - (nx - 1) as f64 * 0.5) * sx,
                        y_m: (iy as f64 - (ny - 1) as f64 * 0.5) * sy,
                        z_m: (iz as f64 - (nz - 1) as f64 * 0.5) * sz,
                    });
                }
            }
        }
    }
    points
}

pub(super) fn surface_points_2d(
    mask: &Array2<bool>,
    sx: f64,
    sy: f64,
    sz: f64,
    slice_index: usize,
    stride: usize,
) -> Vec<Point3> {
    boundary_points_2d(mask, sx, sy, stride)
        .into_iter()
        .map(|point| Point3 {
            z_m: (slice_index as f64) * sz,
            ..point
        })
        .collect()
}

pub(super) fn boundary_points_2d(
    mask: &Array2<bool>,
    sx: f64,
    sy: f64,
    stride: usize,
) -> Vec<Point3> {
    let [nx, ny] = mask.shape();
    let center = centered_origin_2d(nx, ny);
    let mut points = Vec::new();
    for ix in (0..nx).step_by(stride) {
        for iy in (0..ny).step_by(stride) {
            if mask[[ix, iy]] && is_boundary_2d(mask, ix, iy) {
                points.push(Point3 {
                    x_m: (ix as f64 - center.0) * sx,
                    y_m: (iy as f64 - center.1) * sy,
                    z_m: 0.0,
                });
            }
        }
    }
    points
}

pub(super) fn nearest_surface_point(points: &[Point3], focus: Point3) -> Option<Point3> {
    points
        .iter()
        .copied()
        .min_by(|a, b| distance_3d(*a, focus).total_cmp(&distance_3d(*b, focus)))
}
