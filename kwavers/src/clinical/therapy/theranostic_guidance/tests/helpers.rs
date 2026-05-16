use ndarray::Array2;
use std::collections::VecDeque;

use super::super::{Point2, Point3};

pub(super) fn connected_mask_components(mask: &Array2<bool>) -> usize {
    let (nx, ny) = mask.dim();
    let mut visited = Array2::<bool>::from_elem((nx, ny), false);
    let mut components = 0;
    for ix in 0..nx {
        for iy in 0..ny {
            if !mask[[ix, iy]] || visited[[ix, iy]] {
                continue;
            }
            components += 1;
            let mut queue = VecDeque::from([(ix, iy)]);
            visited[[ix, iy]] = true;
            while let Some((x, y)) = queue.pop_front() {
                for (next_x, next_y) in mask_neighbors(x, y, nx, ny) {
                    if mask[[next_x, next_y]] && !visited[[next_x, next_y]] {
                        visited[[next_x, next_y]] = true;
                        queue.push_back((next_x, next_y));
                    }
                }
            }
        }
    }
    components
}

pub(super) fn nearest_mask_distance_m(mask: &Array2<bool>, spacing_m: f64, point: Point2) -> f64 {
    let (nx, ny) = mask.dim();
    let cx = (nx - 1) as f64 * 0.5;
    let cy = (ny - 1) as f64 * 0.5;
    mask.indexed_iter()
        .filter_map(|((ix, iy), active)| {
            active.then(|| {
                let x_m = (ix as f64 - cx) * spacing_m;
                let y_m = (iy as f64 - cy) * spacing_m;
                (x_m - point.x_m).hypot(y_m - point.y_m)
            })
        })
        .fold(f64::INFINITY, f64::min)
}

pub(super) fn mask_neighbors(
    ix: usize,
    iy: usize,
    nx: usize,
    ny: usize,
) -> impl Iterator<Item = (usize, usize)> {
    let mut neighbors = [(ix, iy); 4];
    let mut count = 0;
    if ix > 0 {
        neighbors[count] = (ix - 1, iy);
        count += 1;
    }
    if iy > 0 {
        neighbors[count] = (ix, iy - 1);
        count += 1;
    }
    if ix + 1 < nx {
        neighbors[count] = (ix + 1, iy);
        count += 1;
    }
    if iy + 1 < ny {
        neighbors[count] = (ix, iy + 1);
        count += 1;
    }
    neighbors.into_iter().take(count)
}

pub(super) fn distance_2d(a: Point2, b: Point2) -> f64 {
    (a.x_m - b.x_m).hypot(a.y_m - b.y_m)
}

pub(super) fn skin_normal_projection_2d(point: Point2, skin: Point2, focus: Point2) -> f64 {
    let depth = distance_2d(skin, focus);
    let normal_x = (focus.x_m - skin.x_m) / depth;
    let normal_y = (focus.y_m - skin.y_m) / depth;
    (point.x_m - skin.x_m) * normal_x + (point.y_m - skin.y_m) * normal_y
}

pub(super) fn skin_normal_projection_3d(point: Point3, skin: Point3, focus: Point3) -> f64 {
    let dx = focus.x_m - skin.x_m;
    let dy = focus.y_m - skin.y_m;
    let depth = dx.hypot(dy);
    let normal_x = dx / depth;
    let normal_y = dy / depth;
    (point.x_m - skin.x_m) * normal_x + (point.y_m - skin.y_m) * normal_y
}
