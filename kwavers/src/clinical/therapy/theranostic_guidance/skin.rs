//! External skin-boundary extraction for abdominal placement.

use std::collections::VecDeque;

use crate::core::error::{KwaversError, KwaversResult};
use ndarray::Array2;

use super::geometry::centered_origin_2d;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct SkinPoint2 {
    pub x_m: f64,
    pub y_m: f64,
}

/// Return the external skin point nearest to the acoustic target.
///
/// # Theorem
///
/// Let `B` be the connected body mask and let `E` be the connected component of
/// `not B` that touches the image border. A body voxel belongs to the external
/// skin if and only if it is adjacent to `E` or to the image exterior.
///
/// # Proof sketch
///
/// The complement of `B` partitions into exterior air and enclosed cavities.
/// Four-neighbor flood fill from the image border visits exactly the exterior
/// air component `E`. A body voxel adjacent to `E` has a path from that
/// adjacent air cell to infinity without crossing `B`, so it is on the patient
/// exterior. A body voxel adjacent only to non-exterior complement components
/// borders an enclosed cavity and is not valid for skin-coupled placement.
pub(crate) fn nearest_external_skin_point(
    body: &Array2<bool>,
    spacing_x_m: f64,
    spacing_y_m: f64,
    focus_x_m: f64,
    focus_y_m: f64,
) -> KwaversResult<SkinPoint2> {
    let exterior = exterior_air(body);
    let (nx, ny) = body.dim();
    let center = centered_origin_2d(nx, ny);
    let mut best = None;
    let mut best_distance = f64::INFINITY;
    for ix in 0..nx {
        for iy in 0..ny {
            if !body[[ix, iy]] || !touches_exterior(body, &exterior, ix, iy) {
                continue;
            }
            let point = SkinPoint2 {
                x_m: (ix as f64 - center.0) * spacing_x_m,
                y_m: (iy as f64 - center.1) * spacing_y_m,
            };
            let distance = (point.x_m - focus_x_m).hypot(point.y_m - focus_y_m);
            if distance < best_distance {
                best = Some(point);
                best_distance = distance;
            }
        }
    }
    best.ok_or_else(|| KwaversError::InvalidInput("external skin boundary is empty".to_owned()))
}

fn exterior_air(body: &Array2<bool>) -> Array2<bool> {
    let (nx, ny) = body.dim();
    let mut exterior = Array2::<bool>::from_elem((nx, ny), false);
    let mut queue = VecDeque::new();
    for ix in 0..nx {
        push_exterior_seed(body, &mut exterior, &mut queue, ix, 0);
        if ny > 1 {
            push_exterior_seed(body, &mut exterior, &mut queue, ix, ny - 1);
        }
    }
    for iy in 0..ny {
        push_exterior_seed(body, &mut exterior, &mut queue, 0, iy);
        if nx > 1 {
            push_exterior_seed(body, &mut exterior, &mut queue, nx - 1, iy);
        }
    }
    while let Some((ix, iy)) = queue.pop_front() {
        for (jx, jy) in neighbors(ix, iy, nx, ny) {
            if !body[[jx, jy]] && !exterior[[jx, jy]] {
                exterior[[jx, jy]] = true;
                queue.push_back((jx, jy));
            }
        }
    }
    exterior
}

fn push_exterior_seed(
    body: &Array2<bool>,
    exterior: &mut Array2<bool>,
    queue: &mut VecDeque<(usize, usize)>,
    ix: usize,
    iy: usize,
) {
    if !body[[ix, iy]] && !exterior[[ix, iy]] {
        exterior[[ix, iy]] = true;
        queue.push_back((ix, iy));
    }
}

fn touches_exterior(body: &Array2<bool>, exterior: &Array2<bool>, ix: usize, iy: usize) -> bool {
    let (nx, ny) = body.dim();
    ix == 0
        || iy == 0
        || ix + 1 == nx
        || iy + 1 == ny
        || neighbors(ix, iy, nx, ny).any(|(jx, jy)| !body[[jx, jy]] && exterior[[jx, jy]])
}

fn neighbors(ix: usize, iy: usize, nx: usize, ny: usize) -> impl Iterator<Item = (usize, usize)> {
    let mut points = [(ix, iy); 4];
    let mut count = 0;
    if ix > 0 {
        points[count] = (ix - 1, iy);
        count += 1;
    }
    if ix + 1 < nx {
        points[count] = (ix + 1, iy);
        count += 1;
    }
    if iy > 0 {
        points[count] = (ix, iy - 1);
        count += 1;
    }
    if iy + 1 < ny {
        points[count] = (ix, iy + 1);
        count += 1;
    }
    points.into_iter().take(count)
}
