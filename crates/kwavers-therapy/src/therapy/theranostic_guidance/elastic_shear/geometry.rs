//! Spatial geometry helpers: coordinate mapping, masking, and distance.

use kwavers_core::error::{KwaversError, KwaversResult};
use ndarray::{Array2, Array3};

use super::super::geometry::Point2;

pub(super) fn source_mask(
    target_mask: &Array2<bool>,
    spacing_m: f64,
    focus_m: Point2,
) -> KwaversResult<Array2<bool>> {
    let dims = target_mask.dim();
    let Some((source_ix, source_iy)) = target_mask
        .indexed_iter()
        .filter_map(|((ix, iy), active)| active.then_some((ix, iy)))
        .min_by(|a, b| {
            let da = distance_sq(index_point_m(a.0, a.1, dims, spacing_m), focus_m);
            let db = distance_sq(index_point_m(b.0, b.1, dims, spacing_m), focus_m);
            da.total_cmp(&db)
        })
    else {
        return Err(KwaversError::InvalidInput(
            "elastic shear source requires nonempty target support".to_owned(),
        ));
    };
    Ok(Array2::from_shape_fn(dims, |idx| {
        idx == (source_ix, source_iy)
    }))
}

pub(super) fn mask_points_m(mask: &Array2<bool>, spacing_m: f64) -> KwaversResult<Vec<Point2>> {
    let dims = mask.dim();
    let points = mask
        .indexed_iter()
        .filter_map(|((ix, iy), active)| active.then_some(index_point_m(ix, iy, dims, spacing_m)))
        .collect::<Vec<_>>();
    if points.is_empty() {
        return Err(KwaversError::InvalidInput(
            "elastic shear source mask is empty".to_owned(),
        ));
    }
    Ok(points)
}

pub(super) fn index_point_m(ix: usize, iy: usize, dims: (usize, usize), spacing_m: f64) -> Point2 {
    Point2 {
        x_m: (ix as f64 - 0.5 * (dims.0 - 1) as f64) * spacing_m,
        y_m: (iy as f64 - 0.5 * (dims.1 - 1) as f64) * spacing_m,
    }
}

pub(super) fn mask_indices(mask: &Array3<bool>) -> Vec<(usize, usize, usize)> {
    mask.indexed_iter()
        .filter_map(|(idx, active)| active.then_some(idx))
        .collect()
}

pub(super) fn distance(a: Point2, b: Point2) -> f64 {
    distance_sq(a, b).sqrt()
}

pub(super) fn distance_sq(a: Point2, b: Point2) -> f64 {
    (a.x_m - b.x_m).powi(2) + (a.y_m - b.y_m).powi(2)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn source_mask_selects_nearest_target_focus() {
        let target = Array2::from_shape_vec((2, 2), vec![true, true, true, false]).unwrap();

        let mask = source_mask(
            &target,
            1.0,
            Point2 {
                x_m: 0.5,
                y_m: -0.5,
            },
        )
        .unwrap();

        assert!(!mask[[0, 0]]);
        assert!(!mask[[0, 1]]);
        assert!(mask[[1, 0]]);
        assert!(!mask[[1, 1]]);
    }
}
