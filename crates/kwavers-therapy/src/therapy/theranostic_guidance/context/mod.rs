//! CT-volume placement context for same-device therapy/imaging figures.
//!
//! The inverse solve in this chapter is a bounded 2-D finite-frequency slice
//! solve. This vertical owns the separate full-CT placement context used to
//! audit clinical geometry: a calvarium-only 3-D focused bowl for transcranial arrays
//! and uncropped abdominal slices for skin-coupled histotripsy heads.

mod abdomen;
mod brain;
mod surface;

use kwavers_core::error::{KwaversError, KwaversResult};
use leto::{
    Array2,
    Array3,
};

pub(super) use super::geometry::{centered_origin_2d, IndexBounds3, Point3};
pub use abdomen::build_abdominal_placement_context;
pub use brain::build_brain_placement_context;

#[derive(Clone, Debug)]
pub struct PlacementContext {
    pub ct_hu: Array2<f64>,
    pub body_mask: Array2<bool>,
    pub target_mask: Array2<bool>,
    pub spacing_x_m: f64,
    pub spacing_y_m: f64,
    pub slice_index: usize,
    pub therapy_points_m: Vec<Point3>,
    pub imaging_points_m: Vec<Point3>,
    pub body_surface_points_m: Vec<Point3>,
    pub focus_m: Point3,
    pub skin_contact_m: Point3,
    pub model_name: String,
}

pub(super) fn validate_spacing(spacing_mm: [f64; 3]) -> KwaversResult<()> {
    if spacing_mm
        .iter()
        .any(|spacing| !spacing.is_finite() || *spacing <= 0.0)
    {
        return Err(KwaversError::InvalidInput(
            "placement context spacing must be positive and finite".to_owned(),
        ));
    }
    Ok(())
}

pub(super) fn volume_center(
    _nx: usize,
    _ny: usize,
    _nz: usize,
    _sx: f64,
    _sy: f64,
    _sz: f64,
) -> Point3 {
    Point3 {
        x_m: 0.0,
        y_m: 0.0,
        z_m: 0.0,
    }
}

pub(super) fn distance_3d(a: Point3, b: Point3) -> f64 {
    ((a.x_m - b.x_m).powi(2) + (a.y_m - b.y_m).powi(2) + (a.z_m - b.z_m).powi(2)).sqrt()
}

pub(super) fn centroid_2d(mask: &Array2<bool>, sx: f64, sy: f64) -> Option<Point3> {
    let (nx, ny) = mask.dim();
    let center = centered_origin_2d(nx, ny);
    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    let mut count = 0.0;
    for ((ix, iy), active) in mask.indexed_iter() {
        if *active {
            sum_x += (ix as f64 - center.0) * sx;
            sum_y += (iy as f64 - center.1) * sy;
            count += 1.0;
        }
    }
    (count > 0.0).then_some(Point3 {
        x_m: sum_x / count,
        y_m: sum_y / count,
        z_m: 0.0,
    })
}

pub(super) fn centroid_index(mask: &Array2<bool>) -> Option<(usize, usize)> {
    let mut sx = 0.0;
    let mut sy = 0.0;
    let mut count = 0.0;
    for ((ix, iy), active) in mask.indexed_iter() {
        if *active {
            sx += ix as f64;
            sy += iy as f64;
            count += 1.0;
        }
    }
    (count > 0.0).then_some(((sx / count).round() as usize, (sy / count).round() as usize))
}

pub(super) fn volume_bbox(mask: &Array3<bool>) -> KwaversResult<IndexBounds3> {
    let mut b = IndexBounds3 {
        x0: usize::MAX,
        x1: 0,
        y0: usize::MAX,
        y1: 0,
        z0: usize::MAX,
        z1: 0,
    };
    let mut any = false;
    for ((ix, iy, iz), active) in mask.indexed_iter() {
        if *active {
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
        .ok_or_else(|| KwaversError::InvalidInput("placement body volume is empty".to_owned()))
}
