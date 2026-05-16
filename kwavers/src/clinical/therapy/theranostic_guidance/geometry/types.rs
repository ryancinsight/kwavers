use crate::solver::inverse::same_aperture::PlanarPoint;

pub type Point2 = PlanarPoint;

/// 3-D point in physical space [m].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Point3 {
    pub x_m: f64,
    pub y_m: f64,
    pub z_m: f64,
}

/// 3-D inclusive bounding box in voxel index space.
#[derive(Clone, Copy, Debug)]
pub(crate) struct IndexBounds3 {
    pub x0: usize,
    pub x1: usize,
    pub y0: usize,
    pub y1: usize,
    pub z0: usize,
    pub z1: usize,
}
