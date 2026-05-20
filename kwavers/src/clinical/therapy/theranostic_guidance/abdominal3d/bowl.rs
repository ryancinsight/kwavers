//! Focused bowl element placement through the source-domain spherical cap.

use super::super::geometry::Point3;
use crate::{
    core::error::KwaversResult,
    domain::source::transducers::focused::{SphericalCapConfig, SphericalCapLayout},
};

pub(crate) const BOWL_THETA_CUTOUT_RAD: f64 = 0.175;
pub(crate) const BOWL_THETA_MAX_RAD: f64 = 0.960;

/// Distribute `count` elements on a spherical-cap bowl.
///
/// The bowl is a spherical cap of radius `radius_m` centred at `focus_m`.
/// The cap vertex is `skin_contact_m` (the bowl vertex touching the skin).
/// Elements are placed at polar angles θ ∈ [θ_cutout, θ_max] using the
/// golden-spiral (Fibonacci) method for uniform area density.
///
/// Area-weighted sampling: the spherical-cap area element is `sin(θ) dθ dφ`.
/// Uniform area sampling requires the CDF `F(θ) = (cos(θ_cutout) − cos(θ)) /
/// (cos(θ_cutout) − cos(θ_max))` to be uniformly distributed, so
/// `cos(θ) = cos(θ_cutout) − t·(cos(θ_cutout) − cos(θ_max))`.
pub(crate) fn bowl_elements(
    count: usize,
    skin_contact_m: Point3,
    focus_m: Point3,
    radius_m: f64,
) -> KwaversResult<Vec<Point3>> {
    let layout = SphericalCapLayout::new(SphericalCapConfig::from_vertex_focus(
        count,
        radius_m,
        [skin_contact_m.x_m, skin_contact_m.y_m, skin_contact_m.z_m],
        [focus_m.x_m, focus_m.y_m, focus_m.z_m],
        BOWL_THETA_CUTOUT_RAD,
        BOWL_THETA_MAX_RAD,
    ))?;

    Ok(layout
        .positions()
        .map(|position| Point3 {
            x_m: position[0],
            y_m: position[1],
            z_m: position[2],
        })
        .collect())
}
