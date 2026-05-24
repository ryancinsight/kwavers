//! Focused bowl element placement through the source-domain bowl transducer.

use super::super::geometry::Point3;
use crate::{
    core::error::KwaversResult,
    domain::source::transducers::focused::{BowlAngularBounds, BowlConfig, BowlTransducer},
};

pub(crate) const BOWL_THETA_CUTOUT_RAD: f64 = 0.175;
pub(crate) const BOWL_THETA_MAX_RAD: f64 = 0.960;
const BOWL_GEOMETRY_FREQUENCY_HZ: f64 = 1.0;
const BOWL_GEOMETRY_AMPLITUDE_PA: f64 = 1.0;

/// Distribute `count` elements on a spherical-cap bowl.
///
/// `skin_contact_m` fixes the aperture axis, while `radius_m` controls the
/// actual curvature radius. These differ when the planner chooses a larger
/// radius so the outer rim remains outside the body. Element placement,
/// angular validation, normals, and equal-area weights are delegated to
/// [`BowlTransducer`].
///
/// # Theorem
///
/// Let `A` be `skin_contact_m`, `F` be `focus_m`, and
/// `d = normalize(F - A)`. `BowlConfig::from_axis_reference_focus` constructs
/// the source vertex `V = F - radius_m d`. Delegating to
/// [`BowlTransducer::with_angular_bounds`] then gives every element position
/// `P` the invariant `||P - F|| = radius_m` with polar coverage
/// `[BOWL_THETA_CUTOUT_RAD, BOWL_THETA_MAX_RAD]`.
pub(crate) fn bowl_elements(
    count: usize,
    skin_contact_m: Point3,
    focus_m: Point3,
    radius_m: f64,
) -> KwaversResult<Vec<Point3>> {
    let config = BowlConfig::from_axis_reference_focus(
        [skin_contact_m.x_m, skin_contact_m.y_m, skin_contact_m.z_m],
        [focus_m.x_m, focus_m.y_m, focus_m.z_m],
        radius_m,
        BOWL_GEOMETRY_FREQUENCY_HZ,
        BOWL_GEOMETRY_AMPLITUDE_PA,
    )?;
    let bounds = BowlAngularBounds::new(BOWL_THETA_CUTOUT_RAD, BOWL_THETA_MAX_RAD)?;
    let bowl = BowlTransducer::with_angular_bounds(config, bounds, count)?;

    Ok(bowl
        .element_positions()
        .iter()
        .map(|position| Point3 {
            x_m: position[0],
            y_m: position[1],
            z_m: position[2],
        })
        .collect())
}
