//! Focused bowl element placement through the source-domain bowl transducer.

use super::super::geometry::Point3;
use aequitas::systems::si::quantities::{Angle, Frequency, Length, Pressure};
use aequitas::systems::si::units::{Hertz, Meter, Pascal, Radian};
use kwavers_core::error::KwaversResult;
use kwavers_transducer::transducers::focused::{BowlAngularBounds, BowlConfig, BowlTransducer};

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
    // Chord diameter of the active spherical cap:
    //   aperture = 2 R sin(theta_max), where theta_max = BOWL_THETA_MAX_RAD.
    // This is the correct cap aperture, not the hemisphere diameter (2R).
    let aperture_diameter_m = 2.0 * radius_m * BOWL_THETA_MAX_RAD.sin();
    let config = BowlConfig::from_axis_reference_focus(
        [
            Length::from_unit::<Meter>(skin_contact_m.x_m),
            Length::from_unit::<Meter>(skin_contact_m.y_m),
            Length::from_unit::<Meter>(skin_contact_m.z_m),
        ],
        [
            Length::from_unit::<Meter>(focus_m.x_m),
            Length::from_unit::<Meter>(focus_m.y_m),
            Length::from_unit::<Meter>(focus_m.z_m),
        ],
        Length::from_unit::<Meter>(radius_m),
        Length::from_unit::<Meter>(aperture_diameter_m),
        Frequency::from_unit::<Hertz>(BOWL_GEOMETRY_FREQUENCY_HZ),
        Pressure::from_unit::<Pascal>(BOWL_GEOMETRY_AMPLITUDE_PA),
    )?;
    let bounds = BowlAngularBounds::new(
        Angle::from_unit::<Radian>(BOWL_THETA_CUTOUT_RAD),
        Angle::from_unit::<Radian>(BOWL_THETA_MAX_RAD),
    )?;
    let bowl = BowlTransducer::with_angular_bounds(config, bounds, count)?;

    Ok(bowl
        .element_positions()
        .iter()
        .map(|position| Point3 {
            x_m: position[0].in_unit::<Meter>(),
            y_m: position[1].in_unit::<Meter>(),
            z_m: position[2].in_unit::<Meter>(),
        })
        .collect())
}
