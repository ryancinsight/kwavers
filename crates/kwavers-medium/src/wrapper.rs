//! Continuous-coordinate access for [`CoreMedium`] properties.

use crate::{continuous_to_discrete, CoreMedium};
use aequitas::systems::si::quantities::{Dimensionless, MassDensity, ReciprocalLength, Velocity};
use kwavers_grid::Grid;

/// Get density at continuous coordinates, in kilograms per cubic metre.
#[inline]
pub fn density_at<M: CoreMedium + ?Sized>(
    medium: &M,
    x: f64,
    y: f64,
    z: f64,
    grid: &Grid,
) -> MassDensity<f64> {
    let (i, j, k) = continuous_to_discrete(x, y, z, grid);
    medium.density(i, j, k)
}

/// Get sound speed at continuous coordinates, in metres per second.
#[inline]
pub fn sound_speed_at<M: CoreMedium + ?Sized>(
    medium: &M,
    x: f64,
    y: f64,
    z: f64,
    grid: &Grid,
) -> Velocity<f64> {
    let (i, j, k) = continuous_to_discrete(x, y, z, grid);
    medium.sound_speed(i, j, k)
}

/// Get absorption at continuous coordinates, in nepers per metre.
#[inline]
pub fn absorption_at<M: CoreMedium + ?Sized>(
    medium: &M,
    x: f64,
    y: f64,
    z: f64,
    grid: &Grid,
) -> ReciprocalLength<f64> {
    let (i, j, k) = continuous_to_discrete(x, y, z, grid);
    medium.absorption(i, j, k)
}

/// Get nonlinearity at continuous coordinates (dimensionless B/A).
#[inline]
pub fn nonlinearity_at<M: CoreMedium + ?Sized>(
    medium: &M,
    x: f64,
    y: f64,
    z: f64,
    grid: &Grid,
) -> Dimensionless<f64> {
    let (i, j, k) = continuous_to_discrete(x, y, z, grid);
    medium.nonlinearity(i, j, k)
}
