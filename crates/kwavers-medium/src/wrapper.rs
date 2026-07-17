//! Continuous-coordinate access for [`CoreMedium`] properties.

use crate::{continuous_to_discrete, CoreMedium};
use kwavers_grid::Grid;

/// Get density at continuous coordinates.
#[inline]
pub fn density_at<M: CoreMedium + ?Sized>(medium: &M, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
    let (i, j, k) = continuous_to_discrete(x, y, z, grid);
    medium.density(i, j, k)
}

/// Get sound speed at continuous coordinates.
#[inline]
pub fn sound_speed_at<M: CoreMedium + ?Sized>(
    medium: &M,
    x: f64,
    y: f64,
    z: f64,
    grid: &Grid,
) -> f64 {
    let (i, j, k) = continuous_to_discrete(x, y, z, grid);
    medium.sound_speed(i, j, k)
}

/// Get absorption at continuous coordinates.
#[inline]
pub fn absorption_at<M: CoreMedium + ?Sized>(
    medium: &M,
    x: f64,
    y: f64,
    z: f64,
    grid: &Grid,
) -> f64 {
    let (i, j, k) = continuous_to_discrete(x, y, z, grid);
    medium.absorption(i, j, k)
}

/// Get nonlinearity at continuous coordinates.
#[inline]
pub fn nonlinearity_at<M: CoreMedium + ?Sized>(
    medium: &M,
    x: f64,
    y: f64,
    z: f64,
    grid: &Grid,
) -> f64 {
    let (i, j, k) = continuous_to_discrete(x, y, z, grid);
    medium.nonlinearity(i, j, k)
}
