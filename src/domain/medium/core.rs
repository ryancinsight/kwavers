//! Core medium trait definitions for acoustic simulations
//!
//! This module provides fundamental traits that all medium types must implement,
//! establishing a Single Source of Truth (SSOT) for medium behaviors.

use crate::domain::core::error::KwaversResult;
use crate::domain::grid::Grid;
use ndarray::{Array3, ArrayView3, ArrayViewMut3};
use std::fmt::Debug;

/// Minimum physical density to prevent numerical instabilities (kg/m³)
pub const MIN_PHYSICAL_DENSITY: f64 = 1.0;

/// Minimum physical sound speed to prevent numerical instabilities (m/s)
pub const MIN_PHYSICAL_SOUND_SPEED: f64 = 1.0;

/// Core trait that all medium types must implement
///
/// Provides the fundamental interface for accessing acoustic properties
/// in a spatially-varying medium. Follows SOLID principles by defining
/// a focused interface for medium property access.
pub trait CoreMedium: Debug + Send + Sync {
    /// Get sound speed at a specific grid point
    fn sound_speed(&self, i: usize, j: usize, k: usize) -> f64;

    /// Get density at a specific grid point
    fn density(&self, i: usize, j: usize, k: usize) -> f64;

    /// Get absorption coefficient at a specific grid point
    fn absorption(&self, i: usize, j: usize, k: usize) -> f64;

    /// Get nonlinearity parameter B/A at a specific grid point
    fn nonlinearity(&self, i: usize, j: usize, k: usize) -> f64;

    /// Get maximum sound speed in the medium
    fn max_sound_speed(&self) -> f64;

    /// Check if the medium is homogeneous
    fn is_homogeneous(&self) -> bool;

    /// Get reference frequency for the medium (used for absorption calculations)
    fn reference_frequency(&self) -> f64 {
        1e6 // Default 1 MHz
    }

    /// Validate medium properties against physical constraints
    fn validate(&self, grid: &Grid) -> KwaversResult<()>;
}

/// Trait for array-based access to medium properties
///
/// Provides efficient batch access to medium properties through array views.
/// Implements zero-copy principles for performance optimization.
pub trait ArrayAccess: CoreMedium {
    /// Get sound speed as an array view
    fn sound_speed_array(&self) -> ArrayView3<'_, f64>;

    /// Get density as an array view
    fn density_array(&self) -> ArrayView3<'_, f64>;

    /// Get absorption as an array view
    fn absorption_array(&self) -> ArrayView3<'_, f64>;

    /// Get nonlinearity as an array view
    fn nonlinearity_array(&self) -> ArrayView3<'_, f64>;

    /// Get mutable sound speed array (for heterogeneous media)
    fn sound_speed_array_mut(&mut self) -> Option<ArrayViewMut3<'_, f64>> {
        None
    }

    /// Get mutable density array (for heterogeneous media)
    fn density_array_mut(&mut self) -> Option<ArrayViewMut3<'_, f64>> {
        None
    }

    /// Get mutable absorption array (for heterogeneous media)
    fn absorption_array_mut(&mut self) -> Option<ArrayViewMut3<'_, f64>> {
        None
    }

    /// Get mutable nonlinearity array (for heterogeneous media)
    fn nonlinearity_array_mut(&mut self) -> Option<ArrayViewMut3<'_, f64>> {
        None
    }
}

/// Calculate maximum sound speed from a 3D array
///
/// Efficiently finds the maximum value using iterator combinators
/// for potential SIMD optimization by the compiler.
#[must_use]
pub fn max_sound_speed(sound_speed: &Array3<f64>) -> f64 {
    sound_speed
        .iter()
        .fold(f64::NEG_INFINITY, |max, &val| max.max(val))
}

/// Calculate maximum sound speed with custom point-wise accessor
///
/// Provides flexibility for media that compute properties on-demand
/// rather than storing them in arrays.
pub fn max_sound_speed_pointwise<F>(nx: usize, ny: usize, nz: usize, accessor: F) -> f64
where
    F: Fn(usize, usize, usize) -> f64,
{
    let mut max_speed = f64::NEG_INFINITY;

    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                max_speed = max_speed.max(accessor(i, j, k));
            }
        }
    }

    max_speed
}

/// Convert continuous coordinates to discrete grid indices
///
/// # Arguments
/// * `x` - Continuous x coordinate
/// * `y` - Continuous y coordinate  
/// * `z` - Continuous z coordinate
/// * `grid` - Grid structure containing spacing and dimensions
///
/// # Returns
/// Tuple of (i, j, k) discrete indices
#[inline]
pub fn continuous_to_discrete(x: f64, y: f64, z: f64, grid: &Grid) -> (usize, usize, usize) {
    let i = ((x / grid.dx).round() as usize).min(grid.nx - 1);
    let j = ((y / grid.dy).round() as usize).min(grid.ny - 1);
    let k = ((z / grid.dz).round() as usize).min(grid.nz - 1);
    (i, j, k)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_max_sound_speed() {
        let mut data = Array3::zeros((2, 2, 2));
        data[(0, 0, 0)] = 1500.0;
        data[(1, 1, 1)] = 1600.0;

        assert_eq!(max_sound_speed(&data), 1600.0);
    }

    #[test]
    fn test_max_sound_speed_pointwise() {
        let accessor = |i: usize, j: usize, k: usize| {
            if i == 1 && j == 1 && k == 1 {
                1600.0
            } else {
                1500.0
            }
        };

        assert_eq!(max_sound_speed_pointwise(2, 2, 2, accessor), 1600.0);
    }
}
