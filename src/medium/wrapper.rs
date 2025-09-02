//! Wrapper functions for CoreMedium trait to handle continuous coordinates
//!
//! This module provides compatibility functions that convert continuous
//! coordinates to discrete indices before calling CoreMedium methods.

use crate::grid::Grid;
use crate::medium::{continuous_to_discrete, CoreMedium, Medium};

/// Get density at continuous coordinates
#[inline]
pub fn density_at(medium: &dyn Medium, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
    let (i, j, k) = continuous_to_discrete(x, y, z, grid);
    medium.density(i, j, k)
}

/// Get sound speed at continuous coordinates
#[inline]
pub fn sound_speed_at(medium: &dyn Medium, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
    let (i, j, k) = continuous_to_discrete(x, y, z, grid);
    medium.sound_speed(i, j, k)
}

/// Get absorption at continuous coordinates
#[inline]
pub fn absorption_at(medium: &dyn Medium, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
    let (i, j, k) = continuous_to_discrete(x, y, z, grid);
    medium.absorption(i, j, k)
}

/// Get nonlinearity at continuous coordinates
#[inline]
pub fn nonlinearity_at(medium: &dyn Medium, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
    let (i, j, k) = continuous_to_discrete(x, y, z, grid);
    medium.nonlinearity(i, j, k)
}

// Generic versions that work with any CoreMedium

/// Get density at continuous coordinates (generic version)
#[inline]
pub fn density_at_core<M: CoreMedium + ?Sized>(medium: &M, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
    let (i, j, k) = continuous_to_discrete(x, y, z, grid);
    medium.density(i, j, k)
}

/// Get sound speed at continuous coordinates (generic version)
#[inline]
pub fn sound_speed_at_core<M: CoreMedium + ?Sized>(medium: &M, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
    let (i, j, k) = continuous_to_discrete(x, y, z, grid);
    medium.sound_speed(i, j, k)
}

/// Get absorption at continuous coordinates (generic version)
#[inline]
pub fn absorption_at_core<M: CoreMedium + ?Sized>(medium: &M, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
    let (i, j, k) = continuous_to_discrete(x, y, z, grid);
    medium.absorption(i, j, k)
}

/// Get nonlinearity at continuous coordinates (generic version)
#[inline]
pub fn nonlinearity_at_core<M: CoreMedium + ?Sized>(medium: &M, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
    let (i, j, k) = continuous_to_discrete(x, y, z, grid);
    medium.nonlinearity(i, j, k)
}