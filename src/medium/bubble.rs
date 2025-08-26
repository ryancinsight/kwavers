//! Bubble dynamics properties trait
//!
//! This module defines traits for bubble-related properties including surface tension,
//! vapor pressure, and bubble state management.

use crate::grid::Grid;
use crate::medium::core::CoreMedium;
use ndarray::Array3;

/// Trait for bubble dynamics properties
pub trait BubbleProperties: CoreMedium {
    /// Get surface tension (N/m)
    fn surface_tension(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64;

    /// Get ambient pressure (Pa)
    fn ambient_pressure(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64;

    /// Get vapor pressure (Pa)
    fn vapor_pressure(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64;

    /// Get polytropic index for gas inside bubbles
    fn polytropic_index(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64;

    /// Get gas diffusion coefficient (m²/s)
    fn gas_diffusion_coefficient(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64;
}

/// Trait for bubble state management
pub trait BubbleState: BubbleProperties {
    /// Get current bubble radius field
    fn bubble_radius(&self) -> &Array3<f64>;

    /// Get current bubble velocity field
    fn bubble_velocity(&self) -> &Array3<f64>;

    /// Update bubble state with new radius and velocity fields
    fn update_bubble_state(&mut self, radius: &Array3<f64>, velocity: &Array3<f64>);
}
