//! Thermal properties implementation for heterogeneous media
//!
//! **Domain Focus**: Pure thermal behavior following Domain-Driven Design
//! Evidence-based thermal modeling per Hamilton & Blackstock Ch.8

use crate::grid::Grid;
use crate::medium::{
    heterogeneous::{core::HeterogeneousMedium, interpolation::TrilinearInterpolator},
    thermal::{ThermalField, ThermalProperties},
};
use ndarray::Array3;

impl ThermalProperties for HeterogeneousMedium {
    /// Thermal conductivity at continuous coordinates (W/(m·K))
    #[inline]
    fn thermal_conductivity(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        TrilinearInterpolator::get_field_value(
            &self.thermal_conductivity,
            x,
            y,
            z,
            grid,
            self.use_trilinear_interpolation,
        )
    }

    /// Specific heat at constant volume (J/(kg·K)) at continuous coordinates
    #[inline]
    fn specific_heat(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        TrilinearInterpolator::get_field_value(
            &self.specific_heat,
            x,
            y,
            z,
            grid,
            self.use_trilinear_interpolation,
        )
    }

    /// Thermal diffusivity at continuous coordinates (m²/s)
    #[inline]
    fn thermal_diffusivity(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        TrilinearInterpolator::get_field_value(
            &self.thermal_diffusivity,
            x,
            y,
            z,
            grid,
            self.use_trilinear_interpolation,
        )
    }

    /// Thermal expansion coefficient (1/K) at continuous coordinates
    #[inline]
    fn thermal_expansion(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 {
        // Typical soft tissue thermal expansion ~3e-4 1/K
        3.0e-4
    }
}

impl ThermalField for HeterogeneousMedium {
    /// Update the thermal field
    fn update_thermal_field(&mut self, temperature: &Array3<f64>) {
        self.temperature = temperature.clone();
    }

    /// Get temperature field array reference (zero-copy)
    fn thermal_field(&self) -> &Array3<f64> {
        &self.temperature
    }
}
