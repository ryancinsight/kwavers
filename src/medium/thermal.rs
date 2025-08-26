//! Thermal properties trait for heat transfer and thermal effects
//!
//! This module defines traits for thermal properties including heat capacity,
//! thermal conductivity, and temperature-dependent effects.

use crate::grid::Grid;
use crate::medium::core::CoreMedium;
use ndarray::Array3;

/// Trait for thermal medium properties
pub trait ThermalProperties: CoreMedium {
    /// Get specific heat capacity at constant pressure (J/(kg·K))
    fn specific_heat_capacity(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 {
        4180.0 // Default for water
    }

    /// Get specific heat at constant volume (J/(kg·K))
    /// Note: This is different from specific_heat_capacity (Cp)
    fn specific_heat(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        // For liquids, Cv ≈ Cp
        self.specific_heat_capacity(x, y, z, grid)
    }

    /// Get thermal conductivity (W/(m·K))
    fn thermal_conductivity(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64;

    /// Get thermal diffusivity (m²/s)
    fn thermal_diffusivity(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64;

    /// Get thermal expansion coefficient (1/K)
    fn thermal_expansion(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64;

    /// Get specific heat ratio (γ = Cp/Cv)
    fn specific_heat_ratio(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 {
        1.1 // Default for liquids
    }

    /// Get adiabatic index (same as specific heat ratio)
    fn gamma(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        self.specific_heat_ratio(x, y, z, grid)
    }
}

/// Trait for temperature state management
pub trait TemperatureState: ThermalProperties {
    /// Update the temperature field
    fn update_temperature(&mut self, temperature: &Array3<f64>);

    /// Get the current temperature field
    fn temperature(&self) -> &Array3<f64>;
}
