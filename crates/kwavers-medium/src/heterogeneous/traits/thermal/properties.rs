//! Thermal properties implementation for heterogeneous media
//!
//! **Domain Focus**: Pure thermal behavior following Domain-Driven Design
//! Evidence-based thermal modeling per Hamilton & Blackstock Ch.8

use crate::{
    heterogeneous::core::HeterogeneousMedium,
    thermal::{ThermalField, ThermalProperties},
};
use aequitas::systems::si::quantities::{
    ReciprocalTemperature, SpecificHeatCapacity, ThermalConductivity, ThermalDiffusivity,
};
use kwavers_core::constants::tissue_thermal::THERMAL_EXPANSION_SOFT_TISSUE;
use kwavers_grid::Grid;
use leto::Array3;

impl ThermalProperties for HeterogeneousMedium {
    /// Thermal conductivity at continuous coordinates (W/(m·K))
    #[inline]
    fn thermal_conductivity(&self, x: f64, y: f64, z: f64, grid: &Grid) -> ThermalConductivity {
        ThermalConductivity::from_base(self.lookup_field(&self.thermal_conductivity, x, y, z, grid))
    }

    /// Specific heat at constant volume (J/(kg·K)) at continuous coordinates
    #[inline]
    fn specific_heat(&self, x: f64, y: f64, z: f64, grid: &Grid) -> SpecificHeatCapacity {
        SpecificHeatCapacity::from_base(self.lookup_field(&self.specific_heat, x, y, z, grid))
    }

    /// The stored specific heat is Cp (it feeds thermal diffusivity
    /// α = κ/(ρCp)); without this override Cp would fall back to the trait's
    /// water constant and could violate Cp ≥ Cv (Mayer). Liquids: Cv ≈ Cp,
    /// so γ = Cp/Cv = 1 by the trait defaults.
    fn specific_heat_capacity(&self, x: f64, y: f64, z: f64, grid: &Grid) -> SpecificHeatCapacity {
        self.specific_heat(x, y, z, grid)
    }

    /// Thermal diffusivity at continuous coordinates (m²/s)
    #[inline]
    fn thermal_diffusivity(&self, x: f64, y: f64, z: f64, grid: &Grid) -> ThermalDiffusivity {
        ThermalDiffusivity::from_base(self.lookup_field(&self.thermal_diffusivity, x, y, z, grid))
    }

    /// Thermal expansion coefficient (1/K) at continuous coordinates
    #[inline]
    fn thermal_expansion(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> ReciprocalTemperature {
        ReciprocalTemperature::from_base(THERMAL_EXPANSION_SOFT_TISSUE)
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
