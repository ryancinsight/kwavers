//! Thermal properties implementation for heterogeneous media
//!
//! **Domain Focus**: Pure thermal behavior following Domain-Driven Design
//! Evidence-based thermal modeling per Hamilton & Blackstock Ch.8

use crate::grid::Grid;
use crate::medium::{
    heterogeneous::{core::HeterogeneousMedium, interpolation::TrilinearInterpolator},
    thermal::{ThermalField, ThermalProperties},
};

impl ThermalProperties for HeterogeneousMedium {
    /// Get thermal conductivity at grid point
    #[inline]
    fn thermal_conductivity(&self, i: usize, j: usize, k: usize) -> f64 {
        self.thermal_conductivity[[i, j, k]]
    }

    /// Get specific heat at grid point
    #[inline]
    fn specific_heat(&self, i: usize, j: usize, k: usize) -> f64 {
        self.specific_heat[[i, j, k]]
    }

    /// Get thermal diffusivity at continuous coordinates
    fn thermal_diffusivity_at(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        TrilinearInterpolator::get_field_value(
            &self.thermal_diffusivity,
            x, y, z,
            grid,
            self.use_trilinear_interpolation
        )
    }
}

impl ThermalField for HeterogeneousMedium {
    /// Get temperature field array reference (zero-copy)
    ///
    /// **Performance**: Zero-copy access per TSE 2025 efficiency standards
    fn temperature_field(&self) -> &ndarray::Array3<f64> {
        &self.temperature
    }
}