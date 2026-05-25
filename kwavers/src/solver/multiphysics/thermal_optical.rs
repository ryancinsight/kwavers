//! Thermal-optical coupling module
//!
//! This module provides specialized coupling between thermal and optical fields.

use crate::core::constants::fundamental::DENSITY_WATER_NOMINAL;
use crate::core::constants::thermodynamic::SPECIFIC_HEAT_WATER;
use crate::core::error::KwaversResult;
use crate::domain::grid::Grid;
use ndarray::Array3;

/// Thermal-optical solver for coupled simulations
#[derive(Debug)]
pub struct ThermalOpticalSolver {
    /// Absorption coefficient (m^-1)
    absorption_coefficient: f64,
    /// Grid reference
    _grid: Grid,
}

impl ThermalOpticalSolver {
    /// Create a new thermal-optical solver
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn new(grid: Grid, absorption_coefficient: f64) -> Self {
        Self {
            absorption_coefficient,
            _grid: grid,
        }
    }

    /// Couple optical intensity to temperature
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn couple_fields(
        &self,
        intensity: &Array3<f64>,
        temperature: &mut Array3<f64>,
        dt: f64,
    ) -> KwaversResult<()> {
        // Optical absorption generates heat
        let rho = DENSITY_WATER_NOMINAL; // kg/m³ — nominal water density (1000 kg/m³)
        let c = SPECIFIC_HEAT_WATER;     // J/(kg·K) — water isobaric specific heat at 20°C

        for ((i, j, k), &i_val) in intensity.indexed_iter() {
            // Heat generated per unit volume (W/m³)
            let heat_source = self.absorption_coefficient * i_val;
            // Temperature change
            let delta_t = heat_source * dt / (rho * c);
            temperature[[i, j, k]] += delta_t;
        }

        Ok(())
    }
}
