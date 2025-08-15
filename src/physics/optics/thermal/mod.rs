//! Legacy optical thermal module - DEPRECATED
//!
//! **DEPRECATION NOTICE**: This module is deprecated as of v2.18.0.
//! Please use `physics::thermal` instead, which provides a unified
//! interface for all thermal phenomena including optical heating.
//!
//! Migration guide:
//! - `OpticalThermalModel` → `thermal::ThermalCalculator` with `HeatSource::Optical`

#[deprecated(since = "2.18.0", note = "Use physics::thermal::ThermalCalculator with HeatSource::Optical")]
use crate::grid::Grid;
use crate::medium::Medium;
use crate::physics::field_indices::TEMPERATURE_IDX;
use log::debug;
use ndarray::{Array3, Array4, Axis, Zip};

#[derive(Debug, Clone)]
pub struct OpticalThermalModel {
    temperature_contribution: Array3<f64>, // Optical contribution to temperature field
}

impl OpticalThermalModel {
    pub fn new(grid: &Grid) -> Self {
        debug!("Initializing OpticalThermalModel");
        Self {
            temperature_contribution: Array3::zeros((grid.nx, grid.ny, grid.nz)),
        }
    }

    pub fn update_thermal(
        &mut self,
        fields: &mut Array4<f64>,
        fluence: &Array3<f64>,
        grid: &Grid,
        medium: &dyn Medium,
        dt: f64,
    ) {
        debug!("Updating optical-thermal effects");

        // Calculate heat source from light absorption
        Zip::indexed(&mut self.temperature_contribution)
            .and(fluence)
            .for_each(|(i, j, k), t_contrib, &f_val| {
                let x = i as f64 * grid.dx;
                let y = j as f64 * grid.dy;
                let z = k as f64 * grid.dz;
                let mu_a = medium.optical_absorption_coefficient(x, y, z, grid);
                let rho = medium.density(x, y, z, grid);
                let cp = medium.specific_heat(x, y, z, grid);
                *t_contrib = mu_a * f_val.max(0.0) * dt / (rho * cp); // Heat from light absorption
            });

        // Add to existing temperature field
        let mut temp_field = fields.index_axis(Axis(0), TEMPERATURE_IDX).to_owned();
        Zip::from(&mut temp_field)
            .and(&self.temperature_contribution)
            .for_each(|t, &t_contrib| {
                *t += t_contrib;
                if t.is_nan() || t.is_infinite() {
                    *t = 0.0;
                }
            });

        fields.index_axis_mut(Axis(0), TEMPERATURE_IDX).assign(&temp_field);
    }

    pub fn temperature_contribution(&self) -> &Array3<f64> {
        &self.temperature_contribution
    }
}