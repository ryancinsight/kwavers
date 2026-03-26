//! Acoustic heating rate calculation
//!
//! ## Mathematical Foundation
//!
//! Heating rate: Q = 2α · I
//! where I = p²/(2ρc) is acoustic intensity
//!
//! ## References
//!
//! - Nyborg (1981). "Heat generation by ultrasound in a relaxing medium"

use super::plugin::TransducerFieldCalculatorPlugin;
use crate::core::error::KwaversResult;
use crate::domain::grid::Grid;
use crate::domain::medium::{AcousticProperties, Medium};
use ndarray::{Array3, Zip};

impl TransducerFieldCalculatorPlugin {
    /// Calculate heating rate from acoustic field
    pub fn calculate_heating_rate(
        &self,
        pressure_field: &Array3<f64>,
        medium: &dyn Medium,
        grid: &Grid,
        frequency: f64,
    ) -> KwaversResult<Array3<f64>> {
        let mut heating = Array3::zeros(pressure_field.dim());

        Zip::indexed(&mut heating)
            .and(pressure_field)
            .for_each(|(i, j, k), q, &p| {
                let (x, y, z) = grid.indices_to_coordinates(i, j, k);
                let density = crate::domain::medium::density_at(medium, x, y, z, grid);
                let sound_speed = crate::domain::medium::sound_speed_at(medium, x, y, z, grid);
                let alpha =
                    AcousticProperties::absorption_coefficient(medium, x, y, z, grid, frequency);

                let intensity = (p * p) / (2.0 * density * sound_speed);

                *q = 2.0 * alpha * intensity;
            });

        Ok(heating)
    }
}
