//! Non-equilibrium mass transfer routines

use super::vapor_pressure::ThermodynamicsCalculator;
use crate::core::constants::{M_WATER, R_GAS};

/// Mass transfer model for bubble dynamics
#[derive(Debug, Clone)]
pub struct MassTransferModel {
    pub(crate) thermo: ThermodynamicsCalculator,
    /// Accommodation coefficient (typically 0.04-1.0)
    pub(crate) accommodation_coeff: f64,
    /// Enable non-equilibrium effects
    pub(crate) non_equilibrium: bool,
}

impl MassTransferModel {
    /// Create a new mass transfer model
    #[must_use]
    pub fn new(accommodation_coeff: f64) -> Self {
        Self {
            thermo: ThermodynamicsCalculator::default(),
            accommodation_coeff,
            non_equilibrium: true,
        }
    }

    /// Calculate mass transfer rate for bubble
    ///
    /// # Arguments
    /// * `temperature` - Bubble temperature \[K\]
    /// * `pressure_vapor` - Current vapor pressure in bubble \[Pa\]
    /// * `surface_area` - Bubble surface area \[m²\]
    ///
    /// # Returns
    /// Mass transfer rate [kg/s] (positive for evaporation)
    #[must_use]
    pub fn mass_transfer_rate(
        &self,
        temperature: f64,
        pressure_vapor: f64,
        surface_area: f64,
    ) -> f64 {
        // Saturation pressure at bubble temperature
        let p_sat = self.thermo.vapor_pressure(temperature);

        // Pressure difference drives mass transfer
        let delta_p = p_sat - pressure_vapor;

        // Hertz-Knudsen equation
        let coeff = self
            .thermo
            .mass_transfer_coefficient(temperature, self.accommodation_coeff);
        let rate = coeff * surface_area * delta_p * M_WATER / (R_GAS * temperature);

        // Non-equilibrium correction for rapid dynamics
        if self.non_equilibrium {
            let peclet = pressure_vapor.abs() / p_sat;
            let correction = 1.0 / (1.0 + 0.5 * peclet);
            rate * correction
        } else {
            rate
        }
    }

    /// Calculate heat of phase change
    #[must_use]
    pub fn heat_transfer_rate(&self, mass_rate: f64, temperature: f64) -> f64 {
        let h_vap = self.thermo.enthalpy_vaporization(temperature);
        mass_rate * h_vap / M_WATER // Convert to J/s
    }
}
