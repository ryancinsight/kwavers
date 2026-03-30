//! Electromagnetic-Thermal Coupling (Photothermal Effects)
//!
//! ## Mathematical Foundation
//!
//! Bioheat equation (Pennes 1948):
//! ```text
//! ρ c_p ∂T/∂t = k ∇²T + Q_optical - w_b ρ_b c_b (T - T_b) + Q_met
//! ```
//!
//! Optical heating rate: Q = μ_a Φ
//!
//! ## References
//!
//! - Pennes (1948) J Appl Physiol 1(2):93-122
//! - Welch & van Gemert (2011) "Optical-Thermal Response of Laser-Irradiated Tissue"

use crate::core::constants::fundamental::{DENSITY_BLOOD, DENSITY_WATER_NOMINAL};
use crate::core::constants::thermodynamic::SPECIFIC_HEAT_WATER;
use ndarray::ArrayD;
use super::MultiPhysicsCoupling;

/// Electromagnetic-thermal coupling for photothermal effects
pub trait ElectromagneticThermalCoupling: MultiPhysicsCoupling {
    /// Optical absorption coefficient μ_a (m⁻¹)
    fn optical_absorption_coefficient(&self, position: &[f64], wavelength: f64) -> f64;

    /// Optical heating rate Q = μ_a Φ (W/m³)
    fn optical_heating_rate(
        &self,
        fluence_rate: &ArrayD<f64>,
        position: &[f64],
        wavelength: f64,
    ) -> ArrayD<f64> {
        let mu_a = self.optical_absorption_coefficient(position, wavelength);
        fluence_rate.mapv(|dphi_dt| mu_a * dphi_dt)
    }

    /// Thermal relaxation time τ = ρ C_p / k (s)
    fn thermal_relaxation_time(&self, _position: &[f64]) -> f64 {
        let rho = DENSITY_WATER_NOMINAL;
        let cp = SPECIFIC_HEAT_WATER;
        let k = 0.6;
        rho * cp / k
    }

    /// Perfusion cooling rate (W/m³/K)
    fn perfusion_cooling(&self, temperature: f64, _position: &[f64]) -> f64 {
        let w = 0.01;
        let rho_b = DENSITY_BLOOD;
        let cp_b = 3860.0;
        let tb = 37.0;
        w * rho_b * cp_b * (temperature - tb)
    }

    /// Bioheat equation source term (including perfusion)
    fn bioheat_source(
        &self,
        fluence_rate: &ArrayD<f64>,
        temperature: f64,
        position: &[f64],
        wavelength: f64,
    ) -> ArrayD<f64> {
        let optical_heating = self.optical_heating_rate(fluence_rate, position, wavelength);
        let perfusion = self.perfusion_cooling(temperature, position);
        optical_heating.mapv(|q_opt| q_opt - perfusion)
    }
}
