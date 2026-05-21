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

use super::MultiPhysicsCoupling;
use crate::core::constants::fundamental::{DENSITY_BLOOD, DENSITY_WATER_NOMINAL};
use crate::core::constants::medical::{BLOOD_SPECIFIC_HEAT, TISSUE_PERFUSION_RATE};
use crate::core::constants::thermodynamic::{
    BODY_TEMPERATURE_C, SPECIFIC_HEAT_WATER, THERMAL_CONDUCTIVITY_WATER,
};
use ndarray::ArrayD;

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
        DENSITY_WATER_NOMINAL * SPECIFIC_HEAT_WATER / THERMAL_CONDUCTIVITY_WATER
    }

    /// Perfusion cooling rate (W/m³). Pennes (1948) Eq 1: q_p = w·ρ_b·c_b·(T − T_b).
    fn perfusion_cooling(&self, temperature: f64, _position: &[f64]) -> f64 {
        TISSUE_PERFUSION_RATE * DENSITY_BLOOD * BLOOD_SPECIFIC_HEAT * (temperature - BODY_TEMPERATURE_C)
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
