//! Acoustic-Thermal Coupling (Thermoacoustic Effects)
//!
//! ## Mathematical Foundation
//!
//! Pressure source from temperature changes:
//! ```text
//! ∂p/∂t = β ρ c² ∂T/∂t
//! ```
//!
//! Thermal damping penetration depth:
//! ```text
//! δ = √(2α/(ω ρ c_p))
//! ```
//!
//! ## References
//!
//! - Swift (1988) "Thermoacoustic engines" JASA 84(4):1145-1180

use super::MultiPhysicsCoupling;
use kwavers_core::constants::fundamental::{DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER_SIM};
use kwavers_core::constants::numerical::TWO_PI;
use kwavers_core::constants::thermodynamic::{SPECIFIC_HEAT_WATER, THERMAL_CONDUCTIVITY_WATER};
use ArrayD;

/// Acoustic-thermal coupling for thermoacoustic effects
pub trait AcousticThermalCoupling: MultiPhysicsCoupling {
    /// Thermal expansion coefficient β (1/K)
    fn thermal_expansion_coefficient(&self, position: &[f64]) -> f64;

    /// Compute acoustic pressure source from temperature rate ∂T/∂t
    fn pressure_source_from_temperature(
        &self,
        temperature_rate: &ArrayD<f64>,
        position: &[f64],
    ) -> ArrayD<f64> {
        let beta = self.thermal_expansion_coefficient(position);
        let rho = DENSITY_WATER_NOMINAL;
        let c = SOUND_SPEED_WATER_SIM;
        temperature_rate.mapv(|dtdt| beta * rho * c * c * dtdt)
    }

    /// Viscous dissipation heating rate (W/m³)
    fn viscous_heating(&self, _velocity_field: &ArrayD<f64>, _position: &[f64]) -> f64 {
        0.0 // Override with actual strain-rate computation
    }

    /// Thermal conductivity damping coefficient
    fn thermal_conductivity_damping(&self, frequency: f64, _position: &[f64]) -> f64 {
        let alpha = THERMAL_CONDUCTIVITY_WATER / (DENSITY_WATER_NOMINAL * SPECIFIC_HEAT_WATER);
        let omega = TWO_PI * frequency;
        // δ = √(2·α_th / ω) [m] — classical thermal penetration depth
        (2.0 * alpha / omega).sqrt()
    }
}
