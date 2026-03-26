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

use ndarray::ArrayD;
use super::MultiPhysicsCoupling;

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
        let rho = 1000.0; // kg/m³
        let c = 1500.0;   // m/s
        temperature_rate.mapv(|dtdt| beta * rho * c * c * dtdt)
    }

    /// Viscous dissipation heating rate (W/m³)
    fn viscous_heating(&self, _velocity_field: &ArrayD<f64>, _position: &[f64]) -> f64 {
        0.0 // Override with actual strain-rate computation
    }

    /// Thermal conductivity damping coefficient
    fn thermal_conductivity_damping(&self, frequency: f64, _position: &[f64]) -> f64 {
        let k = 0.6;       // W/m·K
        let rho = 1000.0;  // kg/m³
        let cp = 4186.0;   // J/kg·K
        let alpha = k / (rho * cp);
        let omega = 2.0 * std::f64::consts::PI * frequency;
        (2.0 * alpha / (omega * rho * cp)).sqrt()
    }
}
