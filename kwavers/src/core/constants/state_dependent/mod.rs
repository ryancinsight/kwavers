//! State-Dependent Physical Constants
//!
//! This module provides thermodynamic state-dependent physical properties
//! that vary with temperature, pressure, and frequency.
//!
//! ## Literature References
//!
//! - **Speed of Sound**: Del Grosso (1972) "A new equation for the speed of sound in natural waters"
//! - **Viscosity**: Vogel-Fulcher-Tammann (VFT) equation, NIST data fit
//! - **Surface Tension**: IAPWS correlation (International Association for Properties of Water and Steam)
//! - **Nonlinear Parameter**: Duck (1990) "Physical Properties of Tissue", Law et al. (1985)
//! - **Pressure Effects**: Holton (1951), Wilson (1959)
//!
//! ## Physical Models
//!
//! 1. **Temperature Dependence** (Primary)
//!    - Speed of sound: dc/dT ≈ 3.0 m/s/K (Del Grosso)
//!    - Viscosity: η(T) = A·exp(B/(T-C)) (VFT equation with NIST-fit constants)
//!    - Surface tension: σ(T) = B·τ^μ·(1 + b·τ) (IAPWS correlation)
//!    - B/A: Linear temperature dependence
//!
//! 2. **Pressure Dependence** (Secondary)
//!    - Speed of sound: c(p) = c₀(1 + βp) for compressibility
//!    - Viscosity: Barus equation η(p) = η₀·exp(αp)
//!
//! 3. **Frequency Dependence**
//!    - Attenuation: α(f) = α₀·f^b (power law, b ≈ 1.5-2.0 for tissue)
//!    - Dispersion: Kramers-Kronig relations link attenuation to phase velocity

mod acoustic;
#[cfg(test)]
mod tests;
mod thermo;
mod transport;

/// Temperature-dependent physical constants calculator
#[derive(Debug, Clone)]
pub struct StateDependentConstants {
    /// Reference temperature [°C]
    pub reference_temperature: f64,
    /// Reference pressure [Pa]
    pub reference_pressure: f64,
}

impl Default for StateDependentConstants {
    fn default() -> Self {
        Self {
            reference_temperature: 20.0,  // 20°C (room temperature)
            reference_pressure: 101325.0, // 1 atm
        }
    }
}

impl StateDependentConstants {
    /// Create new state-dependent constants calculator with custom reference state
    pub fn new(reference_temperature: f64, reference_pressure: f64) -> Self {
        Self {
            reference_temperature,
            reference_pressure,
        }
    }
}
