//! Arrhenius temperature-dependent kinetics validator
//!
//! # Mathematical Specification
//!
//! ## Theorem: Arrhenius Rate Scaling
//! For a reaction with activation energy $E_a$ and reference rate $k_0$ at
//! temperature $T_0$, the rate at temperature $T$ is:
//!
//! $$ k(T) = k_0 \cdot \exp\left(-\frac{E_a}{R}\left(\frac{1}{T} - \frac{1}{T_0}\right)\right) $$
//!
//! ## Corollary: Q10 Temperature Coefficient
//! The Q10 factor $Q_{10} = k(T+10)/k(T)$ is bounded in $[1.5, 5.0]$ for
//! biologically relevant reactions (activation energies $40$–$120\,\text{kJ/mol}$).

use crate::core::constants::GAS_CONSTANT;

/// Temperature-dependent Arrhenius kinetics validator
#[derive(Debug)]
pub struct ArrheniusValidator {
    /// Activation energy [J/mol]
    pub activation_energy: f64,
    /// Reference temperature [K]
    pub reference_temperature: f64,
}

impl ArrheniusValidator {
    /// Create validator with Arrhenius parameters
    pub fn new(activation_energy: f64, reference_temperature: f64) -> Self {
        Self {
            activation_energy,
            reference_temperature,
        }
    }

    /// Calculate rate constant at temperature using Arrhenius equation
    ///
    /// k(T) = k₀ · exp(-Eₐ/R·(1/T - 1/T₀))
    pub fn rate_constant_at_temperature(&self, rate_at_reference: f64, temperature: f64) -> f64 {
        let exponent = -self.activation_energy / GAS_CONSTANT
            * (1.0 / temperature - 1.0 / self.reference_temperature);
        rate_at_reference * exponent.exp()
    }

    /// Q10 factor (rate change per 10°C)
    ///
    /// Q10 = k(T+10) / k(T) for typical reactions
    pub fn q10_factor(&self, temperature: f64) -> f64 {
        let k1 = self.rate_constant_at_temperature(1.0, temperature);
        let k2 = self.rate_constant_at_temperature(1.0, temperature + 10.0);
        k2 / k1
    }

    /// Validate Q10 is reasonable (typically 2-4)
    pub fn is_reasonable_q10(&self, temperature: f64) -> bool {
        let q10 = self.q10_factor(temperature);
        (1.5..=5.0).contains(&q10)
    }
}

