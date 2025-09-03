//! Absorption operator for KZK equation
//!
//! Implements frequency-dependent absorption following power law.
//! Reference: Szabo (1994) "Time domain wave equations for lossy media"

use super::KZKConfig;
use ndarray::{Array3, Zip};

/// Absorption operator with power law dependence
pub struct AbsorptionOperator {
    /// Absorption coefficient at 1 `MHz`
    alpha0: f64,
    /// Power law exponent
    power: f64,
    /// Configuration
    config: KZKConfig,
}

impl AbsorptionOperator {
    /// Create new absorption operator
    #[must_use]
    pub fn new(config: &KZKConfig) -> Self {
        // Convert from dB/cm/MHz to Np/m/Hz
        let alpha0_np = config.alpha0 * 100.0 / 8.686 / 1e6;

        Self {
            alpha0: alpha0_np,
            power: config.alpha_power,
            config: config.clone(),
        }
    }

    /// Apply absorption for one step
    /// Models frequency-dependent attenuation: α(f) = α₀|f|^y
    pub fn apply(&mut self, pressure: &mut Array3<f64>, step_size: f64) {
        // For time-domain implementation, use fractional derivative
        // Approximation: apply exponential decay based on operating frequency

        // Use the actual operating frequency from config
        let f0 = self.config.frequency;

        // Compute absorption coefficient in Np/m
        // α(f) = α₀ * f^y where α₀ is already in Np/m/Hz^y
        let alpha = self.alpha0 * f0.powf(self.power);

        // Apply exponential decay: p(z+dz) = p(z) * exp(-α * dz)
        let decay = (-alpha * step_size).exp();

        Zip::from(pressure).for_each(|p| {
            *p *= decay;
        });
    }

    /// Compute absorption for specific frequency
    #[must_use]
    pub fn get_absorption(&self, frequency: f64) -> f64 {
        self.alpha0 * frequency.powf(self.power)
    }

    /// Compute penetration depth (1/e decay)
    #[must_use]
    pub fn penetration_depth(&self, frequency: f64) -> f64 {
        1.0 / self.get_absorption(frequency)
    }
}
