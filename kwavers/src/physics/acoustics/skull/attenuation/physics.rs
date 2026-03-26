//! Frequency and temperature dependent physics implementations

use super::model::SkullAttenuation;

impl SkullAttenuation {
    /// Compute absorption coefficient at given frequency
    ///
    /// α_abs(f) = α₀ · f^n
    ///
    /// # Arguments
    ///
    /// * `frequency` - Frequency (Hz)
    ///
    /// # Returns
    ///
    /// Absorption coefficient (Np/m)
    #[must_use]
    pub fn absorption_coefficient(&self, frequency: f64) -> f64 {
        let freq_mhz = frequency / 1e6;
        self.alpha_0 * freq_mhz.powf(self.exponent)
    }

    /// Compute scattering coefficient at given frequency
    ///
    /// Uses Rayleigh scattering model: α_scatter ∝ f^4 (low frequency limit)
    ///
    /// # Arguments
    ///
    /// * `frequency` - Frequency (Hz)
    ///
    /// # Returns
    ///
    /// Scattering coefficient (Np/m)
    #[must_use]
    pub fn scattering_coefficient(&self, frequency: f64) -> f64 {
        if !self.include_scattering {
            return 0.0;
        }

        let freq_mhz = frequency / 1e6;

        // Rayleigh scattering for low frequencies (f < 2 MHz typically)
        // Transition to geometric scattering at higher frequencies
        if freq_mhz < 2.0 {
            self.scattering_coeff * freq_mhz.powi(4)
        } else {
            // Transition/geometric regime: ~f^2 dependence
            self.scattering_coeff * 16.0 * freq_mhz.powi(2)
        }
    }

    /// Compute total attenuation coefficient
    ///
    /// α_total = α_absorption + α_scattering
    ///
    /// # Arguments
    ///
    /// * `frequency` - Frequency (Hz)
    ///
    /// # Returns
    ///
    /// Total attenuation coefficient (Np/m)
    #[must_use]
    pub fn total_coefficient(&self, frequency: f64) -> f64 {
        self.absorption_coefficient(frequency) + self.scattering_coefficient(frequency)
    }

    /// Compute temperature-dependent attenuation adjustment
    ///
    /// Empirical model: α(T) = α(T_ref) · [1 + β·(T - T_ref)]
    /// where β ≈ 0.01-0.02 /°C for bone
    ///
    /// # Arguments
    ///
    /// * `temperature` - Current temperature (°C)
    ///
    /// # Returns
    ///
    /// Temperature correction factor (multiplicative)
    #[must_use]
    pub fn temperature_correction(&self, temperature: f64) -> f64 {
        let beta = 0.015; // Temperature coefficient (1/°C)
        1.0 + beta * (temperature - self.reference_temperature)
    }
}
