use crate::core::constants::fundamental::SOUND_SPEED_TISSUE;

/// Parameters defining the nonlinear propagation properties of a medium
#[derive(Debug, Clone, Copy)]
pub struct NonlinearParameters {
    /// Density of the medium (rho_0) [kg/m^3]
    pub density: f64,
    /// Small-signal sound speed (c_0) [m/s]
    pub sound_speed: f64,
    /// Nonlinear parameter B/A (dimensionless)
    pub b_over_a: f64,
    /// Coefficient of nonlinearity (beta = 1 + B/2A)
    pub beta: f64,
    /// Attenuation coefficient at 1 MHz [Np/m/MHz^1.1] or similar
    pub attenuation_coeff: f64,
    /// Frequency dependence of attenuation (exponent y, typically ~1.0 for tissue, 2.0 for water)
    pub attenuation_exponent: f64,
}

impl NonlinearParameters {
    /// Create parameters for typical soft tissue
    #[must_use]
    pub fn soft_tissue() -> Self {
        // Typical values for soft tissue (from literature)
        // B/A ~ 7.0 (Duck, 1990)
        let b_over_a = 7.0;
        Self {
            density: 1050.0,
            sound_speed: SOUND_SPEED_TISSUE,
            b_over_a,
            beta: 1.0 + b_over_a / 2.0,
            attenuation_coeff: 0.5 * 100.0 / 8.686, // ~0.5 dB/cm/MHz converted to Np/m/MHz
            attenuation_exponent: 1.1,              // Typical for soft tissue
        }
    }

    /// Create parameters for water
    #[must_use]
    pub fn water() -> Self {
        // B/A for water at 20C ~ 5.0
        let b_over_a = 5.0;
        Self {
            density: 998.0,
            sound_speed: 1482.0,
            b_over_a,
            beta: 1.0 + b_over_a / 2.0,
            attenuation_coeff: 0.0022 * 100.0 / 8.686, // ~0.0022 dB/cm/MHz^2 converted to Np/m/MHz^2
            attenuation_exponent: 2.0,                 // Water follows classical f^2 attenuation
        }
    }

    /// Calculate attenuation at a specific frequency [Np/m]
    #[must_use]
    pub fn attenuation_at_frequency(&self, frequency_hz: f64) -> f64 {
        let f_mhz = frequency_hz / 1e6;
        self.attenuation_coeff * f_mhz.powf(self.attenuation_exponent)
    }
}

/// Properties for tissue harmonic imaging simulation
#[derive(Debug, Clone)]
pub struct TissueHarmonicProperties {
    /// Fundamental frequency [Hz]
    pub fundamental_frequency: f64,
    /// Peak negative pressure of fundamental [Pa]
    pub fundamental_pressure: f64,
    /// Bandwidth of the transducer (fractional)
    pub fractional_bandwidth: f64,
    /// F-number of the imaging system
    pub f_number: f64,
    /// Focal depth [m]
    pub focal_depth: f64,
}
