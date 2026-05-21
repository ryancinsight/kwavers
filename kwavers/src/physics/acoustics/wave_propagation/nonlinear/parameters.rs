use crate::core::constants::acoustic_parameters::NP_TO_DB;
use crate::core::constants::fundamental::{
    C_WATER, DENSITY_TISSUE, DENSITY_WATER, SOUND_SPEED_TISSUE,
};
use crate::core::constants::numerical::CM_TO_M;

/// Parameters defining the nonlinear propagation properties of a medium
#[derive(Debug, Clone, Copy)]
pub struct NonlinearParameters {
    /// Density of the medium (rho_0) [kg/m^3]
    pub density: f64,
    /// Small-signal sound speed (c_0) (m/s)
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
            density: DENSITY_TISSUE,
            sound_speed: SOUND_SPEED_TISSUE,
            b_over_a,
            beta: 1.0 + b_over_a / 2.0,
            // 0.5 dB/(cm·MHz) → Np/(m·MHz): / CM_TO_M / NP_TO_DB (SSOT-sourced)
            attenuation_coeff: 0.5 / CM_TO_M / NP_TO_DB,
            attenuation_exponent: 1.1, // Typical for soft tissue
        }
    }

    /// Create parameters for water
    #[must_use]
    pub fn water() -> Self {
        // B/A for water at 20°C ~ 5.0
        let b_over_a = 5.0;
        Self {
            density: DENSITY_WATER,
            sound_speed: C_WATER,
            b_over_a,
            beta: 1.0 + b_over_a / 2.0,
            // 0.0022 dB/(cm·MHz²) → Np/(m·MHz²) — water classical f² absorption.
            attenuation_coeff: 0.0022 / CM_TO_M / NP_TO_DB,
            attenuation_exponent: 2.0,
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
    /// Fundamental frequency (Hz)
    pub fundamental_frequency: f64,
    /// Peak negative pressure of fundamental (Pa)
    pub fundamental_pressure: f64,
    /// Bandwidth of the transducer (fractional)
    pub fractional_bandwidth: f64,
    /// F-number of the imaging system
    pub f_number: f64,
    /// Focal depth (m)
    pub focal_depth: f64,
}
