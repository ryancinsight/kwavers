//! HAS solver configuration.

use crate::core::constants::acoustic_parameters::NP_TO_DB;
use crate::core::constants::fundamental::{ACOUSTIC_ABSORPTION_TISSUE, DENSITY_WATER_NOMINAL};
use crate::core::constants::numerical::MHZ_TO_HZ;
use crate::core::constants::tissue_acoustics::B_OVER_A_SOFT_TISSUE;
use crate::core::constants::SOUND_SPEED_WATER_SIM;
use crate::core::error::{KwaversError, KwaversResult};

/// Configuration for Hybrid Angular Spectrum solver
#[derive(Debug, Clone)]
pub struct HASConfig {
    /// Sound speed (m/s)
    pub sound_speed: f64,
    /// Medium density (kg/m³)
    pub density: f64,
    /// Nonlinearity parameter B/A
    pub nonlinearity: f64,
    /// Attenuation coefficient [dB/(cm·MHz^y)].
    ///
    /// Converted to Np/m at a given frequency by `attenuation_at_frequency`.
    pub attenuation_coeff: f64,
    /// Power law exponent (y)
    pub power_law_exponent: f64,
    /// Step size in propagation direction (m)
    pub dz: f64,
    /// Reference frequency (Hz) for dispersion
    pub reference_frequency: f64,
}

impl Default for HASConfig {
    fn default() -> Self {
        Self {
            sound_speed: SOUND_SPEED_WATER_SIM,
            density: DENSITY_WATER_NOMINAL,
            nonlinearity: B_OVER_A_SOFT_TISSUE, // 6.5 generic soft tissue (Duck 1990)
            attenuation_coeff: ACOUSTIC_ABSORPTION_TISSUE, // 0.5 dB/(cm·MHz) — Duck (1990)
            power_law_exponent: 2.0,
            dz: 0.0001,
            reference_frequency: MHZ_TO_HZ,
        }
    }
}

impl HASConfig {
    /// Create configuration with validation.
    /// # Errors
    /// - Returns [`KwaversError::InvalidInput`] if any parameter is out of range.
    pub fn new(
        sound_speed: f64,
        density: f64,
        nonlinearity: f64,
        attenuation_coeff: f64,
        power_law_exponent: f64,
        dz: f64,
        reference_frequency: f64,
    ) -> KwaversResult<Self> {
        if sound_speed <= 0.0 {
            return Err(KwaversError::InvalidInput(
                "Sound speed must be positive".to_owned(),
            ));
        }
        if density <= 0.0 {
            return Err(KwaversError::InvalidInput(
                "Density must be positive".to_owned(),
            ));
        }
        if dz <= 0.0 {
            return Err(KwaversError::InvalidInput(
                "Step size must be positive".to_owned(),
            ));
        }
        if !(0.0..=3.0).contains(&power_law_exponent) {
            return Err(KwaversError::InvalidInput(
                "Power law exponent should be between 0 and 3".to_owned(),
            ));
        }
        Ok(Self {
            sound_speed,
            density,
            nonlinearity,
            attenuation_coeff,
            power_law_exponent,
            dz,
            reference_frequency,
        })
    }

    /// Acoustic impedance Z = ρ·c (Pa·s/m).
    #[must_use]
    pub fn impedance(&self) -> f64 {
        self.density * self.sound_speed
    }

    /// Attenuation at `frequency` Hz using power-law model (Np/m).
    #[must_use]
    pub fn attenuation_at_frequency(&self, frequency: f64) -> f64 {
        let freq_mhz = frequency / MHZ_TO_HZ;
        let db_per_cm = self.attenuation_coeff * freq_mhz.powf(self.power_law_exponent);
        db_per_cm * 100.0 / NP_TO_DB
    }
}
