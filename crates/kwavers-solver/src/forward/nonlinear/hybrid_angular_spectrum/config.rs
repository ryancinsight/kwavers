//! HAS solver configuration.

use kwavers_core::constants::acoustic_parameters::NP_TO_DB;
use kwavers_core::constants::fundamental::{ACOUSTIC_ABSORPTION_TISSUE, DENSITY_WATER_NOMINAL};
use kwavers_core::constants::numerical::MHZ_TO_HZ;
use kwavers_core::constants::tissue_acoustics::B_OVER_A_SOFT_TISSUE;
use kwavers_core::constants::SOUND_SPEED_WATER_SIM;
use kwavers_core::error::{KwaversError, KwaversResult};

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
        let config = Self {
            sound_speed,
            density,
            nonlinearity,
            attenuation_coeff,
            power_law_exponent,
            dz,
            reference_frequency,
        };
        config.validate()?;
        Ok(config)
    }

    /// Validate the physical-range invariants of the configuration.
    ///
    /// This is the SSOT for HAS configuration validity, called by
    /// [`HASConfig::new`] and re-checked by consumers (e.g. the absorption
    /// operator) so a config reached via `default()` + field mutation cannot
    /// drive the power-law absorption with a non-positive reference frequency
    /// (`f^y → NaN`) or a negative attenuation coefficient.
    ///
    /// # Errors
    /// - [`KwaversError::InvalidInput`] if any parameter is out of its physical range.
    pub fn validate(&self) -> KwaversResult<()> {
        if self.sound_speed <= 0.0 {
            return Err(KwaversError::InvalidInput(
                "Sound speed must be positive".to_owned(),
            ));
        }
        if self.density <= 0.0 {
            return Err(KwaversError::InvalidInput(
                "Density must be positive".to_owned(),
            ));
        }
        if self.dz <= 0.0 {
            return Err(KwaversError::InvalidInput(
                "Step size must be positive".to_owned(),
            ));
        }
        if !(0.0..=3.0).contains(&self.power_law_exponent) {
            return Err(KwaversError::InvalidInput(
                "Power law exponent should be between 0 and 3".to_owned(),
            ));
        }
        // The power-law absorption raises `(f_ref/MHz)^y`, well-defined only for a
        // positive frequency; reject NaN explicitly (it is neither > nor <= 0).
        if self.reference_frequency.is_nan() || self.reference_frequency <= 0.0 {
            return Err(KwaversError::InvalidInput(
                "Reference frequency must be positive".to_owned(),
            ));
        }
        if !self.attenuation_coeff.is_finite() || self.attenuation_coeff < 0.0 {
            return Err(KwaversError::InvalidInput(
                "Attenuation coefficient must be finite and non-negative".to_owned(),
            ));
        }
        Ok(())
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
