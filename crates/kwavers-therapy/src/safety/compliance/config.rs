use kwavers_core::constants::numerical::{MHZ_TO_HZ, SECONDS_PER_HOUR, SECONDS_PER_MINUTE};
use kwavers_core::error::{KwaversError, KwaversResult};

use super::super::mechanical_index::MechanicalIndexTissueType;
use super::ComplianceConfig;

impl Default for ComplianceConfig {
    fn default() -> Self {
        Self {
            max_power: 50.0,
            max_intensity: 3.0,
            max_temp_rise: 5.0,
            max_session_time: SECONDS_PER_HOUR,
            max_total_dose: 100_000.0,
            tissue_type: MechanicalIndexTissueType::SoftTissue,
            frequency_range: (0.5 * MHZ_TO_HZ, 10.0 * MHZ_TO_HZ),
            max_bnur: 8.0,
            enable_monitoring: true,
            history_window: SECONDS_PER_MINUTE,
        }
    }
}

impl ComplianceConfig {
    /// Validate.
    /// # Errors
    /// - Returns `KwaversError::InvalidInput` if the precondition for invalid or out-of-range input parameters is violated.
    ///
    pub fn validate(&self) -> KwaversResult<()> {
        if self.max_power <= 0.0 {
            return Err(KwaversError::InvalidInput(
                "max_power must be positive".to_owned(),
            ));
        }

        if self.max_intensity <= 0.0 {
            return Err(KwaversError::InvalidInput(
                "max_intensity must be positive".to_owned(),
            ));
        }

        if self.max_temp_rise <= 0.0 {
            return Err(KwaversError::InvalidInput(
                "max_temp_rise must be positive".to_owned(),
            ));
        }

        if self.frequency_range.0 >= self.frequency_range.1 {
            return Err(KwaversError::InvalidInput(
                "frequency_range min must be less than max".to_owned(),
            ));
        }

        Ok(())
    }

    #[must_use]
    pub fn with_power_limit(mut self, watts: f64) -> Self {
        self.max_power = watts;
        self
    }

    #[must_use]
    pub fn with_intensity_limit(mut self, w_cm2: f64) -> Self {
        self.max_intensity = w_cm2;
        self
    }

    #[must_use]
    pub fn with_tissue_type(mut self, tissue: MechanicalIndexTissueType) -> Self {
        self.tissue_type = tissue;
        self
    }
}