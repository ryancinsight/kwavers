//! Thermal Dose Calculation (CEM43)
//!
//! Reference: Sapareto, S. A., & Dewey, W. C. (1984). "Thermal dose determination in
//! cancer therapy." International Journal of Radiation Oncology Biology Physics,
//! 10(6), 787-800.

use crate::error::KwaversResult;
use ndarray::{Array3, Zip};

/// Temperature threshold constants for thermal dose
pub mod thresholds {
    /// Reference temperature for CEM43 calculation [°C]
    pub const REFERENCE_TEMPERATURE_C: f64 = 43.0;

    /// Threshold temperature above which R=0.5 [°C]
    pub const BREAKPOINT_TEMPERATURE_C: f64 = 43.0;

    /// Minimum temperature for dose accumulation [°C]
    pub const MIN_DOSE_TEMPERATURE_C: f64 = 37.0;

    /// R factor above breakpoint
    pub const R_ABOVE_BREAKPOINT: f64 = 0.5;

    /// R factor below breakpoint
    pub const R_BELOW_BREAKPOINT: f64 = 0.25;

    /// Necrosis threshold [CEM43 minutes]
    pub const NECROSIS_THRESHOLD_CEM43: f64 = 240.0;

    /// Significant damage threshold [CEM43 minutes]
    pub const DAMAGE_THRESHOLD_CEM43: f64 = 60.0;
}

/// CEM43 thermal dose calculator
#[derive(Debug)]
pub struct ThermalDoseCalculator {
    /// Cumulative thermal dose [equivalent minutes at 43°C]
    cumulative_dose: Array3<f64>,
    /// Maximum dose achieved
    max_dose: f64,
    /// Time at which max dose was achieved
    max_dose_time: f64,
}

impl ThermalDoseCalculator {
    pub fn new(shape: (usize, usize, usize)) -> Self {
        Self {
            cumulative_dose: Array3::zeros(shape),
            max_dose: 0.0,
            max_dose_time: 0.0,
        }
    }

    /// Update thermal dose using CEM43 formula
    /// CEM43 = Σ R^(43-T) * Δt
    /// where R = 0.5 for T ≥ 43°C, R = 0.25 for T < 43°C
    pub fn update_dose(
        &mut self,
        temperature: &Array3<f64>,
        dt: f64,
        current_time: f64,
    ) -> KwaversResult<()> {
        use thresholds::*;

        Zip::from(&mut self.cumulative_dose)
            .and(temperature)
            .for_each(|dose, &temp_kelvin| {
                // Convert to Celsius
                let temp_celsius = temp_kelvin - 273.15;

                // Only accumulate dose above minimum temperature
                if temp_celsius > MIN_DOSE_TEMPERATURE_C {
                    // Calculate R factor based on temperature
                    let r = if temp_celsius >= BREAKPOINT_TEMPERATURE_C {
                        R_ABOVE_BREAKPOINT
                    } else {
                        R_BELOW_BREAKPOINT
                    };

                    // Calculate dose increment
                    let exponent = REFERENCE_TEMPERATURE_C - temp_celsius;
                    let dose_increment = r.powf(exponent) * dt / 60.0; // Convert seconds to minutes

                    *dose += dose_increment;

                    // Track maximum dose
                    if *dose > self.max_dose {
                        self.max_dose = *dose;
                        self.max_dose_time = current_time;
                    }
                }
            });

        Ok(())
    }

    /// Get the cumulative thermal dose field
    pub fn get_dose(&self) -> &Array3<f64> {
        &self.cumulative_dose
    }

    /// Get maximum dose in the field
    pub fn max_dose(&self) -> f64 {
        self.max_dose
    }

    /// Get time at which maximum dose was achieved
    pub fn max_dose_time(&self) -> f64 {
        self.max_dose_time
    }

    /// Check if thermal damage threshold is exceeded
    pub fn check_damage_threshold(&self, threshold_cem43: f64) -> Array3<bool> {
        self.cumulative_dose.mapv(|dose| dose >= threshold_cem43)
    }

    /// Get necrosis volume fraction
    pub fn necrosis_fraction(&self) -> f64 {
        use thresholds::NECROSIS_THRESHOLD_CEM43;

        let total_points = self.cumulative_dose.len() as f64;
        let necrosed_points = self
            .cumulative_dose
            .iter()
            .filter(|&&dose| dose >= NECROSIS_THRESHOLD_CEM43)
            .count() as f64;

        necrosed_points / total_points
    }

    /// Get damage volume fraction
    pub fn damage_fraction(&self) -> f64 {
        use thresholds::DAMAGE_THRESHOLD_CEM43;

        let total_points = self.cumulative_dose.len() as f64;
        let damaged_points = self
            .cumulative_dose
            .iter()
            .filter(|&&dose| dose >= DAMAGE_THRESHOLD_CEM43)
            .count() as f64;

        damaged_points / total_points
    }

    /// Reset dose accumulation
    pub fn reset(&mut self) {
        self.cumulative_dose.fill(0.0);
        self.max_dose = 0.0;
        self.max_dose_time = 0.0;
    }
}
