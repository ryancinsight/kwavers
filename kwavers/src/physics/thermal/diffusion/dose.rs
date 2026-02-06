//! Thermal Dose Calculation (CEM43)
//!
//! Reference: Sapareto, S. A., & Dewey, W. C. (1984). "Thermal dose determination in
//! cancer therapy." International Journal of Radiation Oncology Biology Physics,
//! 10(6), 787-800.

use crate::core::error::KwaversResult;
use ndarray::{Array3, Zip};

pub mod thresholds {
    pub const REFERENCE_TEMPERATURE_C: f64 = 43.0;
    pub const BREAKPOINT_TEMPERATURE_C: f64 = 43.0;
    pub const MIN_DOSE_TEMPERATURE_C: f64 = 37.0;
    pub const R_ABOVE_BREAKPOINT: f64 = 0.5;
    pub const R_BELOW_BREAKPOINT: f64 = 0.25;
    pub const NECROSIS_THRESHOLD_CEM43: f64 = 240.0;
    pub const DAMAGE_THRESHOLD_CEM43: f64 = 60.0;
}

#[derive(Debug)]
pub struct ThermalDoseCalculator {
    cumulative_dose: Array3<f64>,
    max_dose: f64,
    max_dose_time: f64,
}

impl ThermalDoseCalculator {
    #[must_use]
    pub fn new(shape: (usize, usize, usize)) -> Self {
        Self {
            cumulative_dose: Array3::zeros(shape),
            max_dose: 0.0,
            max_dose_time: 0.0,
        }
    }

    pub fn update_dose(
        &mut self,
        temperature: &Array3<f64>,
        dt: f64,
        current_time: f64,
    ) -> KwaversResult<()> {
        use thresholds::{
            BREAKPOINT_TEMPERATURE_C, MIN_DOSE_TEMPERATURE_C, REFERENCE_TEMPERATURE_C,
            R_ABOVE_BREAKPOINT, R_BELOW_BREAKPOINT,
        };

        Zip::from(&mut self.cumulative_dose)
            .and(temperature)
            .for_each(|dose, &temp_kelvin| {
                let temp_celsius = temp_kelvin - 273.15;

                if temp_celsius > MIN_DOSE_TEMPERATURE_C {
                    let r = if temp_celsius >= BREAKPOINT_TEMPERATURE_C {
                        R_ABOVE_BREAKPOINT
                    } else {
                        R_BELOW_BREAKPOINT
                    };

                    let exponent = REFERENCE_TEMPERATURE_C - temp_celsius;
                    let dose_increment = r.powf(exponent) * dt / 60.0;
                    *dose += dose_increment;

                    if *dose > self.max_dose {
                        self.max_dose = *dose;
                        self.max_dose_time = current_time;
                    }
                }
            });

        Ok(())
    }

    #[must_use]
    pub fn get_dose(&self) -> &Array3<f64> {
        &self.cumulative_dose
    }

    #[must_use]
    pub fn max_dose(&self) -> f64 {
        self.max_dose
    }

    #[must_use]
    pub fn max_dose_time(&self) -> f64 {
        self.max_dose_time
    }

    #[must_use]
    pub fn check_damage_threshold(&self, threshold_cem43: f64) -> Array3<bool> {
        self.cumulative_dose.mapv(|dose| dose >= threshold_cem43)
    }

    #[must_use]
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

    #[must_use]
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

    pub fn reset(&mut self) {
        self.cumulative_dose.fill(0.0);
        self.max_dose = 0.0;
        self.max_dose_time = 0.0;
    }
}
