//! Thermal Dose Calculation (CEM43)
//!
//! Reference: Sapareto, S. A., & Dewey, W. C. (1984). "Thermal dose determination in
//! cancer therapy." International Journal of Radiation Oncology Biology Physics,
//! 10(6), 787-800.

use crate::core::constants::numerical::SECONDS_PER_MINUTE;
use crate::core::constants::thermodynamic::KELVIN_OFFSET_C;
use crate::core::error::KwaversResult;
use ndarray::{Array3, Zip};

pub mod thresholds {
    use crate::core::constants::medical::{
        THERMAL_DOSE_DAMAGE_THRESHOLD_CEM43, THERMAL_DOSE_R_ABOVE_43C, THERMAL_DOSE_R_BELOW_43C,
        THERMAL_DOSE_REFERENCE_TEMP_C, THERMAL_DOSE_THRESHOLD,
    };
    /// CEM43 reference temperature [°C] — delegates to [`THERMAL_DOSE_REFERENCE_TEMP_C`].
    ///
    /// Sapareto & Dewey (1984), Eq. 1.
    pub const REFERENCE_TEMPERATURE_C: f64 = THERMAL_DOSE_REFERENCE_TEMP_C;
    /// Piecewise breakpoint for R-factor selection [°C] — same as `REFERENCE_TEMPERATURE_C`.
    pub const BREAKPOINT_TEMPERATURE_C: f64 = THERMAL_DOSE_REFERENCE_TEMP_C;
    /// Minimum temperature at which dose accumulates [°C] — equals normal body temperature.
    pub const MIN_DOSE_TEMPERATURE_C: f64 =
        crate::core::constants::thermodynamic::BODY_TEMPERATURE_C;
    /// CEM43 R factor above the 43°C breakpoint — delegates to [`THERMAL_DOSE_R_ABOVE_43C`].
    pub const R_ABOVE_BREAKPOINT: f64 = THERMAL_DOSE_R_ABOVE_43C;
    /// CEM43 R factor below the 43°C breakpoint — delegates to [`THERMAL_DOSE_R_BELOW_43C`].
    pub const R_BELOW_BREAKPOINT: f64 = THERMAL_DOSE_R_BELOW_43C;
    /// CEM43 threshold for irreversible necrosis — delegates to [`THERMAL_DOSE_THRESHOLD`].
    pub const NECROSIS_THRESHOLD_CEM43: f64 = THERMAL_DOSE_THRESHOLD;
    /// CEM43 threshold for reversible damage — delegates to [`THERMAL_DOSE_DAMAGE_THRESHOLD_CEM43`].
    pub const DAMAGE_THRESHOLD_CEM43: f64 = THERMAL_DOSE_DAMAGE_THRESHOLD_CEM43;
}

#[derive(Debug)]
pub struct ThermalDoseCalculator {
    cumulative_dose: Array3<f64>,
    max_dose: f64,
    max_dose_time: f64,
}

impl ThermalDoseCalculator {
    /// New.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    #[must_use]
    pub fn new(shape: (usize, usize, usize)) -> Self {
        Self {
            cumulative_dose: Array3::zeros(shape),
            max_dose: 0.0,
            max_dose_time: 0.0,
        }
    }
    /// Update dose.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
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
                let temp_celsius = temp_kelvin - KELVIN_OFFSET_C;

                if temp_celsius > MIN_DOSE_TEMPERATURE_C {
                    let r = if temp_celsius >= BREAKPOINT_TEMPERATURE_C {
                        R_ABOVE_BREAKPOINT
                    } else {
                        R_BELOW_BREAKPOINT
                    };

                    let exponent = REFERENCE_TEMPERATURE_C - temp_celsius;
                    let dose_increment = r.powf(exponent) * dt / SECONDS_PER_MINUTE;
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
