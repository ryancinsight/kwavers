//! Thermal Dose Calculation (CEM43)
//!
//! Reference: Sapareto, S. A., & Dewey, W. C. (1984). "Thermal dose determination in
//! cancer therapy." International Journal of Radiation Oncology Biology Physics,
//! 10(6), 787-800.

use aequitas::systems::si::quantities::Time;
use kwavers_core::constants::thermodynamic::KELVIN_OFFSET_C;
use kwavers_core::error::KwaversResult;
use leto::Array3;

use crate::parallel::zip_mut_ref;
use crate::thermal::response::{checked_cem43_increments, KelvinStorage};

/// Kwavers-owned thermal-dose policy thresholds.
pub mod thresholds {
    use kwavers_core::constants::medical::{
        THERMAL_DOSE_DAMAGE_THRESHOLD_CEM43, THERMAL_DOSE_THRESHOLD,
    };
    /// Minimum Celsius temperature at which dose accumulates.
    pub const MIN_DOSE_TEMPERATURE_C: f64 =
        kwavers_core::constants::thermodynamic::BODY_TEMPERATURE_C;
    /// CEM43 threshold for irreversible necrosis — delegates to [`THERMAL_DOSE_THRESHOLD`].
    pub const NECROSIS_THRESHOLD_CEM43: f64 = THERMAL_DOSE_THRESHOLD;
    /// CEM43 threshold for reversible damage — delegates to [`THERMAL_DOSE_DAMAGE_THRESHOLD_CEM43`].
    pub const DAMAGE_THRESHOLD_CEM43: f64 = THERMAL_DOSE_DAMAGE_THRESHOLD_CEM43;
}

/// Accumulates the CEM43 thermal dose (cumulative equivalent minutes at 43 °C)
/// per voxel over a heating history, tracking the running maximum.
#[derive(Debug)]
pub struct ThermalDoseCalculator {
    cumulative_dose: Array3<f64>,
    increments: Array3<f64>,
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
            increments: Array3::zeros(shape),
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
        use thresholds::MIN_DOSE_TEMPERATURE_C;

        let step = Time::from_base(dt);
        checked_cem43_increments::<KelvinStorage, _>(
            self.increments.view_mut(),
            temperature.view(),
            step,
            |temp_kelvin| temp_kelvin - KELVIN_OFFSET_C > MIN_DOSE_TEMPERATURE_C,
        )?;
        zip_mut_ref(
            self.cumulative_dose.view_mut(),
            self.increments.view(),
            |dose, &increment| *dose += increment,
        );

        let updated_max = self.cumulative_dose.iter().copied().fold(0.0_f64, f64::max);
        if updated_max > self.max_dose {
            self.max_dose = updated_max;
            self.max_dose_time = current_time;
        }

        Ok(())
    }

    /// Per-voxel accumulated CEM43 thermal dose in equivalent minutes at 43 °C.
    #[must_use]
    pub fn get_dose(&self) -> &Array3<f64> {
        &self.cumulative_dose
    }

    /// Peak per-voxel CEM43 dose reached so far.
    #[must_use]
    pub fn max_dose(&self) -> f64 {
        self.max_dose
    }

    /// Simulation time in seconds at which the peak dose was reached.
    #[must_use]
    pub fn max_dose_time(&self) -> f64 {
        self.max_dose_time
    }

    /// Boolean mask of voxels whose accumulated dose meets `threshold_cem43`.
    #[must_use]
    pub fn check_damage_threshold(&self, threshold_cem43: f64) -> Array3<bool> {
        self.cumulative_dose.mapv(|dose| dose >= threshold_cem43)
    }

    /// Fraction of voxels exceeding the irreversible-necrosis CEM43 threshold.
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

    /// Fraction of voxels exceeding the reversible-damage CEM43 threshold.
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

    /// Clear the accumulated dose and reset the running maximum (reuse the
    /// calculator for a fresh heating history).
    pub fn reset(&mut self) {
        self.cumulative_dose.fill(0.0);
        self.increments.fill(0.0);
        self.max_dose = 0.0;
        self.max_dose_time = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_reference_accumulates_one_minute() {
        let mut calculator = ThermalDoseCalculator::new((1, 1, 1));
        let temperature = Array3::from_elem((1, 1, 1), 316.15);
        calculator
            .update_dose(&temperature, 60.0, 60.0)
            .expect("valid reference observation");
        assert_eq!(calculator.get_dose()[[0, 0, 0]], 1.0);
    }

    #[test]
    fn rejected_observation_does_not_mutate_dose_or_maximum() {
        let mut calculator = ThermalDoseCalculator::new((2, 1, 1));
        let reference = Array3::from_elem((2, 1, 1), 316.15);
        calculator
            .update_dose(&reference, 60.0, 60.0)
            .expect("valid reference observation");
        let before = calculator.get_dose().clone();
        let max_before = calculator.max_dose();
        let time_before = calculator.max_dose_time();

        let mut invalid = Array3::from_elem((2, 1, 1), 317.15);
        invalid[[1, 0, 0]] = f64::NAN;
        assert!(calculator.update_dose(&invalid, 60.0, 120.0).is_err());
        assert_eq!(calculator.get_dose(), &before);
        assert_eq!(calculator.max_dose(), max_before);
        assert_eq!(calculator.max_dose_time(), time_before);

        let wrong_shape = Array3::from_elem((1, 1, 1), 317.15);
        assert!(calculator.update_dose(&wrong_shape, 60.0, 120.0).is_err());
        assert_eq!(calculator.get_dose(), &before);
    }
}
