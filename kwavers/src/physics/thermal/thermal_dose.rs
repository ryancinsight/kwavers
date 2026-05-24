//! Thermal dose calculation using CEM43 model
//!
//! References:
//! - Sapareto & Dewey (1984) "Thermal dose determination in cancer therapy"
//! - Dewhirst et al. (2003) "Basic principles of thermal dosimetry"

use crate::core::constants::medical::THERMAL_DOSE_REFERENCE_TEMP_C;
use crate::core::constants::thermodynamic::BODY_TEMPERATURE_C;
use ndarray::{Array3, Zip};

/// CEM43 reference temperature (°C).
///
/// Delegates to [`crate::core::constants::medical::THERMAL_DOSE_REFERENCE_TEMP_C`].
/// Sapareto & Dewey (1984) defined the cumulative equivalent minutes at 43 °C
/// as the canonical thermal-dose metric for hyperthermic therapy.
pub const CEM43_REFERENCE_TEMPERATURE_C: f64 = THERMAL_DOSE_REFERENCE_TEMP_C;

/// Thermal dose calculator using cumulative equivalent minutes at 43°C (CEM43)
#[derive(Debug)]
pub struct ThermalCEM43Grid {
    /// Cumulative thermal dose (CEM43 minutes)
    dose: Array3<f64>,
    /// Grid dimensions
    nx: usize,
    ny: usize,
    nz: usize,
    /// Reference temperature for dose calculation (°C)
    reference_temp: f64,
}

impl ThermalCEM43Grid {
    /// Create new thermal dose calculator
    #[must_use]
    pub fn new(nx: usize, ny: usize, nz: usize) -> Self {
        Self {
            dose: Array3::zeros((nx, ny, nz)),
            nx,
            ny,
            nz,
            reference_temp: CEM43_REFERENCE_TEMPERATURE_C,
        }
    }

    /// Update thermal dose based on current temperature field
    /// dt: time step in seconds
    pub fn update(&mut self, temperature: &Array3<f64>, dt: f64) {
        let dt_minutes = dt / 60.0;

        Zip::from(&mut self.dose)
            .and(temperature)
            .par_for_each(|dose, &temp| {
                // CEM43 formula: t_eq = t * R^(43-T)
                // where R = 0.5 for T > 43°C, R = 0.25 for T < 43°C
                let r = if temp >= self.reference_temp {
                    0.5
                } else {
                    0.25
                };

                if temp > BODY_TEMPERATURE_C {
                    // Only accumulate dose above body temperature
                    let r_value: f64 = r;
                    let equiv_time = dt_minutes * r_value.powf(self.reference_temp - temp);
                    *dose += equiv_time;
                }
            });
    }

    /// Get cumulative thermal dose
    #[must_use]
    pub fn get_dose(&self) -> &Array3<f64> {
        &self.dose
    }

    /// Get maximum thermal dose
    #[must_use]
    pub fn get_max_dose(&self) -> f64 {
        self.dose.iter().fold(0.0_f64, |a, &b| a.max(b))
    }

    /// Get thermal dose at specific point
    #[must_use]
    pub fn get_dose_at(&self, i: usize, j: usize, k: usize) -> f64 {
        self.dose[[i, j, k]]
    }

    /// Check if thermal dose exceeds threshold for tissue damage
    /// Returns fraction of volume exceeding threshold
    #[must_use]
    pub fn fraction_above_threshold(&self, threshold: f64) -> f64 {
        let count = self.dose.iter().filter(|&&d| d > threshold).count();
        count as f64 / (self.nx * self.ny * self.nz) as f64
    }

    /// Reset dose accumulation
    pub fn reset(&mut self) {
        self.dose.fill(0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_thermal_dose_accumulation() {
        let mut dose_calc = ThermalCEM43Grid::new(10, 10, 10);

        // Create temperature field at 45°C
        let temperature = Array3::from_elem((10, 10, 10), 45.0);

        // Update for 60 seconds
        dose_calc.update(&temperature, 60.0);

        // At 45°C with R=0.5: CEM43 = 1 min * 0.5^(43-45) = 1 * 0.5^(-2) = 4 minutes
        let expected_dose = 4.0;
        let actual_dose = dose_calc.get_dose()[[5, 5, 5]];

        assert!(
            (actual_dose - expected_dose).abs() < 0.01,
            "Dose calculation error: expected {}, got {}",
            expected_dose,
            actual_dose
        );
    }

    #[test]
    fn test_dose_threshold() {
        let mut dose_calc = ThermalCEM43Grid::new(10, 10, 10);

        // Set half the volume to accumulate dose
        let mut temperature = Array3::from_elem((10, 10, 10), BODY_TEMPERATURE_C);
        for k in 0..5 {
            for j in 0..10 {
                for i in 0..10 {
                    temperature[[i, j, k]] = 50.0; // High temperature
                }
            }
        }

        // Update for 10 minutes (600 seconds)
        dose_calc.update(&temperature, 600.0);

        // Check fraction above threshold
        let fraction = dose_calc.fraction_above_threshold(100.0);
        assert!(
            (fraction - 0.5).abs() < 0.1,
            "Expected ~50% above threshold, got {:.1}%",
            fraction * 100.0
        );
    }

    #[test]
    fn test_thermal_dose_thresholds_use_canonical_constants() {
        assert_eq!(CEM43_REFERENCE_TEMPERATURE_C, 43.0);
    }
}
