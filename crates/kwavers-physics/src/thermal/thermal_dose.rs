//! Thermal dose calculation using CEM43 model
//!
//! References:
//! - Sapareto & Dewey (1984) "Thermal dose determination in cancer therapy"
//! - Dewhirst et al. (2003) "Basic principles of thermal dosimetry"

use kwavers_core::constants::medical::{
    THERMAL_DOSE_REFERENCE_TEMP_C, THERMAL_DOSE_R_ABOVE_43C, THERMAL_DOSE_R_BELOW_43C,
};
use kwavers_core::constants::numerical::SECONDS_PER_MINUTE;
use kwavers_core::error::{KwaversError, KwaversResult};
use leto::Array3;
use moirai_parallel::{for_each_chunk_mut_enumerated_with, Adaptive};

/// CEM43 reference temperature (°C).
///
/// Delegates to [`kwavers_core::constants::medical::THERMAL_DOSE_REFERENCE_TEMP_C`].
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
            dose: Array3::zeros([nx, ny, nz]),
            nx,
            ny,
            nz,
            reference_temp: CEM43_REFERENCE_TEMPERATURE_C,
        }
    }

    /// Update thermal dose based on current temperature field
    /// dt: time step in seconds
    ///
    /// # Errors
    /// Returns [`KwaversError::DimensionMismatch`] when `temperature` does not
    /// have the same shape as this dose grid.
    pub fn update(&mut self, temperature: &Array3<f64>, dt: f64) -> KwaversResult<()> {
        let dose_shape = self.dose.shape();
        let temperature_shape = temperature.shape();
        if dose_shape != temperature_shape {
            return Err(KwaversError::DimensionMismatch(format!(
                "thermal dose update requires temperature shape {dose_shape:?}, got {temperature_shape:?}"
            )));
        }

        let dt_minutes = dt / SECONDS_PER_MINUTE;
        let reference_temp = self.reference_temp;
        let temperatures = temperature.as_slice().ok_or_else(|| {
            KwaversError::InvalidInput(
                "thermal dose update requires a dense Leto temperature field".to_string(),
            )
        })?;
        let doses = self
            .dose
            .as_slice_mut()
            .expect("invariant: ThermalCEM43Grid owns a dense Leto dose field");
        const CHUNK_SIZE: usize = 1024;

        for_each_chunk_mut_enumerated_with::<Adaptive, _, _>(
            doses,
            CHUNK_SIZE,
            |chunk_index, dose_chunk| {
                let offset = chunk_index * CHUNK_SIZE;
                // CEM43 formula: CEM43 = ∫ R^(T_ref − T) dt
                // Sapareto & Dewey (1984): R = 0.5 for T ≥ 43°C, R = 0.25 for T < 43°C.
                // Accumulates at all physiological temperatures above 0°C; the R=0.25
                // exponent naturally suppresses dose at low temperatures (e.g., at 30°C:
                // 0.25^(43-30) ≈ 1.5e-8 CEM43/min), so no temperature gate is required.
                for (local_index, dose) in dose_chunk.iter_mut().enumerate() {
                    let temp = temperatures[offset + local_index];
                    if temp > 0.0 {
                        let r = if temp >= reference_temp {
                            THERMAL_DOSE_R_ABOVE_43C
                        } else {
                            THERMAL_DOSE_R_BELOW_43C
                        };
                        let equiv_time = dt_minutes * r.powf(reference_temp - temp);
                        *dose += equiv_time;
                    }
                }
            },
        );
        Ok(())
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
    use kwavers_core::constants::medical::{
        THERMAL_DOSE_COAGULATION_CEM43, THERMAL_DOSE_DIAGNOSTIC_SAFETY_CEM43,
        THERMAL_DOSE_PROTEIN_DENATURATION_CEM43,
    };
    use kwavers_core::constants::thermodynamic::BODY_TEMPERATURE_C;

    #[test]
    fn test_thermal_dose_accumulation_above_reference() {
        let mut dose_calc = ThermalCEM43Grid::new(10, 10, 10);

        // At 45°C with R=0.5: CEM43 = 1 min * 0.5^(43−45) = 1 * 4 = 4 minutes
        // Sapareto & Dewey (1984), Table 1.
        let temperature = Array3::from_elem([10, 10, 10], 45.0);
        dose_calc.update(&temperature, 60.0).unwrap();

        let expected = 0.5_f64.powf(43.0 - 45.0); // = 4.0 CEM43 per minute × 1 min
        let actual = dose_calc.get_dose()[[5, 5, 5]];
        assert!(
            (actual - expected).abs() < 1e-10,
            "45°C dose: expected {expected}, got {actual}"
        );
    }

    #[test]
    fn test_thermal_dose_accumulation_mild_hyperthermia() {
        // 40°C is below the 43°C reference but above body temperature.
        // The prior implementation (guard: temp > 37°C) blocked this range.
        // Sapareto & Dewey (1984) use R=0.25 for T < 43°C, giving non-zero accumulation.
        let mut dose_calc = ThermalCEM43Grid::new(5, 5, 5);
        let temperature = Array3::from_elem([5, 5, 5], 40.0);
        dose_calc.update(&temperature, 60.0).unwrap(); // 1 minute

        // expected = 0.25^(43 - 40) = 0.25^3 = 0.015625 CEM43
        let expected = 0.25_f64.powf(43.0 - 40.0);
        let actual = dose_calc.get_dose()[[2, 2, 2]];
        assert!(
            (actual - expected).abs() < 1e-12,
            "40°C mild hyperthermia dose: expected {expected:.6e}, got {actual:.6e}"
        );
        assert!(
            actual > 0.0,
            "CEM43 must accumulate at mild-hyperthermic temperatures (40°C > 0°C)"
        );
    }

    #[test]
    fn test_dose_threshold_ablation() {
        let mut dose_calc = ThermalCEM43Grid::new(10, 10, 10);

        // Half volume at 50°C (ablative), half at 37°C (body temperature).
        // At 37°C: 0.25^(43-37) = 0.25^6 ≈ 2.44e-4 CEM43/min — negligible vs 100 CEM43.
        // At 50°C: 0.5^(43-50) = 0.5^(-7) = 128 CEM43/min × 10 min = 1280 CEM43.
        let mut temperature = Array3::from_elem([10, 10, 10], BODY_TEMPERATURE_C);
        for k in 0..5 {
            for j in 0..10 {
                for i in 0..10 {
                    temperature[[i, j, k]] = 50.0;
                }
            }
        }
        dose_calc.update(&temperature, 600.0).unwrap(); // 10 minutes

        let fraction = dose_calc.fraction_above_threshold(100.0);
        assert!(
            (fraction - 0.5).abs() < 0.05,
            "Expected ~50% above 100 CEM43, got {:.1}%",
            fraction * 100.0
        );
    }

    #[test]
    fn test_canonical_cem43_constants() {
        assert_eq!(CEM43_REFERENCE_TEMPERATURE_C, 43.0);
        assert_eq!(THERMAL_DOSE_PROTEIN_DENATURATION_CEM43, 1.0);
        assert_eq!(THERMAL_DOSE_COAGULATION_CEM43, 10_000.0);
        assert_eq!(THERMAL_DOSE_DIAGNOSTIC_SAFETY_CEM43, 0.1);
    }

    #[test]
    fn test_dose_resets_to_zero() {
        let mut dose_calc = ThermalCEM43Grid::new(5, 5, 5);
        let temperature = Array3::from_elem([5, 5, 5], 45.0);
        dose_calc.update(&temperature, 60.0).unwrap();
        assert!(dose_calc.get_max_dose() > 0.0);
        dose_calc.reset();
        assert_eq!(dose_calc.get_max_dose(), 0.0);
    }

    #[test]
    fn test_update_rejects_shape_mismatch() {
        let mut dose_calc = ThermalCEM43Grid::new(5, 5, 5);
        let temperature = Array3::from_elem([5, 4, 5], 45.0);

        let error = dose_calc.update(&temperature, 60.0).unwrap_err();

        assert!(
            format!("{error}").contains("thermal dose update requires temperature shape"),
            "unexpected error: {error}"
        );
    }
}
