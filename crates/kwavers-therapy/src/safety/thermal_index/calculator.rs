//! Thermal-index calculator implementation.

use kwavers_core::error::{KwaversError, KwaversResult, ValidationError};

use super::types::{ThermalIndexModel, ThermalIndexResult, ThermalIndexStatus};

/// Thermal-index calculator for IEC/AIUM style acoustic-output safety.
#[derive(Debug, Clone)]
pub struct ThermalIndexCalculator {
    model: ThermalIndexModel,
    center_frequency_mhz: f64,
    attenuation_coeff_db_cm_mhz: f64,
    reference_power_w: f64,
    safety_limit: f64,
}

impl ThermalIndexCalculator {
    /// Create a thermal-index calculator.
    ///
    /// # Theorem: ratio definition of thermal index
    ///
    /// For a selected output-display model, let `W_deg` be the model reference
    /// acoustic power predicted to raise tissue temperature by 1 °C, and let
    /// `W` be the acoustic power after attenuation derating to the evaluation
    /// depth. The thermal index is `TI = W / W_deg`. This is dimensionless and
    /// input-sensitive for every acoustic-power, frequency, attenuation, and
    /// depth change.
    ///
    /// # Algorithm
    ///
    /// 1. Validate finite positive frequency and reference power.
    /// 2. Validate finite nonnegative attenuation and depth.
    /// 3. Convert amplitude attenuation `α f z` in dB to power derating
    ///    `10^(-α f z / 10)`.
    /// 4. Return `TI = W0 10^(-α f z / 10) / W_deg`.
    ///
    /// # References
    ///
    /// - IEC 62359:2010+A1:2017, thermal and mechanical index determination.
    /// - AIUM Output Display Standard guidance for TIS/TIB/TIC selection.
    #[must_use]
    pub fn new(
        model: ThermalIndexModel,
        center_frequency_mhz: f64,
        attenuation_coeff_db_cm_mhz: f64,
        reference_power_w: f64,
        safety_limit: f64,
    ) -> Self {
        Self {
            model,
            center_frequency_mhz,
            attenuation_coeff_db_cm_mhz,
            reference_power_w,
            safety_limit,
        }
    }

    /// Calculate thermal index from acoustic power in watts.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn calculate(
        &self,
        acoustic_power_w: f64,
        depth_cm: f64,
    ) -> KwaversResult<ThermalIndexResult> {
        self.validate_domain(acoustic_power_w, depth_cm)?;

        let attenuation_db =
            self.attenuation_coeff_db_cm_mhz * self.center_frequency_mhz * depth_cm;
        let derated_acoustic_power_w = acoustic_power_w * 10.0_f64.powf(-attenuation_db / 10.0);
        let thermal_index = derated_acoustic_power_w / self.reference_power_w;

        let caution_threshold = 0.8 * self.safety_limit;
        let safety_status = if thermal_index > self.safety_limit {
            ThermalIndexStatus::Unsafe
        } else if thermal_index >= caution_threshold {
            ThermalIndexStatus::Caution
        } else {
            ThermalIndexStatus::Safe
        };

        Ok(ThermalIndexResult {
            model: self.model,
            thermal_index,
            derated_acoustic_power_w,
            reference_power_w: self.reference_power_w,
            center_frequency_mhz: self.center_frequency_mhz,
            depth_cm,
            safety_limit: self.safety_limit,
            safety_status,
        })
    }

    fn validate_domain(&self, acoustic_power_w: f64, depth_cm: f64) -> KwaversResult<()> {
        Self::require_positive_finite(
            "center_frequency_mhz",
            self.center_frequency_mhz,
            "Thermal-index attenuation model requires finite positive MHz frequency",
        )?;
        Self::require_nonnegative_finite(
            "attenuation_coeff_db_cm_mhz",
            self.attenuation_coeff_db_cm_mhz,
            "Attenuation coefficient must be finite and nonnegative",
        )?;
        Self::require_positive_finite(
            "reference_power_w",
            self.reference_power_w,
            "Thermal-index denominator W_deg must be finite and positive",
        )?;
        Self::require_positive_finite(
            "safety_limit",
            self.safety_limit,
            "Thermal-index safety limit must be finite and positive",
        )?;
        Self::require_nonnegative_finite(
            "acoustic_power_w",
            acoustic_power_w,
            "Acoustic power must be finite and nonnegative",
        )?;
        Self::require_nonnegative_finite(
            "depth_cm",
            depth_cm,
            "Thermal-index evaluation depth must be finite and nonnegative",
        )
    }

    fn require_positive_finite(parameter: &str, value: f64, reason: &str) -> KwaversResult<()> {
        if value.is_finite() && value > 0.0 {
            Ok(())
        } else {
            Err(Self::invalid_value(parameter, value, reason))
        }
    }

    fn require_nonnegative_finite(parameter: &str, value: f64, reason: &str) -> KwaversResult<()> {
        if value.is_finite() && value >= 0.0 {
            Ok(())
        } else {
            Err(Self::invalid_value(parameter, value, reason))
        }
    }

    fn invalid_value(parameter: &str, value: f64, reason: &str) -> KwaversError {
        KwaversError::Validation(ValidationError::InvalidValue {
            parameter: parameter.to_owned(),
            value,
            reason: reason.to_owned(),
        })
    }
}
