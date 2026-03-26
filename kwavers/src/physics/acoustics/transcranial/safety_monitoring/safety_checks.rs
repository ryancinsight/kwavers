//! Safety limit checking, status reporting, and recommendations

use super::monitor::SafetyMonitor;
use super::types::{SafetyLevel, SafetyReport, SafetyStatus, TreatmentProgress};
use crate::core::error::KwaversResult;

impl SafetyMonitor {
    /// Check safety limits and return warnings
    pub(crate) fn check_safety_limits(&self) -> KwaversResult<()> {
        // Immediate error on temperature limit exceedance
        let max_temp = self.temperature.iter().fold(f64::MIN, |a, &b| a.max(b));
        if max_temp > self.thresholds.max_temperature {
            return Err(crate::core::error::KwaversError::Validation(
                crate::core::error::ValidationError::ConstraintViolation {
                    message: format!(
                        "Temperature {:.1}°C exceeds limit {:.1}°C",
                        max_temp, self.thresholds.max_temperature
                    ),
                },
            ));
        }

        // Check thermal dose limits (strict)
        let max_dose = self
            .thermal_dose
            .current_dose
            .iter()
            .fold(0.0_f64, |a, &b| a.max(b));
        if max_dose > self.thresholds.max_thermal_dose {
            return Err(crate::core::error::KwaversError::Validation(
                crate::core::error::ValidationError::ConstraintViolation {
                    message: format!(
                        "Thermal dose {:.0} CEM43 exceeds limit {:.0} CEM43",
                        max_dose, self.thresholds.max_thermal_dose
                    ),
                },
            ));
        }

        // Check mechanical index
        if self.mechanical_index.current_mi > self.thresholds.max_mechanical_index {
            return Err(crate::core::error::KwaversError::Validation(
                crate::core::error::ValidationError::ConstraintViolation {
                    message: format!(
                        "Mechanical index {:.2} exceeds limit {:.2}",
                        self.mechanical_index.current_mi, self.thresholds.max_mechanical_index
                    ),
                },
            ));
        }

        Ok(())
    }

    /// Get current safety status
    pub fn safety_status(&self) -> SafetyStatus {
        let max_temp = self.temperature.iter().fold(0.0_f64, |a, &b| a.max(b));
        let max_dose = self
            .thermal_dose
            .current_dose
            .iter()
            .fold(0.0_f64, |a, &b| a.max(b));

        SafetyStatus {
            temperature_status: SafetyLevel::from_value(max_temp, self.thresholds.max_temperature),
            thermal_dose_status: SafetyLevel::from_value(
                max_dose,
                self.thresholds.max_thermal_dose,
            ),
            mechanical_index_status: SafetyLevel::from_value(
                self.mechanical_index.current_mi,
                self.thresholds.max_mechanical_index,
            ),
            overall_safety: self.overall_safety_level(),
        }
    }

    /// Calculate overall safety level
    fn overall_safety_level(&self) -> SafetyLevel {
        let status = self.safety_status();
        let levels = [
            status.temperature_status,
            status.thermal_dose_status,
            status.mechanical_index_status,
        ];
        levels.iter().cloned().max().unwrap_or(SafetyLevel::Safe)
    }

    /// Get treatment progress towards target dose
    pub fn treatment_progress(&self, target_dose: f64) -> TreatmentProgress {
        let max_current_dose = self
            .thermal_dose
            .current_dose
            .iter()
            .fold(0.0_f64, |a, &b| a.max(b));
        let progress = (max_current_dose / target_dose).min(1.0);

        let estimated_time_remaining = if progress < 1.0 {
            let max_time_to_target = self
                .thermal_dose
                .time_to_target
                .iter()
                .fold(0.0_f64, |a, &b| a.max(b));
            max_time_to_target
        } else {
            0.0
        };

        TreatmentProgress {
            dose_progress: progress,
            estimated_time_remaining,
            current_max_dose: max_current_dose,
            target_dose,
        }
    }

    /// Generate safety report
    pub fn safety_report(&self) -> SafetyReport {
        let status = self.safety_status();
        let progress = self.treatment_progress(self.thresholds.max_thermal_dose);

        SafetyReport {
            status,
            progress,
            thermal_dose: self.thermal_dose.clone(),
            mechanical_index: self.mechanical_index.clone(),
            recommendations: self.generate_recommendations(),
        }
    }

    /// Generate safety recommendations
    fn generate_recommendations(&self) -> Vec<String> {
        let mut recommendations = Vec::new();
        let status = self.safety_status();

        match status.temperature_status {
            SafetyLevel::Critical => {
                recommendations.push(
                    "CRITICAL: Temperature exceeds safe limits. Stop treatment immediately."
                        .to_string(),
                );
            }
            SafetyLevel::Warning => {
                recommendations
                    .push("WARNING: Temperature approaching limit. Reduce power.".to_string());
            }
            _ => {}
        }

        match status.thermal_dose_status {
            SafetyLevel::Critical => {
                recommendations.push(
                    "CRITICAL: Thermal dose exceeds safe limits. Stop treatment.".to_string(),
                );
            }
            SafetyLevel::Warning => {
                recommendations
                    .push("WARNING: Thermal dose approaching limit. Monitor closely.".to_string());
            }
            _ => {}
        }

        match status.mechanical_index_status {
            SafetyLevel::Critical => {
                recommendations.push(
                    "CRITICAL: Mechanical index exceeds safety limit. Reduce acoustic power."
                        .to_string(),
                );
            }
            SafetyLevel::Warning => {
                recommendations.push(
                    "WARNING: Mechanical index approaching limit. Reduce pressure amplitude."
                        .to_string(),
                );
            }
            _ => {}
        }

        if recommendations.is_empty() {
            recommendations
                .push("All parameters within safe limits. Treatment may continue.".to_string());
        }

        recommendations
    }
}
