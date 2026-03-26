//! Safety constraints validation

use super::planner::TreatmentPlanner;
use super::types::SafetyConstraints;
use crate::core::error::KwaversResult;
use ndarray::Array3;

impl TreatmentPlanner {
    /// Validate safety constraints
    pub(crate) fn validate_safety(
        &self,
        temperature: &Array3<f64>,
        acoustic_field: &Array3<f64>,
        frequency_hz: f64,
    ) -> KwaversResult<()> {
        let constraints = SafetyConstraints::default();

        // Check temperature limits
        for &temp in temperature.iter() {
            if temp > constraints.max_brain_temp {
                return Err(crate::core::error::KwaversError::Validation(
                    crate::core::error::ValidationError::ConstraintViolation {
                        message: format!(
                            "Brain temperature {:.1}°C exceeds limit {:.1}°C",
                            temp, constraints.max_brain_temp
                        ),
                    },
                ));
            }
        }

        // Check mechanical index using MI = P_neg(MPa) / sqrt(frequency_MHz)
        // Convert intensity to pressure using p = sqrt(I * rho * c)
        let max_intensity = acoustic_field.iter().fold(0.0_f64, |a, &b| a.max(b));
        let rho = 1000.0; // kg/m^3
        let c = 1500.0; // m/s
        let p_pa = (max_intensity * rho * c).sqrt();
        let p_mpa = p_pa / 1_000_000.0; // Pa -> MPa
        let freq_mhz = frequency_hz / 1_000_000.0;
        let mi = if freq_mhz > 0.0 {
            p_mpa / freq_mhz.sqrt()
        } else {
            f64::INFINITY
        };

        if mi > constraints.max_mi {
            return Err(crate::core::error::KwaversError::Validation(
                crate::core::error::ValidationError::ConstraintViolation {
                    message: format!(
                        "Mechanical index {:.2} exceeds limit {:.2}",
                        mi, constraints.max_mi
                    ),
                },
            ));
        }

        Ok(())
    }
}
