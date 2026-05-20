//! Safety constraints validation

use super::planner::TreatmentPlanner;
use super::types::TranscranialSafetyConstraints;
use crate::core::error::{KwaversError, KwaversResult, ValidationError};
use crate::physics::acoustics::analysis::calculate_mechanical_index;
use ndarray::Array3;

const BRAIN_DENSITY_KG_PER_M3: f64 = 1040.0;
const BRAIN_SOUND_SPEED_M_PER_S: f64 = 1546.0;

impl TreatmentPlanner {
    /// Validate safety constraints
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub(crate) fn validate_safety(
        &self,
        temperature: &Array3<f64>,
        acoustic_field: &Array3<f64>,
        frequency_hz: f64,
    ) -> KwaversResult<()> {
        let constraints = TranscranialSafetyConstraints::default();
        Self::validate_safety_fields(temperature, acoustic_field, frequency_hz, &constraints)
    }

    fn validate_safety_fields(
        temperature: &Array3<f64>,
        acoustic_field: &Array3<f64>,
        frequency_hz: f64,
        constraints: &TranscranialSafetyConstraints,
    ) -> KwaversResult<()> {
        // Check temperature limits
        for &temp in temperature {
            if !temp.is_finite() {
                return Err(constraint_violation(
                    "Brain temperature field contains a nonfinite value",
                ));
            }

            if temp > constraints.max_brain_temp {
                return Err(constraint_violation(format!(
                    "Brain temperature {:.1}°C exceeds limit {:.1}°C",
                    temp, constraints.max_brain_temp
                )));
            }
        }

        let mi = mechanical_index_from_harmonic_intensity_field(acoustic_field, frequency_hz);

        if !mi.is_finite() || mi > constraints.max_mi {
            return Err(constraint_violation(format!(
                "Mechanical index {:.2} exceeds limit {:.2}",
                mi, constraints.max_mi
            )));
        }

        Ok(())
    }
}

fn constraint_violation(message: impl Into<String>) -> KwaversError {
    KwaversError::Validation(ValidationError::ConstraintViolation {
        message: message.into(),
    })
}

fn mechanical_index_from_harmonic_intensity_field(
    acoustic_field: &Array3<f64>,
    frequency_hz: f64,
) -> f64 {
    if !frequency_hz.is_finite() || frequency_hz <= 0.0 {
        return f64::INFINITY;
    }

    let Some(peak_pressure_pa) = peak_pressure_from_harmonic_intensity_field(acoustic_field) else {
        return f64::INFINITY;
    };

    calculate_mechanical_index(peak_pressure_pa, frequency_hz)
}

fn peak_pressure_from_harmonic_intensity_field(acoustic_field: &Array3<f64>) -> Option<f64> {
    let mut max_intensity = 0.0_f64;

    for &intensity in acoustic_field {
        if !intensity.is_finite() || intensity < 0.0 {
            return None;
        }
        max_intensity = max_intensity.max(intensity);
    }

    Some((2.0 * max_intensity * BRAIN_DENSITY_KG_PER_M3 * BRAIN_SOUND_SPEED_M_PER_S).sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_constraints() -> TranscranialSafetyConstraints {
        TranscranialSafetyConstraints::default()
    }

    #[test]
    fn mechanical_index_from_intensity_matches_harmonic_pressure_contract() {
        let pressure_pa = 1.0e6_f64;
        let intensity =
            pressure_pa.powi(2) / (2.0 * BRAIN_DENSITY_KG_PER_M3 * BRAIN_SOUND_SPEED_M_PER_S);
        let field = Array3::from_elem((2, 2, 2), intensity);

        let mi = mechanical_index_from_harmonic_intensity_field(&field, 1.0e6);

        assert!((mi - 1.0).abs() < 1.0e-12);
    }

    #[test]
    fn mechanical_index_from_intensity_rejects_invalid_domains() {
        let valid = Array3::from_elem((1, 1, 1), 1.0);
        let negative = Array3::from_elem((1, 1, 1), -1.0);
        let nonfinite = Array3::from_elem((1, 1, 1), f64::NAN);

        assert!(mechanical_index_from_harmonic_intensity_field(&valid, 0.0).is_infinite());
        assert!(mechanical_index_from_harmonic_intensity_field(&valid, f64::NAN).is_infinite());
        assert!(mechanical_index_from_harmonic_intensity_field(&negative, 1.0e6).is_infinite());
        assert!(mechanical_index_from_harmonic_intensity_field(&nonfinite, 1.0e6).is_infinite());
    }

    #[test]
    fn validate_safety_fields_accepts_safe_finite_fields() {
        let temperature = Array3::from_elem((2, 2, 2), 37.0);
        let acoustic_field = Array3::from_elem((2, 2, 2), 1.0);
        let constraints = default_constraints();
        let mi = mechanical_index_from_harmonic_intensity_field(&acoustic_field, 1.0e6);

        assert!(mi.is_finite());
        assert!(mi < constraints.max_mi);

        TreatmentPlanner::validate_safety_fields(
            &temperature,
            &acoustic_field,
            1.0e6,
            &constraints,
        )
        .expect("finite low-intensity field must satisfy safety constraints");
    }

    #[test]
    fn validate_safety_fields_rejects_nonfinite_temperature() {
        let temperature = Array3::from_elem((2, 2, 2), f64::NAN);
        let acoustic_field = Array3::from_elem((2, 2, 2), 1.0);

        let result = TreatmentPlanner::validate_safety_fields(
            &temperature,
            &acoustic_field,
            1.0e6,
            &default_constraints(),
        );

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("nonfinite"));
    }

    #[test]
    fn validate_safety_fields_rejects_invalid_frequency() {
        let temperature = Array3::from_elem((2, 2, 2), 37.0);
        let acoustic_field = Array3::from_elem((2, 2, 2), 1.0);

        let result = TreatmentPlanner::validate_safety_fields(
            &temperature,
            &acoustic_field,
            -1.0e6,
            &default_constraints(),
        );

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Mechanical index"));
    }

    #[test]
    fn validate_safety_fields_rejects_negative_intensity() {
        let temperature = Array3::from_elem((2, 2, 2), 37.0);
        let acoustic_field = Array3::from_elem((2, 2, 2), -1.0);

        let result = TreatmentPlanner::validate_safety_fields(
            &temperature,
            &acoustic_field,
            1.0e6,
            &default_constraints(),
        );

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Mechanical index"));
    }
}
