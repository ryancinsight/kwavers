//! Doppler imaging validation with default and configurable thresholds for
//! sensitivity, velocity accuracy, and angle accuracy.

use std::collections::HashMap;

use super::{
    ClinicalValidationResult, ClinicalValidator, DopplerValidationThresholds, SafetyIndices,
};
use crate::core::error::KwaversResult;

impl ClinicalValidator {
    /// Validate Doppler imaging performance
    pub fn validate_doppler(
        &self,
        velocity_accuracy: f64,
        angle_accuracy: f64,
        sensitivity: f64,
        safety_indices: &SafetyIndices,
    ) -> KwaversResult<ClinicalValidationResult> {
        let thresholds = DopplerValidationThresholds::default();
        self.validate_doppler_with_thresholds(
            velocity_accuracy,
            angle_accuracy,
            sensitivity,
            safety_indices,
            &thresholds,
        )
    }

    /// Validate Doppler imaging performance with configurable thresholds
    pub fn validate_doppler_with_thresholds(
        &self,
        velocity_accuracy: f64,
        angle_accuracy: f64,
        sensitivity: f64,
        safety_indices: &SafetyIndices,
        thresholds: &DopplerValidationThresholds,
    ) -> KwaversResult<ClinicalValidationResult> {
        let mut metrics = HashMap::new();
        let mut issues = Vec::new();
        let mut recommendations = Vec::new();

        metrics.insert("velocity_error".to_string(), velocity_accuracy);
        metrics.insert("angle_error".to_string(), angle_accuracy);
        metrics.insert("sensitivity".to_string(), sensitivity);
        metrics.insert(
            "mechanical_index".to_string(),
            safety_indices.mechanical_index,
        );

        // Doppler requirements are configurable; defaults are conservative placeholders.
        //
        // Not yet implemented: full AIUM, IEC, and FDA Doppler validation. Absent:
        // AIUM Acoustic Output Measurement Standard (Ophthalmic and Fetal); IEC 60601-2-37
        // Physiotherapy Equipment section; FDA 510(k) acoustic output notification
        // requirements; WHO Manual Safety and Quality Assurance criteria; and AIUM
        // Contrast-Enhanced Ultrasound accreditation standards.
        let min_sensitivity = thresholds.min_sensitivity_cm_s;
        let max_velocity_error = thresholds.max_velocity_error_percent;
        let max_angle_error = thresholds.max_angle_error_degrees;

        let mut passed = true;

        if sensitivity < min_sensitivity {
            passed = false;
            issues.push(format!(
                "Doppler sensitivity ({:.1} cm/s) below minimum ({:.1} cm/s)",
                sensitivity, min_sensitivity
            ));
            recommendations
                .push("Improve Doppler sensitivity through beamforming optimization".to_string());
        }

        if velocity_accuracy > max_velocity_error {
            passed = false;
            issues.push(format!(
                "Velocity accuracy ({:.1}%) exceeds maximum error ({:.1}%)",
                velocity_accuracy, max_velocity_error
            ));
            recommendations.push("Calibrate Doppler frequency estimation".to_string());
        }

        if angle_accuracy > max_angle_error {
            passed = false;
            issues.push(format!(
                "Angle accuracy ({:.1}°) exceeds maximum error ({:.1}°)",
                angle_accuracy, max_angle_error
            ));
            recommendations.push("Improve beam steering angle accuracy".to_string());
        }

        let clinical_score = 100.0
            - (velocity_accuracy + angle_accuracy) * 2.0
            - (min_sensitivity - sensitivity).max(0.0) * 5.0;

        Ok(ClinicalValidationResult {
            passed,
            metrics,
            issues,
            recommendations,
            regulatory_compliant: passed,
            clinical_score: clinical_score.clamp(0.0, 100.0),
        })
    }
}
