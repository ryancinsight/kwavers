//! B-mode imaging validation against FDA 510(k) image-quality and
//! measurement-accuracy requirements.

use std::collections::HashMap;

use super::{
    ClinicalCategory, ClinicalStandard, ClinicalValidationResult, ClinicalValidator,
    ImageQualityMetrics, MeasurementAccuracy, SafetyIndices,
};
use crate::core::error::{KwaversError, KwaversResult, ValidationError};

impl ClinicalValidator {
    /// Validate B-mode imaging performance
    pub fn validate_bmode(
        &self,
        quality_metrics: &ImageQualityMetrics,
        accuracy_metrics: &MeasurementAccuracy,
        safety_indices: &SafetyIndices,
    ) -> KwaversResult<ClinicalValidationResult> {
        let mut metrics = HashMap::new();
        let mut issues = Vec::new();
        let mut recommendations = Vec::new();

        // Extract metrics
        metrics.insert(
            "contrast_resolution".to_string(),
            quality_metrics.contrast_resolution,
        );
        metrics.insert(
            "axial_resolution".to_string(),
            quality_metrics.axial_resolution,
        );
        metrics.insert(
            "lateral_resolution".to_string(),
            quality_metrics.lateral_resolution,
        );
        metrics.insert("dynamic_range".to_string(), quality_metrics.dynamic_range);
        metrics.insert("snr".to_string(), quality_metrics.snr);
        metrics.insert(
            "distance_error".to_string(),
            accuracy_metrics.distance_error_percent,
        );
        metrics.insert(
            "area_error".to_string(),
            accuracy_metrics.area_error_percent,
        );
        metrics.insert(
            "mechanical_index".to_string(),
            safety_indices.mechanical_index,
        );
        metrics.insert(
            "thermal_index".to_string(),
            safety_indices.thermal_index_soft,
        );

        // Check FDA 510(k) requirements
        let reqs = self
            .requirements
            .get(&(ClinicalStandard::FDA510k, ClinicalCategory::BMode))
            .ok_or_else(|| {
                KwaversError::Validation(ValidationError::ConstraintViolation {
                    message: "FDA 510(k) B-mode requirements not found".to_string(),
                })
            })?;

        let mut passed = true;

        // Check minimum metrics
        for (metric_name, min_value) in &reqs.minimum_metrics {
            if let Some(actual_value) = metrics.get(metric_name) {
                if *actual_value < *min_value {
                    passed = false;
                    issues.push(format!(
                        "{} ({:.2}) below minimum requirement ({:.2})",
                        metric_name, actual_value, min_value
                    ));
                    recommendations.push(format!(
                        "Improve {} to meet FDA 510(k) requirements",
                        metric_name
                    ));
                }
            }
        }

        // Check maximum errors
        for (error_name, max_error) in &reqs.maximum_errors {
            if let Some(actual_error) = metrics.get(error_name) {
                if *actual_error > *max_error {
                    passed = false;
                    issues.push(format!(
                        "{} ({:.2}%) exceeds maximum error ({:.2}%)",
                        error_name, actual_error, max_error
                    ));
                    recommendations.push(format!("Reduce {} measurement error", error_name));
                }
            }
        }

        // Check safety thresholds
        for (safety_name, max_value) in &reqs.safety_thresholds {
            if let Some(actual_value) = metrics.get(safety_name) {
                if *actual_value > *max_value {
                    passed = false;
                    issues.push(format!(
                        "{} ({:.2}) exceeds safety threshold ({:.2})",
                        safety_name, actual_value, max_value
                    ));
                    recommendations
                        .push("Reduce acoustic output to meet safety requirements".to_string());
                }
            }
        }

        // Calculate clinical acceptability score
        let clinical_score = self.calculate_clinical_score(&metrics, reqs);

        Ok(ClinicalValidationResult {
            passed,
            metrics,
            issues,
            recommendations,
            regulatory_compliant: passed,
            clinical_score,
        })
    }
}
