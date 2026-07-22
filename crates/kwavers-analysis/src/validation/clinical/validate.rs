//! Validation methods for `ClinicalValidator`.
//!
//! SRP: changes when clinical criteria or scoring formulas change.

use super::types::{
    ClinicalCategory, ClinicalRequirements, ClinicalStandard, ClinicalValidationResult,
    DopplerValidationThresholds, ImageQualityMetrics, MeasurementAccuracy, SafetyIndices,
};
use super::validator::ClinicalValidator;
use kwavers_core::error::{KwaversError, KwaversResult};
use std::collections::HashMap;

impl ClinicalValidator {
    /// Validate B-mode imaging performance against FDA 510(k) requirements.
    /// # Errors
    /// - Propagates any `KwaversError` returned by called functions.
    ///
    pub fn validate_bmode(
        &self,
        quality_metrics: &ImageQualityMetrics,
        accuracy_metrics: &MeasurementAccuracy,
        safety_indices: &SafetyIndices,
    ) -> KwaversResult<ClinicalValidationResult> {
        let mut metrics = HashMap::new();
        let mut issues = Vec::new();
        let mut recommendations = Vec::new();

        metrics.insert(
            "contrast_resolution".to_owned(),
            quality_metrics.contrast_resolution,
        );
        metrics.insert(
            "axial_resolution".to_owned(),
            quality_metrics.axial_resolution,
        );
        metrics.insert(
            "lateral_resolution".to_owned(),
            quality_metrics.lateral_resolution,
        );
        metrics.insert("dynamic_range".to_owned(), quality_metrics.dynamic_range);
        metrics.insert("snr".to_owned(), quality_metrics.snr);
        metrics.insert(
            "distance_error".to_owned(),
            accuracy_metrics.distance_error_percent,
        );
        metrics.insert("area_error".to_owned(), accuracy_metrics.area_error_percent);
        metrics.insert(
            "mechanical_index".to_owned(),
            safety_indices.mechanical_index,
        );
        metrics.insert(
            "thermal_index".to_owned(),
            safety_indices.thermal_index_soft,
        );

        let reqs = self
            .requirements
            .get(&(ClinicalStandard::FDA510k, ClinicalCategory::BMode))
            .ok_or_else(|| {
                KwaversError::Validation(
                    kwavers_core::error::ValidationError::ConstraintViolation {
                        message: "FDA 510(k) B-mode requirements not found".to_owned(),
                    },
                )
            })?;

        let mut passed = true;

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

        for (safety_name, max_value) in &reqs.safety_thresholds {
            if let Some(actual_value) = metrics.get(safety_name) {
                if *actual_value > *max_value {
                    passed = false;
                    issues.push(format!(
                        "{} ({:.2}) exceeds safety threshold ({:.2})",
                        safety_name, actual_value, max_value
                    ));
                    recommendations
                        .push("Reduce acoustic output to meet safety requirements".to_owned());
                }
            }
        }

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

    /// Validate Doppler imaging with default thresholds.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
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

    /// Validate Doppler imaging with configurable thresholds.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
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

        metrics.insert("velocity_error".to_owned(), velocity_accuracy);
        metrics.insert("angle_error".to_owned(), angle_accuracy);
        metrics.insert("sensitivity".to_owned(), sensitivity);
        metrics.insert(
            "mechanical_index".to_owned(),
            safety_indices.mechanical_index,
        );

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
                .push("Improve Doppler sensitivity through beamforming optimization".to_owned());
        }
        if velocity_accuracy > max_velocity_error {
            passed = false;
            issues.push(format!(
                "Velocity accuracy ({:.1}%) exceeds maximum error ({:.1}%)",
                velocity_accuracy, max_velocity_error
            ));
            recommendations.push("Calibrate Doppler frequency estimation".to_owned());
        }
        if angle_accuracy > max_angle_error {
            passed = false;
            issues.push(format!(
                "Angle accuracy ({:.1}°) exceeds maximum error ({:.1}°)",
                angle_accuracy, max_angle_error
            ));
            recommendations.push("Improve beam steering angle accuracy".to_owned());
        }

        let clinical_score = (min_sensitivity - sensitivity).max(0.0).mul_add(
            -5.0,
            (velocity_accuracy + angle_accuracy).mul_add(-2.0, 100.0),
        );

        Ok(ClinicalValidationResult {
            passed,
            metrics,
            issues,
            recommendations,
            regulatory_compliant: passed,
            clinical_score: clinical_score.clamp(0.0, 100.0),
        })
    }

    /// Validate safety indices against IEC 60601-2-37.
    /// # Errors
    /// - Propagates any `KwaversError` returned by called functions.
    ///
    pub fn validate_safety(
        &self,
        safety_indices: &SafetyIndices,
    ) -> KwaversResult<ClinicalValidationResult> {
        let mut metrics = HashMap::new();
        let mut issues = Vec::new();
        let mut recommendations = Vec::new();

        metrics.insert(
            "mechanical_index_max".to_owned(),
            safety_indices.mechanical_index,
        );
        metrics.insert(
            "thermal_index_bone_max".to_owned(),
            safety_indices.thermal_index_bone,
        );
        metrics.insert(
            "thermal_index_soft_max".to_owned(),
            safety_indices.thermal_index_soft,
        );
        metrics.insert(
            "thermal_index_cranial_max".to_owned(),
            safety_indices.thermal_index_cranial,
        );
        metrics.insert("spta_max".to_owned(), safety_indices.spta_intensity);
        metrics.insert("sppa_max".to_owned(), safety_indices.sppa_intensity);

        let reqs = self
            .requirements
            .get(&(ClinicalStandard::IEC60601_2_37, ClinicalCategory::Safety))
            .ok_or_else(|| {
                KwaversError::Validation(
                    kwavers_core::error::ValidationError::ConstraintViolation {
                        message: "IEC 60601-2-37 safety requirements not found".to_owned(),
                    },
                )
            })?;

        let mut passed = true;

        for (safety_param, max_value) in &reqs.safety_thresholds {
            if let Some(actual_value) = metrics.get(safety_param) {
                if *actual_value > *max_value {
                    passed = false;
                    issues.push(format!(
                        "{} ({:.2}) exceeds IEC 60601-2-37 limit ({:.2})",
                        safety_param, actual_value, max_value
                    ));
                    let recommendation = match safety_param.as_str() {
                        "mechanical_index_max" => "Reduce peak negative pressure to lower MI",
                        "thermal_index_soft_max" => {
                            "Reduce acoustic power or duty cycle for soft tissue"
                        }
                        "thermal_index_bone_max" => "Reduce acoustic power for bone imaging",
                        "thermal_index_cranial_max" => "Reduce acoustic power for cranial imaging",
                        "spta_max" => "Reduce temporal average intensity",
                        "sppa_max" => "Reduce pulse average intensity",
                        _ => "Reduce acoustic output parameters",
                    };
                    recommendations.push(recommendation.to_owned());
                }
            }
        }

        let clinical_score = if passed { 100.0 } else { 50.0 };

        Ok(ClinicalValidationResult {
            passed,
            metrics,
            issues,
            recommendations,
            regulatory_compliant: passed,
            clinical_score,
        })
    }

    /// Calculate overall clinical acceptability score.
    pub(super) fn calculate_clinical_score(
        &self,
        metrics: &HashMap<String, f64>,
        reqs: &ClinicalRequirements,
    ) -> f64 {
        let mut total_score = 0.0;
        let mut total_weight = 0.0;

        for (metric_name, min_value) in &reqs.minimum_metrics {
            if let Some(actual_value) = metrics.get(metric_name) {
                let score = (actual_value / min_value).min(2.0) * 50.0;
                total_score += score;
                total_weight += 50.0;
            }
        }
        for (error_name, max_error) in &reqs.maximum_errors {
            if let Some(actual_error) = metrics.get(error_name) {
                let error_ratio = (actual_error / max_error).min(2.0);
                let score = (2.0 - error_ratio) * 25.0;
                total_score += score;
                total_weight += 25.0;
            }
        }
        for (safety_name, max_value) in &reqs.safety_thresholds {
            if let Some(actual_value) = metrics.get(safety_name) {
                let score = if *actual_value <= *max_value {
                    25.0
                } else {
                    0.0
                };
                total_score += score;
                total_weight += 25.0;
            }
        }

        if total_weight > 0.0 {
            (total_score / total_weight * 100.0).min(100.0)
        } else {
            0.0
        }
    }
}