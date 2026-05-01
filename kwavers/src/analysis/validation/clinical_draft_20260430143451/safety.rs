//! IEC 60601-2-37 acoustic-output safety validation.

use std::collections::HashMap;

use super::{
    ClinicalCategory, ClinicalStandard, ClinicalValidationResult, ClinicalValidator, SafetyIndices,
};
use crate::core::error::{KwaversError, KwaversResult, ValidationError};

impl ClinicalValidator {
    /// Validate safety indices against IEC 60601-2-37
    pub fn validate_safety(
        &self,
        safety_indices: &SafetyIndices,
    ) -> KwaversResult<ClinicalValidationResult> {
        let mut metrics = HashMap::new();
        let mut issues = Vec::new();
        let mut recommendations = Vec::new();

        // Extract all safety metrics
        metrics.insert(
            "mechanical_index_max".to_string(),
            safety_indices.mechanical_index,
        );
        metrics.insert(
            "thermal_index_bone_max".to_string(),
            safety_indices.thermal_index_bone,
        );
        metrics.insert(
            "thermal_index_soft_max".to_string(),
            safety_indices.thermal_index_soft,
        );
        metrics.insert(
            "thermal_index_cranial_max".to_string(),
            safety_indices.thermal_index_cranial,
        );
        metrics.insert("spta_max".to_string(), safety_indices.spta_intensity);
        metrics.insert("sppa_max".to_string(), safety_indices.sppa_intensity);

        let reqs = self
            .requirements
            .get(&(ClinicalStandard::IEC60601_2_37, ClinicalCategory::Safety))
            .ok_or_else(|| {
                KwaversError::Validation(ValidationError::ConstraintViolation {
                    message: "IEC 60601-2-37 safety requirements not found".to_string(),
                })
            })?;

        let mut passed = true;

        // Check all safety thresholds
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
                    recommendations.push(recommendation.to_string());
                }
            }
        }

        let clinical_score = if passed { 100.0 } else { 50.0 }; // Binary for safety

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
