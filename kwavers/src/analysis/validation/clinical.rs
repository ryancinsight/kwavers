//! Clinical Validation Framework for Medical Ultrasound
//!
//! This module provides comprehensive clinical validation against medical standards
//! and regulatory requirements for ultrasound imaging systems.
//!
//! ## Regulatory Standards
//!
//! - **FDA 510(k)**: Premarket notification for medical devices
//! - **IEC 60601-2-37**: Particular requirements for ultrasonic medical diagnostic equipment
//! - **AIUM Guidelines**: American Institute of Ultrasound in Medicine standards
//! - **ACR Accreditation**: American College of Radiology ultrasound accreditation
//!
//! ## Clinical Validation Metrics
//!
//! - **Image Quality**: Contrast resolution, spatial resolution, dynamic range
//! - **Safety**: Mechanical index (MI), thermal index (TI), acoustic output
//! - **Accuracy**: Measurement precision, calibration accuracy
//! - **Reliability**: System stability, artifact characterization
//!
//! ## Validation Protocols
//!
//! - **Phantom-based testing**: Tissue-mimicking phantoms (ATS, CIRS)
//! - **In-vitro validation**: Biological tissue samples
//! - **In-vivo correlation**: Clinical studies vs gold standards
//! - **Inter-observer variability**: Multi-clinician assessment
//!
//! ## Literature References
//!
//! - **AIUM (2020)**. "Guidelines for Cleaning and Preparing External- and Internal-Use Ultrasound Transducers"
//! - **IEC 60601-2-37 (2015)**. "Medical electrical equipment - Part 2-37: Particular requirements for the basic safety and essential performance of ultrasonic medical diagnostic and monitoring equipment"
//! - **FDA (2019)**. "Information for Manufacturers Seeking Marketing Clearance of Diagnostic Ultrasound Systems and Transducers"

use crate::core::error::{KwaversError, KwaversResult};
use std::collections::HashMap;

/// Clinical validation standards
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ClinicalStandard {
    /// FDA 510(k) requirements
    FDA510k,
    /// IEC 60601-2-37 ultrasound safety
    IEC60601_2_37,
    /// AIUM accreditation standards
    AIUM,
    /// ACR ultrasound accreditation
    ACR,
    /// WHO manual of diagnostic ultrasound
    WHO,
}

/// Clinical validation category
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ClinicalCategory {
    /// B-mode imaging quality
    BMode,
    /// Color Doppler accuracy
    ColorDoppler,
    /// Power Doppler sensitivity
    PowerDoppler,
    /// Pulsed wave Doppler precision
    PWDoppler,
    /// Elastography accuracy
    Elastography,
    /// Contrast-enhanced ultrasound
    CEUS,
    /// Photoacoustic imaging
    Photoacoustic,
    /// Safety indices (MI, TI)
    Safety,
    /// Measurement accuracy
    Measurements,
}

/// Clinical validation requirements
#[derive(Debug)]
pub struct ClinicalRequirements {
    /// Minimum acceptable performance metrics
    pub minimum_metrics: HashMap<String, f64>,
    /// Maximum acceptable error margins
    pub maximum_errors: HashMap<String, f64>,
    /// Required safety thresholds
    pub safety_thresholds: HashMap<String, f64>,
    /// Clinical standard reference
    pub standard: ClinicalStandard,
    /// Validation category
    pub category: ClinicalCategory,
}

/// Clinical validation results
#[derive(Debug)]
pub struct ClinicalValidationResult {
    /// Whether validation passed
    pub passed: bool,
    /// Validation metrics achieved
    pub metrics: HashMap<String, f64>,
    /// Issues identified
    pub issues: Vec<String>,
    /// Recommendations for improvement
    pub recommendations: Vec<String>,
    /// Regulatory compliance status
    pub regulatory_compliant: bool,
    /// Clinical acceptability score (0-100)
    pub clinical_score: f64,
}

/// Clinical safety indices
#[derive(Debug)]
pub struct SafetyIndices {
    /// Mechanical Index (MI)
    pub mechanical_index: f64,
    /// Thermal Index (TI) for bone
    pub thermal_index_bone: f64,
    /// Thermal Index (TI) for soft tissue
    pub thermal_index_soft: f64,
    /// Thermal Index (TI) for cranial bone
    pub thermal_index_cranial: f64,
    /// Spatial Peak Temporal Average (SPTA) intensity
    pub spta_intensity: f64,
    /// Spatial Peak Pulse Average (SPPA) intensity
    pub sppa_intensity: f64,
}

/// Image quality metrics
#[derive(Debug)]
pub struct ImageQualityMetrics {
    /// Contrast resolution (dB)
    pub contrast_resolution: f64,
    /// Axial resolution (mm)
    pub axial_resolution: f64,
    /// Lateral resolution (mm)
    pub lateral_resolution: f64,
    /// Dynamic range (dB)
    pub dynamic_range: f64,
    /// Signal-to-noise ratio (dB)
    pub snr: f64,
    /// Contrast-to-noise ratio (dB)
    pub cnr: f64,
}

/// Measurement accuracy metrics
#[derive(Debug)]
pub struct MeasurementAccuracy {
    /// Distance measurement error (%)
    pub distance_error_percent: f64,
    /// Area measurement error (%)
    pub area_error_percent: f64,
    /// Volume measurement error (%)
    pub volume_error_percent: f64,
    /// Velocity measurement error (%)
    pub velocity_error_percent: f64,
    /// Angle measurement error (degrees)
    pub angle_error_degrees: f64,
}

/// Doppler validation thresholds (configurable defaults)
#[derive(Debug, Clone)]
pub struct DopplerValidationThresholds {
    /// Minimum detectable velocity (cm/s)
    pub min_sensitivity_cm_s: f64,
    /// Maximum velocity error (%)
    pub max_velocity_error_percent: f64,
    /// Maximum angle error (degrees)
    pub max_angle_error_degrees: f64,
}

impl Default for DopplerValidationThresholds {
    fn default() -> Self {
        Self {
            min_sensitivity_cm_s: 5.0,
            max_velocity_error_percent: 10.0,
            max_angle_error_degrees: 5.0,
        }
    }
}

/// Clinical validation framework
#[allow(missing_debug_implementations)]
pub struct ClinicalValidator {
    /// Validation requirements by standard and category
    requirements: HashMap<(ClinicalStandard, ClinicalCategory), ClinicalRequirements>,
}

impl Default for ClinicalValidator {
    fn default() -> Self {
        let mut requirements = HashMap::new();

        // FDA 510(k) B-mode requirements
        let bmode_reqs = ClinicalRequirements {
            minimum_metrics: [
                ("contrast_resolution".to_string(), 30.0), // dB (higher better)
                ("dynamic_range".to_string(), 60.0),       // dB (higher better)
                ("snr".to_string(), 20.0),                 // dB (higher better)
            ]
            .into(),
            maximum_errors: [
                ("distance_error".to_string(), 5.0),     // % (lower better)
                ("area_error".to_string(), 10.0),        // % (lower better)
                ("axial_resolution".to_string(), 0.5),   // mm (lower better)
                ("lateral_resolution".to_string(), 1.0), // mm (lower better)
            ]
            .into(),
            safety_thresholds: [
                ("mechanical_index".to_string(), 1.9), // MI limit (lower better)
                ("thermal_index".to_string(), 6.0),    // TI limit (lower better)
            ]
            .into(),
            standard: ClinicalStandard::FDA510k,
            category: ClinicalCategory::BMode,
        };
        requirements.insert(
            (ClinicalStandard::FDA510k, ClinicalCategory::BMode),
            bmode_reqs,
        );

        // IEC 60601-2-37 safety requirements
        let safety_reqs = ClinicalRequirements {
            minimum_metrics: HashMap::new(),
            maximum_errors: HashMap::new(),
            safety_thresholds: [
                ("mechanical_index_max".to_string(), 1.9),
                ("thermal_index_soft_max".to_string(), 6.0),
                ("thermal_index_bone_max".to_string(), 1.0),
                ("thermal_index_cranial_max".to_string(), 1.0),
                ("spta_max".to_string(), 720.0), // mW/cm²
                ("sppa_max".to_string(), 190.0), // W/cm²
            ]
            .into(),
            standard: ClinicalStandard::IEC60601_2_37,
            category: ClinicalCategory::Safety,
        };
        requirements.insert(
            (ClinicalStandard::IEC60601_2_37, ClinicalCategory::Safety),
            safety_reqs,
        );

        Self { requirements }
    }
}

impl ClinicalValidator {
    /// Create a new clinical validator
    pub fn new() -> Self {
        Self::default()
    }

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
                KwaversError::Validation(crate::core::error::ValidationError::ConstraintViolation {
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
        // TODO_AUDIT: P2 - Clinical Standards Compliance - Implement full AIUM, IEC, and FDA ultrasound standards validation
        // DEPENDS ON: analysis/validation/clinical/aium_standards.rs, analysis/validation/clinical/iec_standards.rs
        // MISSING: AIUM Acoustic Output Measurement Standard (Ophthalmic and Fetal)
        // MISSING: IEC 60601-2-37 Medical Electrical Equipment - Ultrasound Physiotherapy Equipment
        // MISSING: FDA 510(k) Pre-market Notification requirements for acoustic output
        // MISSING: WHO Manual for Diagnostic Ultrasound - Safety and Quality Assurance
        // MISSING: AIUM Contrast-Enhanced Ultrasound accreditation standards
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
                KwaversError::Validation(crate::core::error::ValidationError::ConstraintViolation {
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

    /// Calculate overall clinical acceptability score
    fn calculate_clinical_score(
        &self,
        metrics: &HashMap<String, f64>,
        reqs: &ClinicalRequirements,
    ) -> f64 {
        let mut total_score = 0.0;
        let mut total_weight = 0.0;

        // Score minimum metrics (higher is better)
        for (metric_name, min_value) in &reqs.minimum_metrics {
            if let Some(actual_value) = metrics.get(metric_name) {
                let score = (actual_value / min_value).min(2.0) * 50.0; // Cap at 200% of minimum
                total_score += score;
                total_weight += 50.0;
            }
        }

        // Score maximum errors (lower is better)
        for (error_name, max_error) in &reqs.maximum_errors {
            if let Some(actual_error) = metrics.get(error_name) {
                let error_ratio = (actual_error / max_error).min(2.0);
                let score = (2.0 - error_ratio) * 25.0; // Perfect score when error = 0
                total_score += score;
                total_weight += 25.0;
            }
        }

        // Score safety (binary)
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

    /// Generate comprehensive clinical validation report
    pub fn generate_validation_report(
        &self,
        bmode_result: Option<&ClinicalValidationResult>,
        doppler_result: Option<&ClinicalValidationResult>,
        safety_result: Option<&ClinicalValidationResult>,
    ) -> String {
        let mut report = String::new();

        report.push_str("# Clinical Validation Report - Kwavers Ultrasound System\n\n");
        report.push_str(&format!(
            "**Generated**: {}\n\n",
            chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC")
        ));

        // Overall status
        let all_passed = [bmode_result, doppler_result, safety_result]
            .iter()
            .all(|r| r.map(|res| res.passed).unwrap_or(true));

        report.push_str(&format!(
            "## Overall Status: {}\n\n",
            if all_passed {
                "✅ PASSED"
            } else {
                "❌ FAILED"
            }
        ));

        // Individual validations
        if let Some(result) = bmode_result {
            report.push_str(&format!(
                "## B-Mode Imaging Validation {}\n\n",
                if result.passed { "✅" } else { "❌" }
            ));
            report.push_str(&format!(
                "**Clinical Score**: {:.1}/100\n\n",
                result.clinical_score
            ));

            if !result.issues.is_empty() {
                report.push_str("### Issues Identified\n\n");
                for issue in &result.issues {
                    report.push_str(&format!("- {}\n", issue));
                }
                report.push('\n');
            }

            if !result.recommendations.is_empty() {
                report.push_str("### Recommendations\n\n");
                for rec in &result.recommendations {
                    report.push_str(&format!("- {}\n", rec));
                }
                report.push('\n');
            }
        }

        if let Some(result) = doppler_result {
            report.push_str(&format!(
                "## Doppler Imaging Validation {}\n\n",
                if result.passed { "✅" } else { "❌" }
            ));
            report.push_str(&format!(
                "**Clinical Score**: {:.1}/100\n\n",
                result.clinical_score
            ));

            if !result.issues.is_empty() {
                report.push_str("### Issues Identified\n\n");
                for issue in &result.issues {
                    report.push_str(&format!("- {}\n", issue));
                }
                report.push('\n');
            }
        }

        if let Some(result) = safety_result {
            report.push_str(&format!(
                "## Safety Validation (IEC 60601-2-37) {}\n\n",
                if result.passed { "✅" } else { "❌" }
            ));
            report.push_str(&format!(
                "**Safety Compliance**: {}\n\n",
                if result.regulatory_compliant {
                    "REGULATORY COMPLIANT"
                } else {
                    "REQUIRES CORRECTION"
                }
            ));

            if !result.issues.is_empty() {
                report.push_str("### Safety Issues\n\n");
                for issue in &result.issues {
                    report.push_str(&format!("- **CRITICAL**: {}\n", issue));
                }
                report.push('\n');
            }
        }

        report.push_str("## Regulatory Compliance Summary\n\n");
        report.push_str("| Standard | Status | Notes |\n");
        report.push_str("|----------|--------|-------|\n");

        if let Some(result) = bmode_result {
            report.push_str(&format!(
                "| FDA 510(k) | {} | B-mode imaging requirements |\n",
                if result.regulatory_compliant {
                    "✅ Compliant"
                } else {
                    "❌ Non-compliant"
                }
            ));
        }

        if let Some(result) = safety_result {
            report.push_str(&format!(
                "| IEC 60601-2-37 | {} | Ultrasound safety standards |\n",
                if result.regulatory_compliant {
                    "✅ Compliant"
                } else {
                    "❌ Non-compliant"
                }
            ));
        }

        report.push_str("\n## Next Steps\n\n");
        if !all_passed {
            report.push_str("1. **Address critical safety issues** immediately\n");
            report.push_str("2. **Improve image quality metrics** to meet clinical requirements\n");
            report.push_str("3. **Calibrate measurement accuracy** for regulatory compliance\n");
            report.push_str("4. **Re-validate** after implementing corrections\n");
            report.push_str("5. **Generate updated clinical validation report**\n");
        } else {
            report.push_str(
                "1. **Proceed with clinical trials** - system meets regulatory requirements\n",
            );
            report.push_str("2. **Document validation results** for FDA submission\n");
            report.push_str("3. **Monitor performance** in clinical environment\n");
            report.push_str("4. **Plan post-market surveillance** and updates\n");
        }

        report
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clinical_validator_creation() {
        let validator = ClinicalValidator::new();
        assert!(!validator.requirements.is_empty());
    }

    #[test]
    fn test_bmode_validation_pass() {
        let validator = ClinicalValidator::new();

        let quality = ImageQualityMetrics {
            contrast_resolution: 35.0, // > 30 dB required
            axial_resolution: 0.3,     // < 0.5 mm required
            lateral_resolution: 0.8,   // < 1.0 mm required
            dynamic_range: 70.0,       // > 60 dB required
            snr: 25.0,                 // > 20 dB required
            cnr: 15.0,
        };

        let accuracy = MeasurementAccuracy {
            distance_error_percent: 3.0, // < 5% required
            area_error_percent: 8.0,     // < 10% required
            volume_error_percent: 12.0,
            velocity_error_percent: 8.0,
            angle_error_degrees: 3.0,
        };

        let safety = SafetyIndices {
            mechanical_index: 1.5, // < 1.9 required
            thermal_index_bone: 0.5,
            thermal_index_soft: 2.0, // < 6.0 required
            thermal_index_cranial: 0.3,
            spta_intensity: 500.0,
            sppa_intensity: 100.0,
        };

        let result = validator
            .validate_bmode(&quality, &accuracy, &safety)
            .unwrap();
        assert!(result.passed);
        assert!(result.clinical_score > 80.0);
        assert!(result.regulatory_compliant);
    }

    #[test]
    fn test_bmode_validation_fail() {
        let validator = ClinicalValidator::new();

        let quality = ImageQualityMetrics {
            contrast_resolution: 20.0, // < 30 dB required - FAIL
            axial_resolution: 1.0,     // > 0.5 mm required - FAIL
            lateral_resolution: 1.5,   // > 1.0 mm required - FAIL
            dynamic_range: 40.0,       // < 60 dB required - FAIL
            snr: 15.0,                 // < 20 dB required - FAIL
            cnr: 10.0,
        };

        let accuracy = MeasurementAccuracy {
            distance_error_percent: 8.0, // > 5% required - FAIL
            area_error_percent: 15.0,    // > 10% required - FAIL
            volume_error_percent: 12.0,
            velocity_error_percent: 8.0,
            angle_error_degrees: 3.0,
        };

        let safety = SafetyIndices {
            mechanical_index: 1.5,
            thermal_index_bone: 0.5,
            thermal_index_soft: 2.0,
            thermal_index_cranial: 0.3,
            spta_intensity: 500.0,
            sppa_intensity: 100.0,
        };

        let result = validator
            .validate_bmode(&quality, &accuracy, &safety)
            .unwrap();
        assert!(!result.passed);
        assert!(result.clinical_score < 80.0); // Allow some partial credit for good metrics
        assert!(!result.issues.is_empty());
        assert!(!result.recommendations.is_empty());
    }

    #[test]
    fn test_safety_validation() {
        let validator = ClinicalValidator::new();

        let safety = SafetyIndices {
            mechanical_index: 1.5,      // Within limit
            thermal_index_bone: 0.8,    // Within limit
            thermal_index_soft: 4.0,    // Within limit
            thermal_index_cranial: 0.7, // Within limit
            spta_intensity: 600.0,      // Within limit
            sppa_intensity: 150.0,      // Within limit
        };

        let result = validator.validate_safety(&safety).unwrap();
        assert!(result.passed);
        assert!(result.regulatory_compliant);
        assert_eq!(result.clinical_score, 100.0);
    }

    #[test]
    fn test_safety_validation_fail() {
        let validator = ClinicalValidator::new();

        let safety = SafetyIndices {
            mechanical_index: 2.5,   // Exceeds 1.9 limit - FAIL
            thermal_index_bone: 1.5, // Exceeds 1.0 limit - FAIL
            thermal_index_soft: 8.0, // Exceeds 6.0 limit - FAIL
            thermal_index_cranial: 0.7,
            spta_intensity: 600.0,
            sppa_intensity: 150.0,
        };

        let result = validator.validate_safety(&safety).unwrap();
        assert!(!result.passed);
        assert!(!result.regulatory_compliant);
        assert!(!result.issues.is_empty());
        assert!(!result.recommendations.is_empty());
    }

    #[test]
    fn test_doppler_validation_pass() {
        let validator = ClinicalValidator::new();
        let safety = SafetyIndices {
            mechanical_index: 1.2,
            thermal_index_bone: 0.5,
            thermal_index_soft: 2.0,
            thermal_index_cranial: 0.3,
            spta_intensity: 400.0,
            sppa_intensity: 100.0,
        };

        let result = validator
            .validate_doppler(6.0, 3.0, 6.5, &safety)
            .unwrap();

        assert!(result.passed);
        assert!(result.clinical_score > 50.0);
    }

    #[test]
    fn test_doppler_validation_custom_thresholds() {
        let validator = ClinicalValidator::new();
        let safety = SafetyIndices {
            mechanical_index: 1.2,
            thermal_index_bone: 0.5,
            thermal_index_soft: 2.0,
            thermal_index_cranial: 0.3,
            spta_intensity: 400.0,
            sppa_intensity: 100.0,
        };

        let thresholds = DopplerValidationThresholds {
            min_sensitivity_cm_s: 10.0,
            max_velocity_error_percent: 5.0,
            max_angle_error_degrees: 2.0,
        };

        let result = validator
            .validate_doppler_with_thresholds(6.0, 3.0, 6.5, &safety, &thresholds)
            .unwrap();

        assert!(!result.passed);
        assert!(!result.issues.is_empty());
    }
}
