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
//! ## Module layout
//!
//! - [`bmode`]: B-mode imaging validation against FDA 510(k) image-quality
//!   and measurement-accuracy requirements.
//! - [`doppler`]: Doppler imaging validation with default and configurable
//!   sensitivity / velocity-accuracy / angle-accuracy thresholds.
//! - [`safety`]: IEC 60601-2-37 acoustic-output safety validation.
//! - [`score`]: clinical acceptability scoring kernel shared by B-mode.
//! - [`report`]: Markdown-rendered consolidated validation report across
//!   B-mode, Doppler, and safety results.
//!
//! ## Literature References
//!
//! - **AIUM (2020)**. "Guidelines for Cleaning and Preparing External- and Internal-Use Ultrasound Transducers"
//! - **IEC 60601-2-37 (2015)**. "Medical electrical equipment - Part 2-37: Particular requirements for the basic safety and essential performance of ultrasonic medical diagnostic and monitoring equipment"
//! - **FDA (2019)**. "Information for Manufacturers Seeking Marketing Clearance of Diagnostic Ultrasound Systems and Transducers"

mod bmode;
mod doppler;
mod report;
mod safety;
mod score;

#[cfg(test)]
mod tests;

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
    pub(super) requirements: HashMap<(ClinicalStandard, ClinicalCategory), ClinicalRequirements>,
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
}
