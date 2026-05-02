//! Plain data types for the clinical validation framework.
//!
//! SRP: changes when the API data shapes change.

use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ClinicalStandard {
    FDA510k,
    IEC60601_2_37,
    AIUM,
    ACR,
    WHO,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ClinicalCategory {
    BMode,
    ColorDoppler,
    PowerDoppler,
    PWDoppler,
    Elastography,
    CEUS,
    Photoacoustic,
    Safety,
    Measurements,
}

#[derive(Debug)]
pub struct ClinicalRequirements {
    pub minimum_metrics: HashMap<String, f64>,
    pub maximum_errors: HashMap<String, f64>,
    pub safety_thresholds: HashMap<String, f64>,
    pub standard: ClinicalStandard,
    pub category: ClinicalCategory,
}

#[derive(Debug)]
pub struct ClinicalValidationResult {
    pub passed: bool,
    pub metrics: HashMap<String, f64>,
    pub issues: Vec<String>,
    pub recommendations: Vec<String>,
    pub regulatory_compliant: bool,
    pub clinical_score: f64,
}

#[derive(Debug)]
pub struct SafetyIndices {
    pub mechanical_index: f64,
    pub thermal_index_bone: f64,
    pub thermal_index_soft: f64,
    pub thermal_index_cranial: f64,
    pub spta_intensity: f64,
    pub sppa_intensity: f64,
}

#[derive(Debug)]
pub struct ImageQualityMetrics {
    pub contrast_resolution: f64,
    pub axial_resolution: f64,
    pub lateral_resolution: f64,
    pub dynamic_range: f64,
    pub snr: f64,
    pub cnr: f64,
}

#[derive(Debug)]
pub struct MeasurementAccuracy {
    pub distance_error_percent: f64,
    pub area_error_percent: f64,
    pub volume_error_percent: f64,
    pub velocity_error_percent: f64,
    pub angle_error_degrees: f64,
}

#[derive(Debug, Clone)]
pub struct DopplerValidationThresholds {
    pub min_sensitivity_cm_s: f64,
    pub max_velocity_error_percent: f64,
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
