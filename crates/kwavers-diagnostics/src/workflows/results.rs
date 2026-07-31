use crate::photoacoustic::PhotoacousticResult;
use aequitas::systems::si::quantities::{Dimensionless, Time};
use kwavers_imaging::fusion::FusedImageResult;
use kwavers_imaging::ultrasound::elastography::ElasticityMap;
use leto::Array3;
use std::collections::HashMap;

/// Clinical examination result
#[derive(Debug)]
pub struct ClinicalExaminationResult {
    /// Patient identifier
    pub patient_id: String,
    /// Examination timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Fused multi-modal image
    pub fused_image: FusedImageResult,
    /// Tissue classification map
    pub tissue_classification: HashMap<String, Array3<f64>>,
    /// Diagnostic recommendations
    pub diagnostic_recommendations: Vec<DiagnosticRecommendation>,
    /// Quality metrics for each modality
    pub quality_metrics: HashMap<String, f64>,
    /// Processing performance metrics
    pub performance_metrics: WorkflowTimingMetrics,
    /// Clinical confidence score as a percentage (0-100).
    pub confidence_score: Dimensionless<f64>,
}

/// Diagnostic recommendation
#[derive(Debug, Clone)]
pub struct DiagnosticRecommendation {
    /// Finding description
    pub finding: String,
    /// Confidence level as a percentage (0-100).
    pub confidence: Dimensionless<f64>,
    /// Recommended follow-up actions
    pub recommendations: Vec<String>,
    /// Urgency level
    pub urgency: DiagnosticUrgency,
    /// Supporting evidence
    pub evidence: Vec<String>,
}

/// Diagnostic urgency levels
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DiagnosticUrgency {
    /// Immediate intervention required
    Critical,
    /// Urgent follow-up needed
    Urgent,
    /// Standard clinical follow-up
    Routine,
    /// No immediate action required
    Normal,
}

/// Performance metrics for clinical workflows
#[derive(Debug, Clone)]
pub struct WorkflowTimingMetrics {
    /// Total examination time as an SI time quantity.
    pub total_time: Time<f64>,
    /// Time per processing stage as SI time quantities.
    pub stage_times: HashMap<String, Time<f64>>,
    /// Measured GPU utilization percentage, when a telemetry provider supplies it.
    pub gpu_utilization: Option<Dimensionless<f64>>,
    /// Measured resident memory in bytes, when a telemetry provider supplies it.
    ///
    /// Aequitas does not define an information dimension, so bytes remain at
    /// this explicit storage-instrumentation boundary.
    pub memory_usage_bytes: Option<u64>,
    /// Real-time constraint satisfaction
    pub real_time_satisfied: bool,
}

/// Acquisition result from multi-modal scanning
#[derive(Debug)]
pub struct AcquisitionResult {
    /// Acquired ultrasound volume.
    pub ultrasound_data: Array3<f64>,
    /// Acquired photoacoustic result.
    pub photoacoustic_result: PhotoacousticResult,
    /// Acquired elastography map.
    pub elastography_result: ElasticityMap,
    /// Acquisition duration as an SI time quantity.
    pub acquisition_time: Time<f64>,
}

/// Processing result after real-time processing
#[derive(Debug)]
pub struct ProcessingResult {
    /// Quality metrics produced during processing.
    pub quality_metrics: HashMap<String, f64>,
    /// Processing duration as an SI time quantity.
    pub processing_time: Time<f64>,
}

/// AI analysis result
#[derive(Debug)]
pub struct AnalysisResult {
    /// Tissue properties extracted from the fused result.
    pub tissue_properties: HashMap<String, Array3<f64>>,
    /// Diagnostic recommendations derived from the tissue properties.
    pub recommendations: Vec<DiagnosticRecommendation>,
    /// Analysis confidence as a percentage (0-100).
    pub confidence_score: Dimensionless<f64>,
}
