use crate::clinical::imaging::photoacoustic::PhotoacousticResult;
use crate::domain::imaging::ultrasound::elastography::ElasticityMap;
use crate::physics::imaging::fusion::FusedImageResult;
use ndarray::Array3;
use std::collections::HashMap;
use std::time::Duration;

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
    pub performance_metrics: PerformanceMetrics,
    /// Clinical confidence score (0-100)
    pub confidence_score: f64,
}

/// Diagnostic recommendation
#[derive(Debug, Clone)]
pub struct DiagnosticRecommendation {
    /// Finding description
    pub finding: String,
    /// Confidence level (0-100)
    pub confidence: f64,
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
pub struct PerformanceMetrics {
    /// Total examination time
    pub total_time: Duration,
    /// Time per processing stage
    pub stage_times: HashMap<String, Duration>,
    /// GPU utilization percentage
    pub gpu_utilization: f64,
    /// Memory usage (MB)
    pub memory_usage_mb: f64,
    /// Real-time constraint satisfaction
    pub real_time_satisfied: bool,
}

/// Acquisition result from multi-modal scanning
#[derive(Debug)]
pub struct AcquisitionResult {
    pub ultrasound_data: Array3<f64>,
    pub photoacoustic_result: PhotoacousticResult,
    pub elastography_result: ElasticityMap,
    #[allow(dead_code)]
    pub acquisition_time: Duration,
}

/// Processing result after real-time processing
#[derive(Debug)]
pub struct ProcessingResult {
    #[allow(dead_code)]
    pub quality_metrics: HashMap<String, f64>,
    #[allow(dead_code)]
    pub processing_time: Duration,
}

/// AI analysis result
#[derive(Debug)]
pub struct AnalysisResult {
    pub tissue_properties: HashMap<String, Array3<f64>>,
    pub recommendations: Vec<DiagnosticRecommendation>,
    pub confidence_score: f64,
}
