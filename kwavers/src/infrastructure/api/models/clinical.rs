//! Clinical analysis request/response, findings, tissue characterization.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub struct ClinicalAnalysisRequest {
    /// Session identifier
    pub session_id: String,
    /// Device identifier
    pub device_id: String,
    /// Patient identifier (anonymized)
    pub patient_id: String,
    /// Exam type (cardiac, abdominal, vascular, etc.)
    pub exam_type: String,
    /// Priority level (normal, urgent, critical)
    pub priority: AnalysisPriority,
    /// Frame data for analysis
    pub frames: Vec<UltrasoundFrame>,
    /// Clinical context
    pub clinical_context: ClinicalContext,
}

/// Analysis priority levels
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AnalysisPriority {
    Normal,
    Urgent,
    Critical,
}

/// Clinical context information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClinicalContext {
    /// Patient age
    pub patient_age: Option<u32>,
    /// Patient sex
    pub patient_sex: Option<String>,
    /// Indication/symptoms
    pub indications: Vec<String>,
    /// Previous findings
    pub previous_findings: Option<String>,
    /// Operator experience level
    pub operator_level: OperatorLevel,
}

/// Operator experience levels
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OperatorLevel {
    Trainee,
    Resident,
    Attending,
    Specialist,
}

/// Clinical analysis response with AI-enhanced results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClinicalAnalysisResponse {
    /// Analysis session identifier
    pub session_id: String,
    /// Processing timestamp
    pub processed_at: DateTime<Utc>,
    /// Overall analysis confidence (0-1)
    pub confidence_score: f32,
    /// Detected findings
    pub findings: Vec<ClinicalFinding>,
    /// Tissue characterization results
    pub tissue_characterization: TissueCharacterization,
    /// Clinical recommendations
    pub recommendations: Vec<ClinicalRecommendation>,
    /// Processing performance metrics
    pub performance: ProcessingMetrics,
    /// Quality indicators
    pub quality_indicators: QualityIndicators,
}

/// Clinical finding with localization and confidence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClinicalFinding {
    /// Finding type
    pub finding_type: FindingType,
    /// Finding description
    pub description: String,
    /// Confidence score (0-1)
    pub confidence: f32,
    /// Spatial location [x, y, z] in mm
    pub location: [f64; 3],
    /// Size measurements in mm
    pub measurements: FindingMeasurements,
    /// Clinical significance score (0-1)
    pub clinical_significance: f32,
}

/// Types of clinical findings
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FindingType {
    Lesion,
    FluidCollection,
    Calcification,
    VascularAnomaly,
    TissueAbnormality,
    NormalFinding,
}

/// Finding measurements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FindingMeasurements {
    /// Maximum diameter (mm)
    pub max_diameter_mm: f64,
    /// Area/volume measurement (mm² or mm³)
    pub area_volume: f64,
    /// Boundary confidence (0-1)
    pub boundary_confidence: f32,
}

/// Tissue characterization results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TissueCharacterization {
    /// Dominant tissue types per region
    pub tissue_map: HashMap<String, TissueRegion>,
    /// Overall tissue composition percentages
    pub composition_percentages: HashMap<String, f32>,
    /// Tissue homogeneity score (0-1, higher = more homogeneous)
    pub homogeneity_score: f32,
    /// Abnormal tissue regions
    pub abnormal_regions: Vec<AbnormalRegion>,
}

/// Tissue region information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TissueRegion {
    /// Tissue type
    pub tissue_type: String,
    /// Region boundaries [xmin, xmax, ymin, ymax, zmin, zmax] in mm
    pub boundaries: [f64; 6],
    /// Classification confidence (0-1)
    pub confidence: f32,
    /// Tissue properties
    pub properties: TissueProperties,
}

/// Tissue physical properties
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TissueProperties {
    /// Attenuation coefficient (dB/cm/MHz)
    pub attenuation_db_per_cm_mhz: f64,
    /// Backscatter coefficient
    pub backscatter_coefficient: f64,
    /// Speed of sound (m/s)
    pub speed_of_sound: f64,
}

/// Abnormal tissue region
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AbnormalRegion {
    /// Region identifier
    pub region_id: String,
    /// Location [x, y, z] in mm
    pub location: [f64; 3],
    /// Abnormality type
    pub abnormality_type: String,
    /// Severity score (0-1)
    pub severity: f32,
    /// Confidence score (0-1)
    pub confidence: f32,
}

/// Clinical recommendation with rationale
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClinicalRecommendation {
    /// Recommendation type
    pub recommendation_type: RecommendationType,
    /// Recommendation text
    pub text: String,
    /// Rationale for recommendation
    pub rationale: String,
    /// Urgency level
    pub urgency: UrgencyLevel,
    /// Supporting evidence from AI analysis
    pub supporting_evidence: Vec<String>,
}

/// Types of clinical recommendations
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RecommendationType {
    FollowUpImaging,
    BiopsyConsideration,
    SpecialistReferral,
    ProtocolAdjustment,
    NormalFindings,
    UrgentEvaluation,
}

/// Recommendation urgency levels
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum UrgencyLevel {
    Routine,
    Priority,
    Urgent,
    Emergency,
}

/// Processing performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingMetrics {
    /// Total processing time (ms)
    pub total_time_ms: u64,
    /// AI inference time (ms)
    pub ai_inference_time_ms: u64,
    /// Beamforming time (ms)
    pub beamforming_time_ms: u64,
    /// Feature extraction time (ms)
    pub feature_extraction_time_ms: u64,
    /// Memory usage (MB)
    pub memory_usage_mb: f64,
    /// GPU utilization (%)
    pub gpu_utilization_percent: Option<f64>,
}

/// Quality indicators for clinical workflow
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityIndicators {
    /// Image quality score (0-1)
    pub image_quality: f32,
    /// Motion artifact level (0-1, lower = better)
    pub motion_artifacts: f32,
    /// Acoustic shadowing (0-1)
    pub acoustic_shadowing: f32,
    /// Signal-to-noise ratio
    pub snr: f64,
    /// Frame rate achieved (Hz)
    pub frame_rate_hz: f64,
}
