//! Additional API data models
//!
//! This module contains supplementary data models used by the PINN API
//! for job queuing, result storage, and operational metadata.
//!
//! ## Clinical Ultrasound API
//!
//! Additional models for point-of-care ultrasound integration:
//! - Device connectivity and real-time imaging
//! - AI-enhanced clinical decision support
//! - DICOM/HL7 standards compliance
//! - Mobile-optimized workflows

use crate::api::TrainingConfig;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Job queue entry for PINN training tasks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobQueueEntry {
    /// Unique job identifier
    pub job_id: String,
    /// User who submitted the job
    pub user_id: String,
    /// Job status
    pub status: crate::api::JobStatus,
    /// Job priority (higher numbers = higher priority)
    pub priority: i32,
    /// Submission timestamp
    pub submitted_at: DateTime<Utc>,
    /// Started timestamp (if running)
    pub started_at: Option<DateTime<Utc>>,
    /// Completed timestamp (if finished)
    pub completed_at: Option<DateTime<Utc>>,
    /// Job configuration
    pub config: PINNJobConfig,
    /// Progress information
    pub progress: Option<JobProgress>,
    /// Error message (if failed)
    pub error_message: Option<String>,
}

/// PINN job configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PINNJobConfig {
    /// Physics domain
    pub physics_domain: String,
    /// Training configuration
    pub training_config: TrainingConfig,
    /// Geometry specification
    pub geometry: crate::api::GeometrySpec,
    /// Physics parameters
    pub physics_params: crate::api::PhysicsParameters,
    /// Callback URL for notifications
    pub callback_url: Option<String>,
    /// User metadata
    pub metadata: Option<HashMap<String, serde_json::Value>>,
}

/// Job progress information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobProgress {
    /// Current epoch
    pub current_epoch: usize,
    /// Total epochs
    pub total_epochs: usize,
    /// Current loss value
    pub current_loss: f64,
    /// Best loss achieved
    pub best_loss: f64,
    /// Training time elapsed (seconds)
    pub elapsed_seconds: u64,
    /// Estimated time remaining (seconds)
    pub estimated_remaining: u64,
    /// GPU memory usage (MB)
    pub gpu_memory_mb: Option<usize>,
    /// CPU usage percentage
    pub cpu_usage_percent: Option<f64>,
}

/// Training result storage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingResult {
    /// Job ID this result belongs to
    pub job_id: String,
    /// Model identifier
    pub model_id: String,
    /// Training completion timestamp
    pub completed_at: DateTime<Utc>,
    /// Final training metrics
    pub metrics: crate::api::TrainingMetrics,
    /// Model artifact location
    pub model_location: String,
    /// Validation results
    pub validation_results: Option<ValidationResults>,
    /// Performance benchmarks
    pub benchmarks: Option<PerformanceBenchmarks>,
}

/// Validation results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResults {
    /// Mean absolute error
    pub mae: f64,
    /// Root mean square error
    pub rmse: f64,
    /// Relative L2 error
    pub relative_l2: f64,
    /// Maximum pointwise error
    pub max_error: f64,
    /// Physics constraint satisfaction (0-1)
    pub physics_satisfaction: f64,
}

/// Performance benchmarks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceBenchmarks {
    /// Training time per epoch (seconds)
    pub time_per_epoch: f64,
    /// Peak memory usage (MB)
    pub peak_memory_mb: usize,
    /// GPU utilization percentage
    pub gpu_utilization_percent: Option<f64>,
    /// Final convergence rate
    pub convergence_rate: f64,
    /// Scalability factor
    pub scalability_factor: f64,
}

/// API usage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct APIUsageStats {
    /// User ID
    pub user_id: String,
    /// Time period start
    pub period_start: DateTime<Utc>,
    /// Time period end
    pub period_end: DateTime<Utc>,
    /// Total API calls
    pub total_calls: usize,
    /// Successful calls
    pub successful_calls: usize,
    /// Failed calls
    pub failed_calls: usize,
    /// Total processing time (seconds)
    pub total_processing_time: f64,
    /// Average response time (milliseconds)
    pub avg_response_time_ms: f64,
    /// Peak concurrent requests
    pub peak_concurrent_requests: usize,
    /// Rate limit hits
    pub rate_limit_hits: usize,
}

/// System health metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemHealthMetrics {
    /// Timestamp of measurement
    pub timestamp: DateTime<Utc>,
    /// CPU usage percentage
    pub cpu_usage_percent: f64,
    /// Memory usage (MB)
    pub memory_usage_mb: usize,
    /// Disk usage percentage
    pub disk_usage_percent: f64,
    /// Network I/O (bytes/second)
    pub network_io_bps: u64,
    /// Active connections
    pub active_connections: usize,
    /// Queue depth
    pub queue_depth: usize,
    /// Error rate (errors per minute)
    pub error_rate_per_minute: f64,
}

/// Audit log entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditLogEntry {
    /// Entry ID
    pub id: String,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// User ID (if applicable)
    pub user_id: Option<String>,
    /// Action performed
    pub action: String,
    /// Resource affected
    pub resource: String,
    /// Resource ID
    pub resource_id: Option<String>,
    /// IP address
    pub ip_address: String,
    /// User agent
    pub user_agent: String,
    /// Success flag
    pub success: bool,
    /// Additional metadata
    pub metadata: Option<HashMap<String, serde_json::Value>>,
}

/// Notification configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotificationConfig {
    /// Notification ID
    pub id: String,
    /// User ID
    pub user_id: String,
    /// Notification type
    pub notification_type: NotificationType,
    /// Delivery method
    pub delivery_method: DeliveryMethod,
    /// Destination (email, webhook URL, etc.)
    pub destination: String,
    /// Enabled flag
    pub enabled: bool,
    /// Created timestamp
    pub created_at: DateTime<Utc>,
}

/// Notification types
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NotificationType {
    JobCompleted,
    JobFailed,
    ModelReady,
    SystemAlert,
}

/// Delivery methods
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DeliveryMethod {
    Email,
    Webhook,
    Slack,
    Sms,
}

impl Default for JobQueueEntry {
    fn default() -> Self {
        Self {
            job_id: String::new(),
            user_id: String::new(),
            status: crate::api::JobStatus::Queued,
            priority: 0,
            submitted_at: Utc::now(),
            started_at: None,
            completed_at: None,
            config: PINNJobConfig {
                physics_domain: String::new(),
                training_config: crate::api::TrainingConfig::default(),
                geometry: crate::api::GeometrySpec {
                    bounds: vec![],
                    obstacles: vec![],
                    boundary_conditions: vec![],
                },
                physics_params: crate::api::PhysicsParameters {
                    material_properties: HashMap::new(),
                    boundary_values: HashMap::new(),
                    initial_values: HashMap::new(),
                    domain_params: HashMap::new(),
                },
                callback_url: None,
                metadata: None,
            },
            progress: None,
            error_message: None,
        }
    }
}

/// Device types for ultrasound systems
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DeviceType {
    /// Standard ultrasound system
    Ultrasound,
    /// Handheld point-of-care device
    Handheld,
    /// Robotic ultrasound probe
    Robotic,
    /// Simulated device for testing
    Simulated,
}

/// Device capabilities for clinical workflow
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DeviceCapability {
    /// 2D B-mode imaging
    Imaging2D,
    /// 3D/4D volumetric imaging
    Imaging3D,
    /// Doppler flow analysis
    Doppler,
    /// Color flow mapping
    ColorFlow,
    /// Elastography tissue characterization
    Elastography,
    /// Contrast-enhanced ultrasound
    ContrastEnhanced,
}

/// Comprehensive ultrasound device information for point-of-care integration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceInfo {
    /// Unique device identifier
    pub id: String,
    /// Type of ultrasound device
    pub device_type: DeviceType,
    /// Device model name
    pub model: String,
    /// Device manufacturer
    pub manufacturer: String,
    /// List of supported clinical capabilities
    pub capabilities: Vec<DeviceCapability>,
    /// Current operational status
    pub status: DeviceStatus,
    /// Timestamp of last calibration
    pub last_calibration: DateTime<Utc>,
    /// Firmware version string
    pub firmware_version: String,
}

/// Ultrasound device information for point-of-care integration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UltrasoundDevice {
    /// Unique device identifier
    pub device_id: String,
    /// Device model/manufacturer
    pub model: String,
    /// Device capabilities (linear, convex, phased array, etc.)
    pub capabilities: Vec<String>,
    /// Supported imaging modes
    pub imaging_modes: Vec<String>,
    /// Maximum frame rate (Hz)
    pub max_frame_rate: u32,
    /// Battery level (0-100)
    pub battery_level: Option<u8>,
    /// Device status
    pub status: DeviceStatus,
    /// Last seen timestamp
    pub last_seen: DateTime<Utc>,
}

/// Device connection status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DeviceStatus {
    Connected,
    Disconnected,
    Error,
    Charging,
    Available,
    InUse,
    Calibrating,
}

/// Real-time ultrasound frame data for AI processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UltrasoundFrame {
    /// Frame sequence number
    pub frame_id: u64,
    /// Device identifier
    pub device_id: String,
    /// Timestamp when frame was captured
    pub timestamp: DateTime<Utc>,
    /// RF data dimensions [time_samples, channels, spatial_points]
    pub dimensions: Vec<usize>,
    /// RF data as base64-encoded bytes
    pub rf_data: String,
    /// Imaging parameters
    pub parameters: ImagingParameters,
    /// Device metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Imaging parameters for beamforming
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImagingParameters {
    /// Sound speed (m/s)
    pub sound_speed: f64,
    /// Sampling frequency (Hz)
    pub sampling_frequency: f64,
    /// Center frequency (Hz)
    pub center_frequency: f64,
    /// Number of active elements
    pub num_elements: usize,
    /// Element spacing (m)
    pub element_spacing: f64,
    /// Steering angles for each frame (radians)
    pub steering_angles: Vec<f64>,
    /// Depth range [start, end] in meters
    pub depth_range: [f64; 2],
}

/// Clinical analysis request for AI-enhanced beamforming
#[derive(Debug, Clone, Serialize, Deserialize)]
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

/// DICOM integration request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DICOMIntegrationRequest {
    /// Study instance UID
    pub study_instance_uid: String,
    /// Series instance UID
    pub series_instance_uid: String,
    /// SOP instance UID
    pub sop_instance_uid: String,
    /// DICOM tags to extract
    pub requested_tags: Vec<String>,
    /// Include pixel data
    pub include_pixel_data: bool,
}

/// DICOM integration response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DICOMIntegrationResponse {
    /// Extracted DICOM metadata
    pub metadata: HashMap<String, DICOMValue>,
    /// Pixel data (if requested) as base64
    pub pixel_data: Option<String>,
    /// Study information
    pub study_info: DICOMStudyInfo,
}

/// DICOM value wrapper
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum DICOMValue {
    String(String),
    Number(f64),
    Integer(i64),
    Sequence(Vec<HashMap<String, DICOMValue>>),
}

/// DICOM study information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DICOMStudyInfo {
    /// Patient ID
    pub patient_id: String,
    /// Study date
    pub study_date: String,
    /// Study description
    pub study_description: String,
    /// Modality
    pub modality: String,
    /// Institution name
    pub institution_name: Option<String>,
}

/// Mobile device optimization request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MobileOptimizationRequest {
    /// Device capabilities
    pub device_capabilities: DeviceCapabilities,
    /// Network conditions
    pub network_conditions: NetworkConditions,
    /// Power management settings
    pub power_settings: PowerSettings,
    /// Target performance requirements
    pub performance_targets: PerformanceTargets,
}

/// Device capabilities for mobile optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceCapabilities {
    /// CPU cores available
    pub cpu_cores: usize,
    /// RAM available (MB)
    pub ram_mb: usize,
    /// GPU available
    pub has_gpu: bool,
    /// SIMD support
    pub has_simd: bool,
    /// Battery capacity (mAh)
    pub battery_mah: Option<u32>,
}

/// Network conditions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkConditions {
    /// Connection type
    pub connection_type: ConnectionType,
    /// Bandwidth (Mbps)
    pub bandwidth_mbps: f64,
    /// Latency (ms)
    pub latency_ms: u64,
    /// Packet loss (%)
    pub packet_loss_percent: f64,
}

/// Connection types
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConnectionType {
    Wifi,
    Cellular4G,
    Cellular5G,
    Ethernet,
    Bluetooth,
}

/// Power management settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PowerSettings {
    /// Battery level (0-100)
    pub battery_level: u8,
    /// Power saving mode enabled
    pub power_saving_mode: bool,
    /// Screen brightness (0-100)
    pub screen_brightness: u8,
    /// Thermal throttling active
    pub thermal_throttling: bool,
}

/// Performance targets for mobile devices
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceTargets {
    /// Target frame rate (Hz)
    pub target_frame_rate_hz: f64,
    /// Maximum latency (ms)
    pub max_latency_ms: u64,
    /// Acceptable image quality (0-1)
    pub acceptable_quality: f32,
    /// Battery usage limit (% per hour)
    pub battery_usage_limit_percent: f64,
}

/// Mobile optimization response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MobileOptimizationResponse {
    /// Recommended processing configuration
    pub recommended_config: ProcessingConfig,
    /// Performance predictions
    pub performance_predictions: PerformancePredictions,
    /// Power consumption estimates
    pub power_estimates: PowerEstimates,
    /// Optimization recommendations
    pub recommendations: Vec<String>,
}

/// Processing configuration for mobile devices
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingConfig {
    /// Frame rate to use (Hz)
    pub frame_rate_hz: f64,
    /// Image resolution scaling factor
    pub resolution_scale: f64,
    /// AI model precision (fp32, fp16, int8)
    pub model_precision: String,
    /// Enable SIMD acceleration
    pub enable_simd: bool,
    /// Batch processing size
    pub batch_size: usize,
}

/// Performance predictions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformancePredictions {
    /// Predicted latency (ms)
    pub predicted_latency_ms: f64,
    /// Predicted frame rate (Hz)
    pub predicted_frame_rate_hz: f64,
    /// Predicted image quality (0-1)
    pub predicted_quality: f32,
    /// Confidence in predictions (0-1)
    pub prediction_confidence: f32,
}

/// Power consumption estimates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PowerEstimates {
    /// CPU usage (%)
    pub cpu_usage_percent: f64,
    /// GPU usage (%)
    pub gpu_usage_percent: Option<f64>,
    /// Battery drain rate (% per hour)
    pub battery_drain_percent_per_hour: f64,
    /// Thermal impact score (0-1)
    pub thermal_impact: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_job_queue_entry_defaults() {
        let entry = JobQueueEntry::default();
        assert_eq!(entry.priority, 0);
        assert!(matches!(entry.status, crate::api::JobStatus::Queued));
        assert!(entry.progress.is_none());
    }

    #[test]
    fn test_training_config_defaults() {
        let config = TrainingConfig::default();
        assert_eq!(config.collocation_points, 1000);
        assert_eq!(config.batch_size, 32);
        assert_eq!(config.epochs, 100);
    }

    #[test]
    fn test_audit_log_serialization() {
        let entry = AuditLogEntry {
            id: "audit_123".to_string(),
            timestamp: Utc::now(),
            user_id: Some("user_456".to_string()),
            action: "train_pinn".to_string(),
            resource: "job".to_string(),
            resource_id: Some("job_789".to_string()),
            ip_address: "192.168.1.1".to_string(),
            user_agent: "PINN-API-Client/1.0".to_string(),
            success: true,
            metadata: Some(HashMap::from([(
                "duration_ms".to_string(),
                serde_json::json!(1500),
            )])),
        };

        let json = serde_json::to_string(&entry).unwrap();
        let deserialized: AuditLogEntry = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.action, "train_pinn");
        assert!(deserialized.success);
    }
}
