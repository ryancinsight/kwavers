//! Enterprise API for PINN Services
//!
//! This module provides RESTful APIs for Physics-Informed Neural Network training
//! and inference with enterprise-grade security, monitoring, and scalability features.
//!
//! ## Architecture
//!
//! The API provides:
//! - RESTful endpoints for PINN operations (training, inference, monitoring)
//! - JWT-based authentication and RBAC authorization
//! - Rate limiting and request validation
//! - Asynchronous job processing
//! - Comprehensive monitoring and metrics
//! - OpenAPI 3.0 specification compliance

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub mod auth;
#[cfg(feature = "pinn")]
pub mod clinical_handlers;
pub mod handlers;
pub mod job_manager;
pub mod middleware;
pub mod model_registry;
pub mod models;
pub mod rate_limiter;
pub mod router;

// ===== CLINICAL ULTRASOUND API TYPE RE-EXPORTS =====
#[cfg(feature = "pinn")]
pub use models::{
    AbnormalRegion,
    AnalysisPriority,
    // Clinical analysis
    ClinicalAnalysisRequest,
    ClinicalAnalysisResponse,
    ClinicalContext,
    ClinicalFinding,
    ClinicalRecommendation,
    ConnectionType,
    // Standards compliance
    DICOMIntegrationRequest,
    DICOMIntegrationResponse,
    DICOMStudyInfo,

    DICOMValue,
    DeviceCapabilities,
    DeviceCapability,
    DeviceInfo,
    DeviceStatus,
    DeviceType,
    FindingMeasurements,
    FindingType,
    ImagingParameters,

    // Mobile optimization
    MobileOptimizationRequest,
    MobileOptimizationResponse,
    NetworkConditions,
    OperatorLevel,
    PerformancePredictions,
    PerformanceTargets,
    PowerEstimates,
    PowerSettings,
    ProcessingConfig,
    // Performance and quality
    ProcessingMetrics,
    QualityIndicators,

    RecommendationType,
    TissueCharacterization,
    TissueProperties,
    TissueRegion,
    // Device integration
    UltrasoundDevice,
    UltrasoundFrame,
    UrgencyLevel,
};

/// API version information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct APIVersion {
    pub version: String,
    pub build_date: String,
    pub commit_hash: String,
}

/// Health check response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheck {
    pub status: HealthStatus,
    pub version: APIVersion,
    pub uptime_seconds: u64,
    pub database: ServiceStatus,
    pub cache: ServiceStatus,
    pub queue: ServiceStatus,
}

/// Service health status
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum HealthStatus {
    #[default]
    Healthy,
    Degraded,
    Unhealthy,
}

/// Individual service status
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ServiceStatus {
    pub status: HealthStatus,
    pub latency_ms: Option<u64>,
    pub error_message: Option<String>,
}

/// Job status enumeration
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum JobStatus {
    #[default]
    Queued,
    Running,
    Completed,
    Failed,
    Cancelled,
}

/// Training job request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PINNTrainingRequest {
    /// Physics domain (e.g., "navier_stokes", "heat_transfer")
    pub physics_domain: String,
    /// Problem geometry specification
    pub geometry: GeometrySpec,
    /// Physics parameters
    pub physics_params: PhysicsParameters,
    /// Training configuration
    pub training_config: TrainingConfig,
    /// Optional callback URL for completion notifications
    pub callback_url: Option<String>,
    /// User-defined metadata
    pub metadata: Option<HashMap<String, serde_json::Value>>,
}

/// Geometry specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeometrySpec {
    /// Domain boundaries [xmin, xmax, ymin, ymax, zmin, zmax]
    pub bounds: Vec<f64>,
    /// Obstacle specifications
    pub obstacles: Vec<ObstacleSpec>,
    /// Boundary conditions
    pub boundary_conditions: Vec<BoundaryConditionSpec>,
}

impl Default for GeometrySpec {
    fn default() -> Self {
        Self {
            bounds: vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0], // Unit cube
            obstacles: Vec::new(),
            boundary_conditions: Vec::new(),
        }
    }
}

/// Obstacle specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObstacleSpec {
    pub shape: String, // "circle", "rectangle", "polygon"
    pub center: Vec<f64>,
    pub parameters: HashMap<String, f64>,
}

/// Boundary condition specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundaryConditionSpec {
    pub boundary: String,       // "left", "right", "top", "bottom", "front", "back"
    pub condition_type: String, // "dirichlet", "neumann", "robin"
    pub value: f64,
}

/// Physics parameters
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PhysicsParameters {
    /// Material properties
    pub material_properties: HashMap<String, f64>,
    /// Boundary values
    pub boundary_values: HashMap<String, f64>,
    /// Initial values
    pub initial_values: HashMap<String, f64>,
    /// Domain parameters
    pub domain_params: HashMap<String, f64>,
}

/// Training configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    /// Number of collocation points
    pub collocation_points: usize,
    /// Batch size
    pub batch_size: usize,
    /// Number of training epochs
    pub epochs: usize,
    /// Learning rate
    pub learning_rate: f64,
    /// Hidden layer configuration
    pub hidden_layers: Vec<usize>,
    /// Use adaptive sampling
    pub adaptive_sampling: bool,
    /// Use GPU acceleration
    pub use_gpu: bool,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            collocation_points: 1000,
            batch_size: 32,
            epochs: 100,
            learning_rate: 0.001,
            hidden_layers: vec![64, 64, 32],
            adaptive_sampling: false,
            use_gpu: false,
        }
    }
}

/// Training job response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PINNTrainingResponse {
    /// Unique job identifier
    pub job_id: String,
    /// Current job status
    pub status: JobStatus,
    /// Estimated completion time
    pub estimated_completion: DateTime<Utc>,
    /// Current progress (if running)
    pub progress: Option<TrainingProgress>,
}

/// Training progress information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingProgress {
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
}

/// Job information response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobInfoResponse {
    pub job_id: String,
    pub status: JobStatus,
    pub created_at: DateTime<Utc>,
    pub started_at: Option<DateTime<Utc>>,
    pub completed_at: Option<DateTime<Utc>>,
    pub progress: Option<TrainingProgress>,
    pub result_url: Option<String>,
    pub error_message: Option<String>,
}

/// Inference request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PINNInferenceRequest {
    /// Model ID to use for inference
    pub model_id: String,
    /// Input coordinates (x, y, t) - shape: [n_points, 3]
    pub coordinates: Vec<Vec<f64>>,
    /// Optional physics parameters override
    pub physics_params: Option<PhysicsParameters>,
}

/// Inference response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PINNInferenceResponse {
    /// Model predictions - shape: [n_points, n_outputs]
    pub predictions: Vec<Vec<f64>>,
    /// Prediction uncertainties (if available)
    pub uncertainties: Option<Vec<Vec<f64>>>,
    /// Processing time (milliseconds)
    pub processing_time_ms: u64,
}

/// Model metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    pub model_id: String,
    pub physics_domain: String,
    pub created_at: DateTime<Utc>,
    pub training_config: TrainingConfig,
    pub performance_metrics: TrainingMetrics,
    pub geometry_spec: GeometrySpec,
}

/// Training metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingMetrics {
    pub final_loss: f64,
    pub best_loss: f64,
    pub total_epochs: usize,
    pub training_time_seconds: u64,
    pub convergence_epoch: Option<usize>,
    pub final_validation_error: Option<f64>,
}

impl Default for TrainingMetrics {
    fn default() -> Self {
        Self {
            final_loss: 0.0,
            best_loss: 0.0,
            total_epochs: 0,
            training_time_seconds: 0,
            convergence_epoch: None,
            final_validation_error: None,
        }
    }
}

#[cfg(feature = "pinn")]
impl From<crate::solver::inverse::pinn::ml::trainer::TrainingMetrics> for TrainingMetrics {
    fn from(metrics: crate::solver::inverse::pinn::ml::trainer::TrainingMetrics) -> Self {
        match metrics {
            crate::solver::inverse::pinn::ml::trainer::TrainingMetrics::OneD(m) => Self {
                final_loss: m.total_loss.last().copied().unwrap_or(0.0),
                best_loss: m
                    .total_loss
                    .iter()
                    .copied()
                    .fold(f64::INFINITY, |acc: f64, value| acc.min(value)),
                total_epochs: m.epochs_completed,
                training_time_seconds: m.training_time_secs as u64,
                convergence_epoch: None,
                final_validation_error: None,
            },
            crate::solver::inverse::pinn::ml::trainer::TrainingMetrics::TwoD(m) => Self {
                final_loss: m.total_loss.last().copied().unwrap_or(0.0),
                best_loss: m
                    .total_loss
                    .iter()
                    .copied()
                    .fold(f64::INFINITY, |acc: f64, value| acc.min(value)),
                total_epochs: m.epochs_completed,
                training_time_seconds: m.training_time_secs as u64,
                convergence_epoch: None,
                final_validation_error: None,
            },
        }
    }
}

/// List models response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListModelsResponse {
    pub models: Vec<ModelMetadata>,
    pub total_count: usize,
    pub page: usize,
    pub page_size: usize,
}

/// API error response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct APIError {
    pub error: APIErrorType,
    pub message: String,
    pub details: Option<HashMap<String, serde_json::Value>>,
}

/// API error types
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum APIErrorType {
    InvalidRequest,
    AuthenticationFailed,
    AuthorizationFailed,
    RateLimitExceeded,
    ResourceNotFound,
    ResourceConflict,
    InternalError,
    ServiceUnavailable,
}

/// Pagination parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PaginationParams {
    pub page: Option<usize>,
    pub page_size: Option<usize>,
    pub sort_by: Option<String>,
    pub sort_order: Option<String>,
}

impl Default for PaginationParams {
    fn default() -> Self {
        Self {
            page: Some(1),
            page_size: Some(50),
            sort_by: Some("created_at".to_string()),
            sort_order: Some("desc".to_string()),
        }
    }
}

/// Rate limit information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitInfo {
    pub limit: usize,
    pub remaining: usize,
    pub reset_time: DateTime<Utc>,
}

/// API configuration
#[derive(Debug, Clone)]
pub struct APIConfig {
    /// Server bind address
    pub bind_address: String,
    /// Server port
    pub port: u16,
    /// JWT secret for authentication
    pub jwt_secret: String,
    /// JWT expiration time (seconds)
    pub jwt_expiration: u64,
    /// Rate limiting configuration
    pub rate_limits: RateLimitConfig,
    /// Request timeout (seconds)
    pub request_timeout: u64,
    /// Maximum request body size (bytes)
    pub max_body_size: usize,
}

/// Rate limiting configuration
#[derive(Debug, Clone)]
pub struct RateLimitConfig {
    /// Requests per minute for anonymous users
    pub anonymous_rpm: usize,
    /// Requests per minute for authenticated users
    pub authenticated_rpm: usize,
    /// Burst allowance
    pub burst_allowance: usize,
}

impl Default for APIConfig {
    fn default() -> Self {
        Self {
            bind_address: "0.0.0.0".to_string(),
            port: 8080,
            jwt_secret: "change-me-in-production".to_string(),
            jwt_expiration: 3600, // 1 hour
            rate_limits: RateLimitConfig {
                anonymous_rpm: 60,
                authenticated_rpm: 600,
                burst_allowance: 10,
            },
            request_timeout: 300,            // 5 minutes
            max_body_size: 10 * 1024 * 1024, // 10MB
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_api_config_defaults() {
        let config = APIConfig::default();
        assert_eq!(config.port, 8080);
        assert_eq!(config.bind_address, "0.0.0.0");
        assert_eq!(config.jwt_expiration, 3600);
    }

    #[test]
    fn test_pagination_defaults() {
        let pagination = PaginationParams::default();
        assert_eq!(pagination.page, Some(1));
        assert_eq!(pagination.page_size, Some(50));
        assert_eq!(pagination.sort_by.as_deref(), Some("created_at"));
        assert_eq!(pagination.sort_order.as_deref(), Some("desc"));
    }

    #[test]
    fn test_training_request_serialization() {
        let request = PINNTrainingRequest {
            physics_domain: "acoustic_wave".to_string(),
            geometry: GeometrySpec {
                bounds: vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
                obstacles: vec![],
                boundary_conditions: vec![],
            },
            physics_params: PhysicsParameters {
                material_properties: HashMap::new(),
                boundary_values: HashMap::new(),
                initial_values: HashMap::new(),
                domain_params: HashMap::new(),
            },
            training_config: TrainingConfig {
                collocation_points: 1000,
                batch_size: 32,
                epochs: 100,
                learning_rate: 0.001,
                hidden_layers: vec![64, 64],
                adaptive_sampling: false,
                use_gpu: true,
            },
            callback_url: None,
            metadata: None,
        };

        let json = serde_json::to_string(&request).unwrap();
        let deserialized: PINNTrainingRequest = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.physics_domain, "acoustic_wave");
        assert_eq!(deserialized.training_config.collocation_points, 1000);
    }

    // ===== CLINICAL ULTRASOUND API TYPE RE-EXPORTS =====

    #[test]
    fn test_health_status_default() {
        assert!(matches!(HealthStatus::default(), HealthStatus::Healthy));
    }

    #[test]
    fn test_service_status_default() {
        let status = ServiceStatus::default();
        assert!(matches!(status.status, HealthStatus::Healthy));
        assert!(status.latency_ms.is_none());
        assert!(status.error_message.is_none());
    }
}
