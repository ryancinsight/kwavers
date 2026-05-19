//! API request/response types and configuration for PINN services.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// API version information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct APIVersion {
    pub version: String,
    pub build_date: String,
    pub commit_hash: String,
}

/// Health check response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheck {
    pub status: HealthStatus,
    pub version: APIVersion,
    pub uptime_seconds: u64,
    pub database: ServiceStatus,
    pub cache: ServiceStatus,
    pub queue: ServiceStatus,
}

/// Service health status.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum HealthStatus {
    #[default]
    Healthy,
    Degraded,
    Unhealthy,
}

/// Individual service status.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ServiceStatus {
    pub status: HealthStatus,
    pub latency_ms: Option<u64>,
    pub error_message: Option<String>,
}

/// Job status enumeration.
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

/// Training job request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PINNTrainingRequest {
    pub physics_domain: String,
    pub geometry: GeometrySpec,
    pub physics_params: PinnApiPhysicsParameters,
    pub training_config: PinnApiTrainingConfig,
    pub callback_url: Option<String>,
    pub metadata: Option<HashMap<String, serde_json::Value>>,
}

/// Geometry specification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeometrySpec {
    pub bounds: Vec<f64>,
    pub obstacles: Vec<ObstacleSpec>,
    pub boundary_conditions: Vec<ApiBoundaryConditionSpec>,
}

impl Default for GeometrySpec {
    fn default() -> Self {
        Self {
            bounds: vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
            obstacles: Vec::new(),
            boundary_conditions: Vec::new(),
        }
    }
}

/// Obstacle specification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObstacleSpec {
    pub shape: String,
    pub center: Vec<f64>,
    pub parameters: HashMap<String, f64>,
}

/// Boundary condition specification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiBoundaryConditionSpec {
    pub boundary: String,
    pub condition_type: String,
    pub value: f64,
}

/// Physics parameters.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PinnApiPhysicsParameters {
    pub material_properties: HashMap<String, f64>,
    pub boundary_values: HashMap<String, f64>,
    pub initial_values: HashMap<String, f64>,
    pub domain_params: HashMap<String, f64>,
}

/// Training configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PinnApiTrainingConfig {
    pub collocation_points: usize,
    pub batch_size: usize,
    pub epochs: usize,
    pub learning_rate: f64,
    pub hidden_layers: Vec<usize>,
    pub adaptive_sampling: bool,
    pub use_gpu: bool,
}

impl Default for PinnApiTrainingConfig {
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

/// Training job response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PINNTrainingResponse {
    pub job_id: String,
    pub status: JobStatus,
    pub estimated_completion: DateTime<Utc>,
    pub progress: Option<TrainingProgress>,
}

/// Training progress information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingProgress {
    pub current_epoch: usize,
    pub total_epochs: usize,
    pub current_loss: f64,
    pub best_loss: f64,
    pub elapsed_seconds: u64,
    pub estimated_remaining: u64,
}

/// Job information response.
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

/// Inference request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PINNInferenceRequest {
    pub model_id: String,
    pub coordinates: Vec<Vec<f64>>,
    pub physics_params: Option<PinnApiPhysicsParameters>,
}

/// Inference response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PINNInferenceResponse {
    pub predictions: Vec<Vec<f64>>,
    pub uncertainties: Option<Vec<Vec<f64>>>,
    pub processing_time_ms: u64,
}

/// Model metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiModelMetadata {
    pub model_id: String,
    pub physics_domain: String,
    pub created_at: DateTime<Utc>,
    pub training_config: PinnApiTrainingConfig,
    pub performance_metrics: PinnApiTrainingMetrics,
    pub geometry_spec: GeometrySpec,
}

/// Training metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PinnApiTrainingMetrics {
    pub final_loss: f64,
    pub best_loss: f64,
    pub total_epochs: usize,
    pub training_time_seconds: u64,
    pub convergence_epoch: Option<usize>,
    pub final_validation_error: Option<f64>,
}

impl Default for PinnApiTrainingMetrics {
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
impl From<crate::solver::inverse::pinn::ml::trainer::BurnPinnTrainingMetrics>
    for PinnApiTrainingMetrics
{
    fn from(metrics: crate::solver::inverse::pinn::ml::trainer::BurnPinnTrainingMetrics) -> Self {
        match metrics {
            crate::solver::inverse::pinn::ml::trainer::BurnPinnTrainingMetrics::OneD(m) => Self {
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
            crate::solver::inverse::pinn::ml::trainer::BurnPinnTrainingMetrics::TwoD(m) => Self {
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

/// List models response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListModelsResponse {
    pub models: Vec<ApiModelMetadata>,
    pub total_count: usize,
    pub page: usize,
    pub page_size: usize,
}

/// API error response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct APIError {
    pub error: APIErrorType,
    pub message: String,
    pub details: Option<HashMap<String, serde_json::Value>>,
}

/// API error types.
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

/// Pagination parameters.
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

/// Rate limit information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitInfo {
    pub limit: usize,
    pub remaining: usize,
    pub reset_time: DateTime<Utc>,
}

/// API configuration.
#[derive(Debug, Clone)]
pub struct APIConfig {
    pub bind_address: String,
    pub port: u16,
    pub jwt_secret: String,
    pub jwt_expiration: u64,
    pub rate_limits: ApiRateLimitConfig,
    pub request_timeout: u64,
    pub max_body_size: usize,
}

/// Rate limiting configuration.
#[derive(Debug, Clone)]
pub struct ApiRateLimitConfig {
    pub anonymous_rpm: usize,
    pub authenticated_rpm: usize,
    pub burst_allowance: usize,
}

impl Default for APIConfig {
    fn default() -> Self {
        Self {
            bind_address: "0.0.0.0".to_string(),
            port: 8080,
            jwt_secret: "change-me-in-production".to_string(),
            jwt_expiration: 3600,
            rate_limits: ApiRateLimitConfig {
                anonymous_rpm: 60,
                authenticated_rpm: 600,
                burst_allowance: 10,
            },
            request_timeout: 300,
            max_body_size: 10 * 1024 * 1024,
        }
    }
}
