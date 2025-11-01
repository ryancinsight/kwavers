//! Additional API data models
//!
//! This module contains supplementary data models used by the PINN API
//! for job queuing, result storage, and operational metadata.

use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};
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
    pub training_config: crate::api::TrainingConfig,
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
            metadata: Some(HashMap::from([
                ("duration_ms".to_string(), serde_json::json!(1500)),
            ])),
        };

        let json = serde_json::to_string(&entry).unwrap();
        let deserialized: AuditLogEntry = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.action, "train_pinn");
        assert!(deserialized.success);
    }
}
