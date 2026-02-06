//! Cloud Deployment Domain Types
//!
//! This module defines core domain types for cloud PINN deployments, following
//! Domain-Driven Design principles with clear value objects and entities.
//!
//! # Architecture
//!
//! Types are organized by domain concepts:
//! - **Provider**: Cloud platform abstraction (AWS, GCP, Azure)
//! - **Deployment**: Deployment lifecycle and state
//! - **Metrics**: Performance and health monitoring
//! - **Model**: Model artifact representation
//!
//! # Literature References
//!
//! - Evans, E. (2003). Domain-Driven Design: Tackling Complexity in the Heart of Software.
//!   Addison-Wesley. ISBN: 978-0321125217
//! - Vernon, V. (2013). Implementing Domain-Driven Design. Addison-Wesley. ISBN: 978-0321834577

/// Cloud provider enumeration
///
/// Represents the major cloud platforms supported for PINN deployment.
///
/// # Design
///
/// This is a closed enum representing the specific cloud platforms we integrate with.
/// Each variant corresponds to a specific provider implementation module.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CloudProvider {
    /// Amazon Web Services (SageMaker, Lambda, EC2)
    AWS,
    /// Google Cloud Platform (Vertex AI, Cloud Functions)
    GCP,
    /// Microsoft Azure (Azure ML, Functions)
    Azure,
}

impl CloudProvider {
    /// Get the human-readable name of the provider
    pub fn name(&self) -> &'static str {
        match self {
            CloudProvider::AWS => "Amazon Web Services",
            CloudProvider::GCP => "Google Cloud Platform",
            CloudProvider::Azure => "Microsoft Azure",
        }
    }

    /// Get the short identifier for the provider
    pub fn identifier(&self) -> &'static str {
        match self {
            CloudProvider::AWS => "aws",
            CloudProvider::GCP => "gcp",
            CloudProvider::Azure => "azure",
        }
    }
}

impl std::fmt::Display for CloudProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// Deployment status enumeration
///
/// Represents the lifecycle state of a cloud deployment.
///
/// # State Transitions
///
/// ```text
/// Creating → Active → Scaling → Active
///                   ↓         ↓
///                 Error   Terminating
/// ```
///
/// # Invariants
///
/// - Creating → Active: Deployment successfully initialized
/// - Active → Scaling: Scaling operation initiated
/// - Scaling → Active: Scaling completed successfully
/// - * → Error: Any operation failed
/// - * → Terminating: Explicit termination requested
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeploymentStatus {
    /// Deployment is being created and provisioned
    Creating,
    /// Deployment is active and serving requests
    Active,
    /// Deployment is undergoing a scaling operation
    Scaling,
    /// Deployment encountered an error (with diagnostic message)
    Error(String),
    /// Deployment is being terminated and resources released
    Terminating,
}

impl DeploymentStatus {
    /// Check if deployment is in a healthy state
    pub fn is_healthy(&self) -> bool {
        matches!(self, DeploymentStatus::Active | DeploymentStatus::Scaling)
    }

    /// Check if deployment is terminal (cannot transition further)
    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            DeploymentStatus::Error(_) | DeploymentStatus::Terminating
        )
    }

    /// Check if deployment is operational (can serve requests)
    pub fn is_operational(&self) -> bool {
        matches!(self, DeploymentStatus::Active)
    }
}

impl std::fmt::Display for DeploymentStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DeploymentStatus::Creating => write!(f, "Creating"),
            DeploymentStatus::Active => write!(f, "Active"),
            DeploymentStatus::Scaling => write!(f, "Scaling"),
            DeploymentStatus::Error(msg) => write!(f, "Error: {}", msg),
            DeploymentStatus::Terminating => write!(f, "Terminating"),
        }
    }
}

/// Deployment handle for managing cloud deployments
///
/// Represents a deployed PINN model instance with endpoint and metrics.
///
/// # Design
///
/// This is an entity in DDD terms - it has identity (id) and mutable state (status, metrics).
///
/// # Example
///
/// ```
/// use kwavers::infra::cloud::{DeploymentHandle, CloudProvider, DeploymentStatus};
///
/// let handle = DeploymentHandle {
///     id: "deploy-123".to_string(),
///     provider: CloudProvider::AWS,
///     endpoint: "https://my-model.sagemaker.us-east-1.amazonaws.com".to_string(),
///     status: DeploymentStatus::Active,
///     metrics: None,
/// };
///
/// assert!(handle.status.is_operational());
/// ```
#[derive(Debug, Clone)]
pub struct DeploymentHandle {
    /// Unique deployment identifier (UUID)
    pub id: String,
    /// Cloud provider hosting this deployment
    pub provider: CloudProvider,
    /// Deployment endpoint URL for inference requests
    pub endpoint: String,
    /// Current deployment status
    pub status: DeploymentStatus,
    /// Performance metrics (if available)
    pub metrics: Option<DeploymentMetrics>,
}

impl DeploymentHandle {
    /// Check if deployment is ready to serve requests
    pub fn is_ready(&self) -> bool {
        self.status.is_operational()
    }

    /// Get current instance count (if metrics available)
    pub fn instance_count(&self) -> Option<usize> {
        self.metrics.as_ref().map(|m| m.instance_count)
    }

    /// Get average GPU utilization (if metrics available)
    pub fn gpu_utilization(&self) -> Option<f64> {
        self.metrics.as_ref().map(|m| m.avg_gpu_utilization)
    }
}

/// Deployment performance metrics
///
/// Captures real-time performance and health indicators for a deployment.
///
/// # Metrics
///
/// - **Instance count**: Number of active compute instances
/// - **GPU utilization**: Average GPU usage across instances (0.0-1.0)
/// - **Memory utilization**: Average memory usage across instances (0.0-1.0)
/// - **Response time**: Average inference latency in milliseconds
/// - **Request rate**: Throughput in requests per second
/// - **Error rate**: Percentage of failed requests (0.0-1.0)
///
/// # Literature
///
/// Metrics follow SRE best practices:
/// - Beyer, B., et al. (2016). Site Reliability Engineering: How Google Runs Production Systems. O'Reilly.
/// - Four golden signals: Latency, Traffic, Errors, Saturation
#[derive(Debug, Clone)]
pub struct DeploymentMetrics {
    /// Current number of instances serving the deployment
    pub instance_count: usize,
    /// Average GPU utilization (0.0 = idle, 1.0 = fully utilized)
    pub avg_gpu_utilization: f64,
    /// Average memory utilization (0.0 = empty, 1.0 = full)
    pub avg_memory_utilization: f64,
    /// Average response time for inference requests (milliseconds)
    pub avg_response_time_ms: f64,
    /// Throughput: requests served per second
    pub requests_per_second: f64,
    /// Error rate: fraction of failed requests (0.0-1.0)
    pub error_rate: f64,
}

impl DeploymentMetrics {
    /// Create metrics with zero values
    pub fn zero(instance_count: usize) -> Self {
        Self {
            instance_count,
            avg_gpu_utilization: 0.0,
            avg_memory_utilization: 0.0,
            avg_response_time_ms: 0.0,
            requests_per_second: 0.0,
            error_rate: 0.0,
        }
    }

    /// Check if deployment is under high load
    ///
    /// Returns true if GPU or memory utilization exceeds 80%
    pub fn is_high_load(&self) -> bool {
        self.avg_gpu_utilization > 0.8 || self.avg_memory_utilization > 0.8
    }

    /// Check if deployment has high error rate
    ///
    /// Returns true if error rate exceeds 5%
    pub fn is_unhealthy(&self) -> bool {
        self.error_rate > 0.05
    }

    /// Calculate saturation score (0.0-1.0)
    ///
    /// Combines GPU and memory utilization as a saturation indicator
    pub fn saturation(&self) -> f64 {
        (self.avg_gpu_utilization + self.avg_memory_utilization) / 2.0
    }
}

/// Model deployment data for cloud storage
///
/// Represents a serialized PINN model artifact ready for cloud deployment.
///
/// # Design
///
/// This is a value object representing the model artifact with:
/// - Storage location (URL)
/// - Size metadata
///
/// # Example
///
/// ```
/// use kwavers::infra::cloud::ModelDeploymentData;
///
/// let model_data = ModelDeploymentData {
///     model_url: "s3://kwavers-models/model-123.bin".to_string(),
///     model_size_bytes: 1024 * 1024 * 100, // 100 MB
/// };
/// ```
#[derive(Debug, Clone)]
pub struct ModelDeploymentData {
    /// URL to the serialized model in cloud storage
    ///
    /// Format depends on provider:
    /// - AWS: `s3://bucket/key`
    /// - GCP: `gs://bucket/key`
    /// - Azure: `https://account.blob.core.windows.net/container/blob`
    pub model_url: String,
    /// Size of the serialized model in bytes
    pub model_size_bytes: usize,
}

impl ModelDeploymentData {
    /// Get model size in megabytes
    pub fn size_mb(&self) -> f64 {
        self.model_size_bytes as f64 / (1024.0 * 1024.0)
    }

    /// Get model size in gigabytes
    pub fn size_gb(&self) -> f64 {
        self.model_size_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cloud_provider_display() {
        assert_eq!(CloudProvider::AWS.to_string(), "Amazon Web Services");
        assert_eq!(CloudProvider::GCP.to_string(), "Google Cloud Platform");
        assert_eq!(CloudProvider::Azure.to_string(), "Microsoft Azure");
    }

    #[test]
    fn test_cloud_provider_identifier() {
        assert_eq!(CloudProvider::AWS.identifier(), "aws");
        assert_eq!(CloudProvider::GCP.identifier(), "gcp");
        assert_eq!(CloudProvider::Azure.identifier(), "azure");
    }

    #[test]
    fn test_deployment_status_healthy() {
        assert!(DeploymentStatus::Active.is_healthy());
        assert!(DeploymentStatus::Scaling.is_healthy());
        assert!(!DeploymentStatus::Creating.is_healthy());
        assert!(!DeploymentStatus::Error("test".to_string()).is_healthy());
        assert!(!DeploymentStatus::Terminating.is_healthy());
    }

    #[test]
    fn test_deployment_status_operational() {
        assert!(DeploymentStatus::Active.is_operational());
        assert!(!DeploymentStatus::Scaling.is_operational());
        assert!(!DeploymentStatus::Creating.is_operational());
    }

    #[test]
    fn test_deployment_status_terminal() {
        assert!(DeploymentStatus::Error("test".to_string()).is_terminal());
        assert!(DeploymentStatus::Terminating.is_terminal());
        assert!(!DeploymentStatus::Active.is_terminal());
    }

    #[test]
    fn test_deployment_handle_ready() {
        let handle = DeploymentHandle {
            id: "test-123".to_string(),
            provider: CloudProvider::AWS,
            endpoint: "https://test.example.com".to_string(),
            status: DeploymentStatus::Active,
            metrics: None,
        };

        assert!(handle.is_ready());

        let handle_creating = DeploymentHandle {
            status: DeploymentStatus::Creating,
            ..handle
        };

        assert!(!handle_creating.is_ready());
    }

    #[test]
    fn test_deployment_metrics_zero() {
        let metrics = DeploymentMetrics::zero(3);
        assert_eq!(metrics.instance_count, 3);
        assert_eq!(metrics.avg_gpu_utilization, 0.0);
        assert_eq!(metrics.requests_per_second, 0.0);
    }

    #[test]
    fn test_deployment_metrics_high_load() {
        let metrics = DeploymentMetrics {
            instance_count: 5,
            avg_gpu_utilization: 0.85,
            avg_memory_utilization: 0.5,
            avg_response_time_ms: 100.0,
            requests_per_second: 50.0,
            error_rate: 0.01,
        };

        assert!(metrics.is_high_load());
        assert!(!metrics.is_unhealthy());
    }

    #[test]
    fn test_deployment_metrics_saturation() {
        let metrics = DeploymentMetrics {
            instance_count: 5,
            avg_gpu_utilization: 0.8,
            avg_memory_utilization: 0.6,
            avg_response_time_ms: 100.0,
            requests_per_second: 50.0,
            error_rate: 0.01,
        };

        assert_eq!(metrics.saturation(), 0.7);
    }

    #[test]
    fn test_model_deployment_data_size_conversions() {
        let data = ModelDeploymentData {
            model_url: "s3://bucket/model.bin".to_string(),
            model_size_bytes: 100 * 1024 * 1024, // 100 MB
        };

        assert!((data.size_mb() - 100.0).abs() < 0.01);
        assert!((data.size_gb() - 0.09765625).abs() < 0.001);
    }

    #[test]
    fn test_deployment_handle_accessors() {
        let metrics = DeploymentMetrics::zero(3);
        let handle = DeploymentHandle {
            id: "test-123".to_string(),
            provider: CloudProvider::AWS,
            endpoint: "https://test.example.com".to_string(),
            status: DeploymentStatus::Active,
            metrics: Some(metrics),
        };

        assert_eq!(handle.instance_count(), Some(3));
        assert_eq!(handle.gpu_utilization(), Some(0.0));
    }
}
