//! Cloud Integration Module for PINN Deployment
//!
//! This module provides cloud-agnostic deployment capabilities for Physics-Informed Neural Networks,
//! enabling seamless deployment across major cloud platforms (AWS, GCP, Azure) with enterprise features.
//!
//! # Architecture
//!
//! The module follows Clean Architecture principles with clear layer separation:
//!
//! ## Domain Layer
//! - **types**: Core domain types (CloudProvider, DeploymentStatus, DeploymentHandle, etc.)
//! - **config**: Configuration value objects with validation
//!
//! ## Application Layer
//! - **service**: CloudPINNService orchestrator for deployment use cases
//!
//! ## Infrastructure Layer
//! - **providers**: Provider-specific implementations (AWS, GCP, Azure)
//! - **utilities**: Configuration loading and model serialization
//!
//! ## Interface Layer
//! - This module (mod.rs): Public API surface
//!
//! # Design Patterns
//!
//! - **Clean Architecture**: Dependency inversion, layer isolation
//! - **Strategy**: Provider-specific deployment strategies
//! - **Facade**: Unified interface hiding provider complexity
//! - **Builder**: Configuration construction with defaults
//!
//! # Features
//!
//! ## Multi-Cloud Support
//! - AWS SageMaker with ELB and Auto Scaling
//! - GCP Vertex AI with Cloud Load Balancing
//! - Azure Machine Learning with Azure Functions
//!
//! ## Auto-Scaling
//! - Dynamic instance scaling based on GPU utilization
//! - Configurable scale-up/scale-down thresholds
//! - Cooldown periods to prevent flapping
//!
//! ## Monitoring
//! - Real-time performance metrics (GPU, memory, latency)
//! - Configurable alert thresholds
//! - Health status tracking
//!
//! # Example Usage
//!
//! ## Basic Deployment
//!
//! ```ignore
//! use kwavers::infra::cloud::{
//!     CloudPINNService, CloudProvider, DeploymentConfig,
//!     AutoScalingConfig, MonitoringConfig
//! };
//! use kwavers::ml::pinn::BurnPINN2DWave;
//!
//! # #[cfg(feature = "pinn")]
//! # async fn example<B: burn::tensor::backend::AutodiffBackend>(model: &BurnPINN2DWave<B>) {
//! // Initialize cloud service
//! let mut service = CloudPINNService::new(CloudProvider::AWS).await.unwrap();
//!
//! // Configure deployment
//! let config = DeploymentConfig {
//!     provider: CloudProvider::AWS,
//!     region: "us-east-1".to_string(),
//!     instance_type: "p3.2xlarge".to_string(),
//!     gpu_count: 1,
//!     memory_gb: 16,
//!     auto_scaling: AutoScalingConfig::default(),
//!     monitoring: MonitoringConfig::default(),
//! };
//!
//! // Deploy model
//! let handle = service.deploy_model(model, config).await.unwrap();
//! println!("Deployed to: {}", handle.endpoint);
//!
//! // Check status
//! let status = service.get_deployment_status(&handle.id).await.unwrap();
//! println!("Status: {:?}", status);
//!
//! // Scale deployment
//! service.scale_deployment(&handle.id, 5).await.unwrap();
//!
//! // Terminate when done
//! service.terminate_deployment(&handle.id).await.unwrap();
//! # }
//! ```
//!
//! ## Custom Configuration
//!
//! ```
//! use kwavers::infra::cloud::{AutoScalingConfig, MonitoringConfig, AlertThresholds};
//!
//! let auto_scaling = AutoScalingConfig {
//!     min_instances: 2,
//!     max_instances: 20,
//!     target_gpu_utilization: 0.8,
//!     scale_up_threshold: 0.9,
//!     scale_down_threshold: 0.4,
//!     cooldown_seconds: 600,
//! };
//!
//! let monitoring = MonitoringConfig {
//!     enable_detailed_metrics: true,
//!     metrics_interval_seconds: 30,
//!     alert_thresholds: AlertThresholds {
//!         gpu_utilization_threshold: 0.95,
//!         memory_usage_threshold: 0.95,
//!         error_rate_threshold: 0.02,
//!     },
//! };
//! ```
//!
//! # Provider-Specific Requirements
//!
//! ## AWS
//! Environment variables:
//! - `AWS_ACCESS_KEY_ID`: AWS access key
//! - `AWS_SECRET_ACCESS_KEY`: AWS secret key
//! - `AWS_REGION`: Deployment region (default: us-east-1)
//! - `AWS_SAGEMAKER_EXECUTION_ROLE_ARN`: SageMaker execution role
//!
//! ## GCP
//! Environment variables:
//! - `GOOGLE_CLOUD_PROJECT`: GCP project ID
//! - `GOOGLE_APPLICATION_CREDENTIALS`: Path to service account JSON
//! - `GOOGLE_ACCESS_TOKEN`: Access token for API calls
//!
//! ## Azure
//! Environment variables:
//! - `AZURE_SUBSCRIPTION_ID`: Azure subscription ID
//! - `AZURE_CLIENT_ID`: Service principal client ID
//! - `AZURE_CLIENT_SECRET`: Service principal secret
//! - `AZURE_TENANT_ID`: Azure AD tenant ID
//! - `AZURE_RESOURCE_GROUP`: Resource group name
//! - `AZURE_ML_WORKSPACE`: ML workspace name
//!
//! # Mathematical Specifications
//!
//! ## Auto-Scaling Algorithm
//!
//! The auto-scaling algorithm follows Kubernetes HPA (Horizontal Pod Autoscaler) principles:
//!
//! **Desired replicas** = ceil(current_replicas × (current_metric / target_metric))
//!
//! With constraints:
//! - min_instances ≤ desired_replicas ≤ max_instances
//! - Cooldown period enforced to prevent oscillation
//!
//! ## Metrics Calculation
//!
//! **Saturation** = (GPU_utilization + Memory_utilization) / 2
//!
//! **Health score** = 1.0 - error_rate
//!
//! # Literature References
//!
//! - Barr, J., et al. (2018). "Amazon SageMaker: A fully managed service for machine learning." AWS Blog.
//! - Bisong, E. (2019). "Google Colaboratory." In Building Machine Learning and Deep Learning Models on Google Cloud Platform. Apress.
//! - Lakshmanan, V., et al. (2020). Machine Learning Design Patterns. O'Reilly. ISBN: 978-1098115784
//! - Beyer, B., et al. (2016). Site Reliability Engineering: How Google Runs Production Systems. O'Reilly.
//! - Martin, R. C. (2017). Clean Architecture: A Craftsman's Guide to Software Structure and Design.
//!   Prentice Hall. ISBN: 978-0134494166
//! - Evans, E. (2003). Domain-Driven Design: Tackling Complexity in the Heart of Software.
//!   Addison-Wesley. ISBN: 978-0321125217
//!
//! # Feature Flags
//!
//! - `pinn`: Required for all cloud deployment features
//! - `api`: Required for AWS deployment (enables AWS SDK dependencies)

// Public API exports
pub mod config;
pub mod providers;
pub mod service;
pub mod types;
pub mod utilities;

// Re-export primary types for convenience
pub use config::{AlertThresholds, AutoScalingConfig, DeploymentConfig, MonitoringConfig};
pub use service::CloudPINNService;
pub use types::{
    CloudProvider, DeploymentHandle, DeploymentMetrics, DeploymentStatus, ModelDeploymentData,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_cloud_service_creation() {
        let service = CloudPINNService::new(CloudProvider::AWS).await;
        assert!(service.is_ok());
    }

    #[test]
    fn test_deployment_config_validation() {
        let valid_config = DeploymentConfig {
            provider: CloudProvider::AWS,
            region: "us-east-1".to_string(),
            instance_type: "p3.2xlarge".to_string(),
            gpu_count: 1,
            memory_gb: 16,
            auto_scaling: AutoScalingConfig::default(),
            monitoring: MonitoringConfig::default(),
        };

        assert!(valid_config.validate().is_ok());

        let invalid_config = DeploymentConfig {
            provider: CloudProvider::AWS,
            region: "us-east-1".to_string(),
            instance_type: "p3.2xlarge".to_string(),
            gpu_count: 0, // Invalid
            memory_gb: 16,
            auto_scaling: AutoScalingConfig::default(),
            monitoring: MonitoringConfig::default(),
        };

        assert!(invalid_config.validate().is_err());
    }

    #[test]
    fn test_default_configs() {
        let auto_scaling = AutoScalingConfig::default();
        assert_eq!(auto_scaling.min_instances, 1);
        assert_eq!(auto_scaling.max_instances, 10);
        assert_eq!(auto_scaling.target_gpu_utilization, 0.7);
        assert!(auto_scaling.validate().is_ok());

        let monitoring = MonitoringConfig::default();
        assert!(monitoring.enable_detailed_metrics);
        assert_eq!(monitoring.metrics_interval_seconds, 60);
        assert!(monitoring.validate().is_ok());

        let alerts = AlertThresholds::default();
        assert_eq!(alerts.gpu_utilization_threshold, 0.9);
        assert_eq!(alerts.memory_usage_threshold, 0.9);
        assert_eq!(alerts.error_rate_threshold, 0.05);
        assert!(alerts.validate().is_ok());
    }

    #[test]
    fn test_cloud_provider_display() {
        assert_eq!(CloudProvider::AWS.to_string(), "Amazon Web Services");
        assert_eq!(CloudProvider::GCP.to_string(), "Google Cloud Platform");
        assert_eq!(CloudProvider::Azure.to_string(), "Microsoft Azure");
    }

    #[test]
    fn test_deployment_status_helpers() {
        assert!(DeploymentStatus::Active.is_operational());
        assert!(DeploymentStatus::Active.is_healthy());
        assert!(!DeploymentStatus::Active.is_terminal());

        assert!(!DeploymentStatus::Creating.is_operational());
        assert!(!DeploymentStatus::Creating.is_healthy());

        assert!(DeploymentStatus::Error("test".to_string()).is_terminal());
        assert!(!DeploymentStatus::Error("test".to_string()).is_healthy());
    }

    #[test]
    fn test_deployment_metrics_helpers() {
        let metrics = DeploymentMetrics {
            instance_count: 5,
            avg_gpu_utilization: 0.85,
            avg_memory_utilization: 0.75,
            avg_response_time_ms: 100.0,
            requests_per_second: 50.0,
            error_rate: 0.02,
        };

        assert!(metrics.is_high_load());
        assert!(!metrics.is_unhealthy());
        assert_eq!(metrics.saturation(), 0.8);
    }
}
