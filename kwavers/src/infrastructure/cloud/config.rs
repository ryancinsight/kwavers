//! Cloud Deployment Configuration
//!
//! This module provides configuration types for cloud PINN deployments with validation,
//! defaults, and builder patterns following the Builder pattern and Clean Architecture principles.
//!
//! # Architecture
//!
//! Configuration types represent domain value objects with:
//! - Immutable fields after construction
//! - Built-in validation
//! - Sensible defaults
//! - Clear error messages
//!
//! # Literature References
//!
//! - Evans, E. (2003). Domain-Driven Design: Tackling Complexity in the Heart of Software.
//!   Addison-Wesley. ISBN: 978-0321125217
//! - Martin, R. C. (2017). Clean Architecture: A Craftsman's Guide to Software Structure and Design.
//!   Prentice Hall. ISBN: 978-0134494166

use crate::core::error::{KwaversError, KwaversResult};

use super::types::CloudProvider;

/// Cloud deployment configuration
///
/// Specifies all parameters required to deploy a PINN model to a cloud provider
/// with auto-scaling and monitoring capabilities.
///
/// # Example
///
/// ```
/// use kwavers::infra::cloud::{DeploymentConfig, CloudProvider, AutoScalingConfig, MonitoringConfig};
///
/// let config = DeploymentConfig {
///     provider: CloudProvider::AWS,
///     region: "us-east-1".to_string(),
///     instance_type: "p3.2xlarge".to_string(),
///     gpu_count: 1,
///     memory_gb: 16,
///     auto_scaling: AutoScalingConfig::default(),
///     monitoring: MonitoringConfig::default(),
/// };
/// ```
#[derive(Debug, Clone)]
pub struct DeploymentConfig {
    /// Cloud provider
    pub provider: CloudProvider,
    /// Region for deployment (e.g., "us-east-1", "europe-west1")
    pub region: String,
    /// Instance type/family (e.g., "p3.2xlarge", "n1-standard-8")
    pub instance_type: String,
    /// Number of GPU instances
    pub gpu_count: usize,
    /// Memory allocation in gigabytes
    pub memory_gb: usize,
    /// Auto-scaling configuration
    pub auto_scaling: AutoScalingConfig,
    /// Monitoring configuration
    pub monitoring: MonitoringConfig,
}

impl DeploymentConfig {
    /// Validate deployment configuration
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - GPU count is zero
    /// - Memory allocation is zero
    /// - Region is empty
    /// - Instance type is empty
    /// - Auto-scaling configuration is invalid
    pub fn validate(&self) -> KwaversResult<()> {
        if self.gpu_count == 0 {
            return Err(KwaversError::System(
                crate::core::error::SystemError::InvalidConfiguration {
                    parameter: "gpu_count".to_string(),
                    reason: "Must specify at least 1 GPU".to_string(),
                },
            ));
        }

        if self.memory_gb == 0 {
            return Err(KwaversError::System(
                crate::core::error::SystemError::InvalidConfiguration {
                    parameter: "memory_gb".to_string(),
                    reason: "Must specify memory allocation".to_string(),
                },
            ));
        }

        if self.region.is_empty() {
            return Err(KwaversError::System(
                crate::core::error::SystemError::InvalidConfiguration {
                    parameter: "region".to_string(),
                    reason: "Region cannot be empty".to_string(),
                },
            ));
        }

        if self.instance_type.is_empty() {
            return Err(KwaversError::System(
                crate::core::error::SystemError::InvalidConfiguration {
                    parameter: "instance_type".to_string(),
                    reason: "Instance type cannot be empty".to_string(),
                },
            ));
        }

        self.auto_scaling.validate()?;
        self.monitoring.validate()?;

        Ok(())
    }
}

/// Auto-scaling configuration for cloud deployments
///
/// Controls dynamic scaling behavior based on resource utilization metrics.
///
/// # Invariants
///
/// - `min_instances` ≤ `max_instances`
/// - `0.0 < target_gpu_utilization < 1.0`
/// - `scale_down_threshold < scale_up_threshold`
/// - `cooldown_seconds > 0`
///
/// # Literature
///
/// Auto-scaling strategies based on:
/// - Kubernetes Horizontal Pod Autoscaler algorithm
/// - AWS Auto Scaling target tracking policies
#[derive(Debug, Clone)]
pub struct AutoScalingConfig {
    /// Minimum number of instances (must be ≥ 1)
    pub min_instances: usize,
    /// Maximum number of instances (must be ≥ min_instances)
    pub max_instances: usize,
    /// Target GPU utilization (0.0-1.0)
    pub target_gpu_utilization: f64,
    /// Scale-up threshold (triggers adding instances)
    pub scale_up_threshold: f64,
    /// Scale-down threshold (triggers removing instances)
    pub scale_down_threshold: f64,
    /// Cooldown period in seconds (prevents flapping)
    pub cooldown_seconds: u64,
}

impl AutoScalingConfig {
    /// Validate auto-scaling configuration
    ///
    /// # Errors
    ///
    /// Returns error if invariants are violated
    pub fn validate(&self) -> KwaversResult<()> {
        if self.min_instances == 0 {
            return Err(KwaversError::System(
                crate::core::error::SystemError::InvalidConfiguration {
                    parameter: "min_instances".to_string(),
                    reason: "Must have at least 1 instance".to_string(),
                },
            ));
        }

        if self.max_instances < self.min_instances {
            return Err(KwaversError::System(
                crate::core::error::SystemError::InvalidConfiguration {
                    parameter: "max_instances".to_string(),
                    reason: format!(
                        "max_instances ({}) must be >= min_instances ({})",
                        self.max_instances, self.min_instances
                    ),
                },
            ));
        }

        if self.target_gpu_utilization <= 0.0 || self.target_gpu_utilization >= 1.0 {
            return Err(KwaversError::System(
                crate::core::error::SystemError::InvalidConfiguration {
                    parameter: "target_gpu_utilization".to_string(),
                    reason: "Must be between 0.0 and 1.0".to_string(),
                },
            ));
        }

        if self.scale_down_threshold >= self.scale_up_threshold {
            return Err(KwaversError::System(
                crate::core::error::SystemError::InvalidConfiguration {
                    parameter: "scale_down_threshold".to_string(),
                    reason: "scale_down_threshold must be < scale_up_threshold".to_string(),
                },
            ));
        }

        if self.cooldown_seconds == 0 {
            return Err(KwaversError::System(
                crate::core::error::SystemError::InvalidConfiguration {
                    parameter: "cooldown_seconds".to_string(),
                    reason: "Cooldown period must be > 0".to_string(),
                },
            ));
        }

        Ok(())
    }
}

impl Default for AutoScalingConfig {
    /// Default auto-scaling configuration
    ///
    /// - 1-10 instances
    /// - Target 70% GPU utilization
    /// - Scale up at 80%, down at 30%
    /// - 5 minute cooldown
    fn default() -> Self {
        Self {
            min_instances: 1,
            max_instances: 10,
            target_gpu_utilization: 0.7,
            scale_up_threshold: 0.8,
            scale_down_threshold: 0.3,
            cooldown_seconds: 300,
        }
    }
}

/// Monitoring configuration for cloud deployments
///
/// Controls metrics collection and alerting behavior.
///
/// # Literature
///
/// Monitoring best practices from:
/// - Beyer, B., et al. (2016). Site Reliability Engineering: How Google Runs Production Systems. O'Reilly.
/// - Prometheus monitoring and alerting framework documentation
#[derive(Debug, Clone)]
pub struct MonitoringConfig {
    /// Enable detailed metrics collection
    pub enable_detailed_metrics: bool,
    /// Metrics collection interval in seconds
    pub metrics_interval_seconds: u64,
    /// Alert thresholds
    pub alert_thresholds: AlertThresholds,
}

impl MonitoringConfig {
    /// Validate monitoring configuration
    ///
    /// # Errors
    ///
    /// Returns error if metrics interval is zero or thresholds are invalid
    pub fn validate(&self) -> KwaversResult<()> {
        if self.metrics_interval_seconds == 0 {
            return Err(KwaversError::System(
                crate::core::error::SystemError::InvalidConfiguration {
                    parameter: "metrics_interval_seconds".to_string(),
                    reason: "Metrics interval must be > 0".to_string(),
                },
            ));
        }

        self.alert_thresholds.validate()?;

        Ok(())
    }
}

impl Default for MonitoringConfig {
    /// Default monitoring configuration
    ///
    /// - Detailed metrics enabled
    /// - 60 second collection interval
    /// - Default alert thresholds
    fn default() -> Self {
        Self {
            enable_detailed_metrics: true,
            metrics_interval_seconds: 60,
            alert_thresholds: AlertThresholds::default(),
        }
    }
}

/// Alert thresholds for monitoring
///
/// Defines when to trigger alerts based on resource utilization and error rates.
///
/// # Invariants
///
/// All thresholds must be in range [0.0, 1.0]
#[derive(Debug, Clone)]
pub struct AlertThresholds {
    /// GPU utilization alert threshold (0.0-1.0)
    pub gpu_utilization_threshold: f64,
    /// Memory usage alert threshold (0.0-1.0)
    pub memory_usage_threshold: f64,
    /// Error rate alert threshold (0.0-1.0)
    pub error_rate_threshold: f64,
}

impl AlertThresholds {
    /// Validate alert thresholds
    ///
    /// # Errors
    ///
    /// Returns error if any threshold is outside [0.0, 1.0]
    pub fn validate(&self) -> KwaversResult<()> {
        let thresholds = [
            ("gpu_utilization_threshold", self.gpu_utilization_threshold),
            ("memory_usage_threshold", self.memory_usage_threshold),
            ("error_rate_threshold", self.error_rate_threshold),
        ];

        for (name, value) in &thresholds {
            if *value < 0.0 || *value > 1.0 {
                return Err(KwaversError::System(
                    crate::core::error::SystemError::InvalidConfiguration {
                        parameter: name.to_string(),
                        reason: "Threshold must be between 0.0 and 1.0".to_string(),
                    },
                ));
            }
        }

        Ok(())
    }
}

impl Default for AlertThresholds {
    /// Default alert thresholds
    ///
    /// - GPU utilization: 90%
    /// - Memory usage: 90%
    /// - Error rate: 5%
    fn default() -> Self {
        Self {
            gpu_utilization_threshold: 0.9,
            memory_usage_threshold: 0.9,
            error_rate_threshold: 0.05,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deployment_config_validation_success() {
        let config = DeploymentConfig {
            provider: CloudProvider::AWS,
            region: "us-east-1".to_string(),
            instance_type: "p3.2xlarge".to_string(),
            gpu_count: 1,
            memory_gb: 16,
            auto_scaling: AutoScalingConfig::default(),
            monitoring: MonitoringConfig::default(),
        };

        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_deployment_config_validation_zero_gpu() {
        let config = DeploymentConfig {
            provider: CloudProvider::AWS,
            region: "us-east-1".to_string(),
            instance_type: "p3.2xlarge".to_string(),
            gpu_count: 0,
            memory_gb: 16,
            auto_scaling: AutoScalingConfig::default(),
            monitoring: MonitoringConfig::default(),
        };

        assert!(config.validate().is_err());
    }

    #[test]
    fn test_deployment_config_validation_zero_memory() {
        let config = DeploymentConfig {
            provider: CloudProvider::AWS,
            region: "us-east-1".to_string(),
            instance_type: "p3.2xlarge".to_string(),
            gpu_count: 1,
            memory_gb: 0,
            auto_scaling: AutoScalingConfig::default(),
            monitoring: MonitoringConfig::default(),
        };

        assert!(config.validate().is_err());
    }

    #[test]
    fn test_deployment_config_validation_empty_region() {
        let config = DeploymentConfig {
            provider: CloudProvider::AWS,
            region: String::new(),
            instance_type: "p3.2xlarge".to_string(),
            gpu_count: 1,
            memory_gb: 16,
            auto_scaling: AutoScalingConfig::default(),
            monitoring: MonitoringConfig::default(),
        };

        assert!(config.validate().is_err());
    }

    #[test]
    fn test_auto_scaling_config_default() {
        let config = AutoScalingConfig::default();
        assert_eq!(config.min_instances, 1);
        assert_eq!(config.max_instances, 10);
        assert_eq!(config.target_gpu_utilization, 0.7);
        assert_eq!(config.scale_up_threshold, 0.8);
        assert_eq!(config.scale_down_threshold, 0.3);
        assert_eq!(config.cooldown_seconds, 300);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_auto_scaling_validation_min_max() {
        let config = AutoScalingConfig {
            min_instances: 5,
            max_instances: 3,
            target_gpu_utilization: 0.7,
            scale_up_threshold: 0.8,
            scale_down_threshold: 0.3,
            cooldown_seconds: 300,
        };

        assert!(config.validate().is_err());
    }

    #[test]
    fn test_auto_scaling_validation_threshold_order() {
        let config = AutoScalingConfig {
            min_instances: 1,
            max_instances: 10,
            target_gpu_utilization: 0.7,
            scale_up_threshold: 0.3,
            scale_down_threshold: 0.8,
            cooldown_seconds: 300,
        };

        assert!(config.validate().is_err());
    }

    #[test]
    fn test_monitoring_config_default() {
        let config = MonitoringConfig::default();
        assert!(config.enable_detailed_metrics);
        assert_eq!(config.metrics_interval_seconds, 60);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_alert_thresholds_default() {
        let thresholds = AlertThresholds::default();
        assert_eq!(thresholds.gpu_utilization_threshold, 0.9);
        assert_eq!(thresholds.memory_usage_threshold, 0.9);
        assert_eq!(thresholds.error_rate_threshold, 0.05);
        assert!(thresholds.validate().is_ok());
    }

    #[test]
    fn test_alert_thresholds_validation_out_of_range() {
        let thresholds = AlertThresholds {
            gpu_utilization_threshold: 1.5,
            memory_usage_threshold: 0.9,
            error_rate_threshold: 0.05,
        };

        assert!(thresholds.validate().is_err());
    }
}
