//! Cloud deployment configuration types with validation.

use super::super::types::CloudProvider;
use crate::core::error::{KwaversError, KwaversResult};

/// Cloud deployment configuration
///
/// Specifies all parameters required to deploy a PINN model to a cloud provider
/// with auto-scaling and monitoring capabilities.
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
    /// # Errors
    /// - Returns [`KwaversError::System`] if the precondition for a System-class constraint is violated.
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
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
    /// # Errors
    /// - Returns [`KwaversError::System`] if the precondition for a System-class constraint is violated.
    ///
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
    /// # Errors
    /// - Returns [`KwaversError::System`] if the precondition for a System-class constraint is violated.
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
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
    /// # Errors
    /// - Returns [`KwaversError::System`] if the precondition for a System-class constraint is violated.
    ///
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
    fn default() -> Self {
        Self {
            gpu_utilization_threshold: 0.9,
            memory_usage_threshold: 0.9,
            error_rate_threshold: 0.05,
        }
    }
}
