use super::types::{AlertThresholds, AutoScalingConfig, CloudMonitoringConfig, DeploymentConfig};
use crate::infrastructure::cloud::types::CloudProvider;

#[test]
fn test_deployment_config_validation_success() {
    let config = DeploymentConfig {
        provider: CloudProvider::AWS,
        region: "us-east-1".to_string(),
        instance_type: "p3.2xlarge".to_string(),
        gpu_count: 1,
        memory_gb: 16,
        auto_scaling: AutoScalingConfig::default(),
        monitoring: CloudMonitoringConfig::default(),
    };

    config.validate().unwrap();
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
        monitoring: CloudMonitoringConfig::default(),
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
        monitoring: CloudMonitoringConfig::default(),
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
        monitoring: CloudMonitoringConfig::default(),
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
    config.validate().unwrap();
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
    let config = CloudMonitoringConfig::default();
    assert!(config.enable_detailed_metrics);
    assert_eq!(config.metrics_interval_seconds, 60);
    config.validate().unwrap();
}

#[test]
fn test_alert_thresholds_default() {
    let thresholds = AlertThresholds::default();
    assert_eq!(thresholds.gpu_utilization_threshold, 0.9);
    assert_eq!(thresholds.memory_usage_threshold, 0.9);
    assert_eq!(thresholds.error_rate_threshold, 0.05);
    thresholds.validate().unwrap();
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
