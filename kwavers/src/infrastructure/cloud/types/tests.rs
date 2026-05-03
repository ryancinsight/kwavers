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
