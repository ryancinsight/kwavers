use super::super::types::CloudProvider;
use super::orchestrator::CloudPINNService;

#[tokio::test]
async fn test_cloud_service_creation() {
    let service = CloudPINNService::new(CloudProvider::AWS).await;
    assert!(service.is_ok());
}

#[tokio::test]
async fn test_cloud_service_provider_types() {
    let aws = CloudPINNService::new(CloudProvider::AWS).await.unwrap();
    assert_eq!(aws.provider, CloudProvider::AWS);

    let gcp = CloudPINNService::new(CloudProvider::GCP).await.unwrap();
    assert_eq!(gcp.provider, CloudProvider::GCP);

    let azure = CloudPINNService::new(CloudProvider::Azure).await.unwrap();
    assert_eq!(azure.provider, CloudProvider::Azure);
}

#[tokio::test]
async fn test_deployment_count_initially_zero() {
    let service = CloudPINNService::new(CloudProvider::AWS).await.unwrap();
    assert_eq!(service.deployment_count(), 0);
}

#[tokio::test]
async fn test_get_nonexistent_deployment_status() {
    let service = CloudPINNService::new(CloudProvider::AWS).await.unwrap();
    let result = service.get_deployment_status("nonexistent").await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_scale_nonexistent_deployment() {
    let mut service = CloudPINNService::new(CloudProvider::AWS).await.unwrap();
    let result = service.scale_deployment("nonexistent", 5).await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_scale_zero_instances() {
    let mut service = CloudPINNService::new(CloudProvider::AWS).await.unwrap();
    let result = service.scale_deployment("any-id", 0).await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_terminate_nonexistent_deployment() {
    let mut service = CloudPINNService::new(CloudProvider::AWS).await.unwrap();
    let result = service.terminate_deployment("nonexistent").await;
    assert!(result.is_err());
}
