//! Tests for infrastructure API types and configuration.

use super::api_types::{
    APIConfig, GeometrySpec, HealthStatus, PINNTrainingRequest, PaginationParams,
    PinnApiPhysicsParameters, PinnApiTrainingConfig, ServiceStatus,
};
use std::collections::HashMap;

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
        physics_params: PinnApiPhysicsParameters {
            material_properties: HashMap::new(),
            boundary_values: HashMap::new(),
            initial_values: HashMap::new(),
            domain_params: HashMap::new(),
        },
        training_config: PinnApiTrainingConfig {
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
