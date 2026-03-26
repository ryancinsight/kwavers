//! Tests for cloud dynamics

use super::*;

#[test]
fn test_cloud_initialization() {
    let config = CloudConfig {
        num_bubbles: 100,
        ..Default::default()
    };

    let mut cloud = CloudDynamics::new(config).unwrap();
    cloud.initialize_cloud().unwrap();

    assert_eq!(cloud.bubbles.len(), 100);
    assert!(cloud.bubbles.iter().all(|b| b.active));
}

#[test]
fn test_cloud_simulation() {
    let config = CloudConfig {
        num_bubbles: 10,
        duration: 1e-4,
        ..Default::default()
    };

    let mut cloud = CloudDynamics::new(config).unwrap();
    cloud.initialize_cloud().unwrap();

    let field = IncidentField::plane_wave(100_000.0, 1e6, [1.0, 0.0, 0.0]);
    cloud.set_incident_field(field);

    let response = cloud.simulate().unwrap();

    assert!(response.time_steps.len() > 1);

    for state in &response.time_steps {
        assert!(!state.bubbles.is_empty());
    }
}

#[test]
fn test_coalescence() {
    let config = CloudConfig {
        num_bubbles: 2,
        coalescence_distance: 1e-6,
        ..Default::default()
    };

    let mut cloud = CloudDynamics::new(config).unwrap();
    cloud.initialize_cloud().unwrap();

    cloud.bubbles[0].position = [0.0, 0.0, 0.0];
    cloud.bubbles[1].position = [0.5e-6, 0.0, 0.0];

    cloud.handle_interactions().unwrap();

    let active_count = cloud.bubbles.iter().filter(|b| b.active).count();
    assert_eq!(active_count, 1);
}
