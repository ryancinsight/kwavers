use super::solver::UniversalPINNSolver;
use super::types::{GeometricFeature, Geometry2D, UniversalSolverStats, UniversalTrainingConfig};
use std::time::Duration;

#[test]
fn test_universal_solver_creation() {
    let solver = UniversalPINNSolver::<burn::backend::Autodiff<burn::backend::NdArray<f32>>>::new();
    let solver = solver.unwrap();
    assert!(solver.available_domains().is_empty());
}

#[test]
fn test_geometry_creation() {
    let geometry = Geometry2D::rectangle(0.0, 1.0, 0.0, 1.0);
    assert_eq!(geometry.bounds, [0.0, 1.0, 0.0, 1.0]);
    assert!(geometry.features.is_empty());
}

#[test]
fn test_geometry_with_obstacle() {
    let geometry = Geometry2D::rectangle(0.0, 2.0, 0.0, 1.0).with_circle_obstacle((0.5, 0.5), 0.1);
    assert_eq!(geometry.features.len(), 1);
    match &geometry.features[0] {
        GeometricFeature::Circle { center, radius } => {
            assert_eq!(*center, (0.5, 0.5));
            assert_eq!(*radius, 0.1);
        }
        _ => panic!("Expected Circle feature"),
    }
}

#[test]
fn test_point_in_geometry() {
    let solver =
        UniversalPINNSolver::<burn::backend::Autodiff<burn::backend::NdArray<f32>>>::new().unwrap();
    let geometry = Geometry2D::rectangle(0.0, 1.0, 0.0, 1.0).with_circle_obstacle((0.5, 0.5), 0.2);

    assert!(solver.is_point_in_geometry(0.8, 0.8, &geometry));
    assert!(!solver.is_point_in_geometry(1.5, 0.5, &geometry));
    assert!(!solver.is_point_in_geometry(0.5, 0.5, &geometry));
}

#[test]
fn test_training_config_defaults() {
    let config = UniversalTrainingConfig::default();
    assert_eq!(config.epochs, 1000);
    assert_eq!(config.learning_rate, 0.001);
    assert_eq!(config.collocation_points, 1000);
    assert_eq!(config.boundary_points, 200);
    assert!(config.adaptive_sampling);
    assert!(config.early_stopping.as_ref().unwrap().patience > 0);
}

#[test]
fn test_solver_stats_defaults() {
    let stats = UniversalSolverStats::default();
    assert_eq!(stats.training_time, Duration::default());
    assert!(stats.final_losses.is_empty());
    assert!(stats.loss_history.is_empty());
    assert!(!stats.convergence_info.converged);
    assert_eq!(stats.convergence_info.final_epoch, 0);
}
