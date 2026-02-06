//! Tests for Kuznetsov equation solver

#[cfg(test)]
use super::*;
#[cfg(test)]
use crate::domain::grid::Grid;

#[test]
fn test_kuznetsov_creation() {
    // Test that we can create a Kuznetsov solver with default configuration
    let grid = Grid::new(32, 32, 32, 0.001, 0.001, 0.001).unwrap();
    let config = config::KuznetsovConfig::default();
    let solver = solver::KuznetsovWave::new(config, &grid);

    assert!(
        solver.is_ok(),
        "Should create KuznetsovWave solver successfully"
    );
}

#[test]
fn test_kuznetsov_config_validation() {
    // Test configuration validation
    let grid = Grid::new(32, 32, 32, 0.001, 0.001, 0.001).unwrap();

    // Test valid config
    let valid_config = config::KuznetsovConfig::default();
    assert!(valid_config.validate(&grid).is_ok());

    // Test invalid CFL factor
    let invalid_config = config::KuznetsovConfig {
        cfl_factor: 2.0, // Too high
        ..Default::default()
    };
    assert!(invalid_config.validate(&grid).is_err());
}

#[test]
fn test_numerical_methods() {
    // Test that numerical methods produce correct dimensions
    let grid = Grid::new(16, 16, 16, 0.001, 0.001, 0.001).unwrap();
    let field = ndarray::Array3::from_elem((16, 16, 16), 1.0);

    // Test Laplacian
    let laplacian = numerical::compute_laplacian(&field, &grid);
    assert_eq!(laplacian.dim(), field.dim());

    // Test gradient
    let (grad_x, grad_y, grad_z) = numerical::compute_gradient(&field, &grid);
    assert_eq!(grad_x.dim(), field.dim());
    assert_eq!(grad_y.dim(), field.dim());
    assert_eq!(grad_z.dim(), field.dim());
}
