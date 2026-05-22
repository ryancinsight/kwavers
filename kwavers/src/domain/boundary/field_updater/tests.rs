use super::*;
use crate::core::constants::fundamental::SOUND_SPEED_WATER_SIM;
use crate::domain::boundary::cpml::{CPMLBoundary, CPMLConfig};
use crate::domain::grid::{CartesianTopology, Grid};
use ndarray::Array3;

#[test]
fn test_field_updater_creation() {
    let grid = Grid::new(32, 32, 32, 1e-3, 1e-3, 1e-3).unwrap();
    let config = CPMLConfig::default();
    let boundary = CPMLBoundary::new(config, &grid, SOUND_SPEED_WATER_SIM).unwrap();

    let updater = FieldUpdater::new(boundary);
    assert_eq!(updater.boundary().name(), "CPML (Convolutional PML)");
}

#[test]
fn test_gradient_updater() {
    let topo = CartesianTopology::new([32, 32, 32], [1e-3, 1e-3, 1e-3], [0.0, 0.0, 0.0]).unwrap();
    let mut grad_updater = GradientFieldUpdater::new(&topo);

    // Create test field with linear gradient in x
    let mut field = Array3::zeros((32, 32, 32));
    for i in 0..32 {
        for j in 0..32 {
            for k in 0..32 {
                field[[i, j, k]] = i as f64;
            }
        }
    }

    grad_updater.compute_gradients(&field, &topo);

    // Check gradient is approximately 1/dx in interior
    let expected_grad = 1.0 / 1e-3;
    for i in 1..31 {
        for j in 0..32 {
            for k in 0..32 {
                let computed = grad_updater.grad_x[[i, j, k]];
                assert!((computed - expected_grad).abs() < 1e-6);
            }
        }
    }
}

#[test]
fn test_divergence_computation() {
    let topo = CartesianTopology::new([16, 16, 16], [0.1, 0.1, 0.1], [0.0, 0.0, 0.0]).unwrap();

    // Create uniform expansion field: v = (x, y, z)
    let mut vx = Array3::zeros((16, 16, 16));
    let mut vy = Array3::zeros((16, 16, 16));
    let mut vz = Array3::zeros((16, 16, 16));

    for i in 0..16 {
        for j in 0..16 {
            for k in 0..16 {
                vx[[i, j, k]] = i as f64 * 0.1;
                vy[[i, j, k]] = j as f64 * 0.1;
                vz[[i, j, k]] = k as f64 * 0.1;
            }
        }
    }

    let div = GradientFieldUpdater::compute_divergence(&vx, &vy, &vz, &topo);

    // Divergence should be approximately 3.0 (1.0 from each component)
    for i in 1..15 {
        for j in 1..15 {
            for k in 1..15 {
                assert!((div[[i, j, k]] - 3.0).abs() < 1e-6);
            }
        }
    }
}
