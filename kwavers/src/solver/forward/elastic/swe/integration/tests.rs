//! Tests for velocity-Verlet time integration.

use super::integrator::TimeIntegrator;
use crate::domain::grid::Grid;
use crate::solver::forward::elastic::swe::types::ElasticWaveField;
use ndarray::Array3;

#[test]
fn test_time_integrator_creation() {
    let grid = Grid::new(10, 10, 10, 1e-3, 1e-3, 1e-3).unwrap();
    let lambda = Array3::<f64>::from_elem((10, 10, 10), 1e9);
    let mu = Array3::<f64>::from_elem((10, 10, 10), 1e9);
    let density = Array3::<f64>::from_elem((10, 10, 10), 1000.0);
    let pml_sigma = Array3::<f64>::zeros((10, 10, 10));

    let integrator = TimeIntegrator::new(&grid, &lambda, &mu, &density, &pml_sigma);

    let dt = integrator.calculate_stable_timestep(0.5);
    assert!(dt > 0.0);
    assert!(dt < 1e-6);
}

#[test]
fn test_velocity_verlet_step() {
    let grid = Grid::new(10, 10, 10, 1e-3, 1e-3, 1e-3).unwrap();
    let lambda = Array3::<f64>::from_elem((10, 10, 10), 1e9);
    let mu = Array3::<f64>::from_elem((10, 10, 10), 1e9);
    let density = Array3::<f64>::from_elem((10, 10, 10), 1000.0);
    let pml_sigma = Array3::<f64>::zeros((10, 10, 10));

    let integrator = TimeIntegrator::new(&grid, &lambda, &mu, &density, &pml_sigma);
    let mut field = ElasticWaveField::new(10, 10, 10);

    let dt = integrator.calculate_stable_timestep(0.5);
    integrator.step(&mut field, dt, None).unwrap();
}

#[test]
fn test_pml_damping() {
    let grid = Grid::new(10, 10, 10, 1e-3, 1e-3, 1e-3).unwrap();
    let lambda = Array3::<f64>::from_elem((10, 10, 10), 1e9);
    let mu = Array3::<f64>::from_elem((10, 10, 10), 1e9);
    let density = Array3::<f64>::from_elem((10, 10, 10), 1000.0);
    let mut pml_sigma = Array3::<f64>::zeros((10, 10, 10));

    pml_sigma[[0, 5, 5]] = 100.0;

    let integrator = TimeIntegrator::new(&grid, &lambda, &mu, &density, &pml_sigma);
    let mut field = ElasticWaveField::new(10, 10, 10);

    field.vx[[0, 5, 5]] = 1.0;
    let initial_velocity = field.vx[[0, 5, 5]];

    let dt = 1e-7;
    integrator.apply_pml_damping(&mut field, dt);

    assert!(field.vx[[0, 5, 5]] < initial_velocity);
    assert!(field.vx[[0, 5, 5]] > 0.0);
}
