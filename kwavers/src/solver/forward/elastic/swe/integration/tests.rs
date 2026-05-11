//! Tests for velocity-Verlet time integration.

use super::integrator::TimeIntegrator;
use crate::domain::grid::Grid;
use crate::solver::forward::elastic::swe::boundary::{PMLBoundary, PMLConfig};
use crate::solver::forward::elastic::swe::types::ElasticWaveField;
use ndarray::Array3;

fn make_integrator(grid: &Grid, lambda_val: f64, mu_val: f64) -> (Array3<f64>, Array3<f64>, Array3<f64>, PMLBoundary) {
    let (nx, ny, nz) = grid.dimensions();
    let lambda = Array3::<f64>::from_elem((nx, ny, nz), lambda_val);
    let mu = Array3::<f64>::from_elem((nx, ny, nz), mu_val);
    let density = Array3::<f64>::from_elem((nx, ny, nz), 1000.0);
    let pml = PMLBoundary::new(grid, PMLConfig::default());
    (lambda, mu, density, pml)
}

#[test]
fn test_time_integrator_creation() {
    let grid = Grid::new(10, 10, 10, 1e-3, 1e-3, 1e-3).unwrap();
    let (lambda, mu, density, pml) = make_integrator(&grid, 1e9, 1e9);
    let integrator = TimeIntegrator::new(&grid, &lambda, &mu, &density, &pml);
    let dt = integrator.calculate_stable_timestep(0.5);
    assert!(dt > 0.0);
    assert!(dt < 1e-6);
}

#[test]
fn test_velocity_verlet_step() {
    let grid = Grid::new(10, 10, 10, 1e-3, 1e-3, 1e-3).unwrap();
    let (lambda, mu, density, pml) = make_integrator(&grid, 1e9, 1e9);
    let integrator = TimeIntegrator::new(&grid, &lambda, &mu, &density, &pml);
    let mut field = ElasticWaveField::new(10, 10, 10);
    let dt = integrator.calculate_stable_timestep(0.5);
    integrator.step(&mut field, dt, None).unwrap();
}

#[test]
fn test_pml_damping() {
    // Use a grid with PML thickness = 2 (default is 10, but grid is 10 cells —
    // use a 32-cell grid so the PML is non-degenerate).
    let grid = Grid::new(32, 32, 32, 1e-3, 1e-3, 1e-3).unwrap();
    let (lambda, mu, density, pml) = make_integrator(&grid, 1e9, 1e9);

    let integrator = TimeIntegrator::new(&grid, &lambda, &mu, &density, &pml);
    let mut field = ElasticWaveField::new(32, 32, 32);

    // Set vx at the boundary corner (index 0 is inside the PML layer).
    field.vx[[0, 16, 16]] = 1.0;
    let initial_velocity = field.vx[[0, 16, 16]];

    let dt = 1e-7;
    integrator.apply_pml_damping(&mut field, dt);

    // Cell (0,16,16) is in the x-PML: sigma_x[0] > 0, sigma_y[16] = 0,
    // sigma_z[16] = 0 → damping factor = exp(-sigma_x[0]*dt) < 1.
    assert!(
        field.vx[[0, 16, 16]] < initial_velocity,
        "vx in PML must be attenuated"
    );
    assert!(field.vx[[0, 16, 16]] > 0.0, "damping must be positive");

    // Interior cell must be unmodified (d = exp(0)*exp(0)*exp(0) = 1.0).
    field.vx[[16, 16, 16]] = 1.0;
    integrator.apply_pml_damping(&mut field, dt);
    assert!(
        (field.vx[[16, 16, 16]] - 1.0).abs() < 1e-12,
        "interior cell must be unmodified"
    );
}
