//! Tests for velocity-Verlet time integration.

use super::integrator::TimeIntegrator;
use crate::domain::grid::Grid;
use crate::solver::forward::elastic::swe::boundary::{ElasticSwePMLBoundary, SwePmlConfig};
use crate::solver::forward::elastic::swe::scratch::ElasticStepScratch;
use crate::solver::forward::elastic::swe::types::ElasticWaveField;
use ndarray::Array3;

fn make_integrator(
    grid: &Grid,
    lambda_val: f64,
    mu_val: f64,
) -> (Array3<f64>, Array3<f64>, Array3<f64>, ElasticSwePMLBoundary) {
    let (nx, ny, nz) = grid.dimensions();
    let lambda = Array3::<f64>::from_elem((nx, ny, nz), lambda_val);
    let mu = Array3::<f64>::from_elem((nx, ny, nz), mu_val);
    let density = Array3::<f64>::from_elem((nx, ny, nz), 1000.0);
    let pml = ElasticSwePMLBoundary::new(grid, SwePmlConfig::default());
    (lambda, mu, density, pml)
}

/// TimeIntegrator constructs without panic and produces a positive stable dt.
#[test]
fn test_time_integrator_creation() {
    let grid = Grid::new(10, 10, 10, 1e-3, 1e-3, 1e-3).unwrap();
    let (lambda, mu, density, pml) = make_integrator(&grid, 1e9, 1e9);
    let integrator = TimeIntegrator::new(&grid, &lambda, &mu, &density, &pml);
    let dt = integrator.calculate_stable_timestep(0.5);
    assert!(dt > 0.0, "stable dt must be positive, got {dt}");
    assert!(
        dt < 1e-6,
        "stable dt must be sub-microsecond for Δx=1 mm at E=1 GPa, got {dt}"
    );
}

/// One velocity-Verlet step completes without error on a zero-initial field.
///
/// ## Theorem (zero-field fixed point)
///
/// If u = v = 0 and f = 0, then ∇·σ = 0 everywhere (zero strain → zero
/// stress → zero divergence).  Therefore a = 0 at all sub-steps and the
/// field remains identically zero after any number of steps.  The assertion
/// below validates the fixed-point invariant.
#[test]
fn test_velocity_verlet_zero_field_fixed_point() {
    let grid = Grid::new(10, 10, 10, 1e-3, 1e-3, 1e-3).unwrap();
    let (lambda, mu, density, pml) = make_integrator(&grid, 1e9, 1e9);
    let integrator = TimeIntegrator::new(&grid, &lambda, &mu, &density, &pml);
    let mut field = ElasticWaveField::new(10, 10, 10);
    let dt = integrator.calculate_stable_timestep(0.5);
    let mut scratch = ElasticStepScratch::new(10, 10, 10);
    integrator.step(&mut field, dt, None, &mut scratch).unwrap();

    // Zero field, no body force → acceleration = 0 → field stays zero.
    assert_eq!(field.ux[[5, 5, 5]], 0.0, "ux must stay zero");
    assert_eq!(field.vx[[5, 5, 5]], 0.0, "vx must stay zero");
}

/// PML damping attenuates velocity in the absorbing layer and leaves the
/// interior unchanged.
///
/// ## Theorem (separable exponential PML)
///
/// For cell `(0, 16, 16)`, σ_x[0] > 0 and σ_y[16] = σ_z[16] = 0 (interior
/// y and z), so `d = exp(−σ_x[0]·dt) < 1`.  For cell `(16, 16, 16)`, all
/// three σ values are zero (interior), so `d = 1` and the velocity is
/// unchanged.  The test asserts both predictions.
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
        "vx in PML must be attenuated, got {} ≥ {initial_velocity}",
        field.vx[[0, 16, 16]]
    );
    assert!(
        field.vx[[0, 16, 16]] > 0.0,
        "damping must be positive, got {}",
        field.vx[[0, 16, 16]]
    );

    // Interior cell must be unmodified (d = exp(0)*exp(0)*exp(0) = 1.0).
    field.vx[[16, 16, 16]] = 1.0;
    integrator.apply_pml_damping(&mut field, dt);
    assert!(
        (field.vx[[16, 16, 16]] - 1.0).abs() < 1e-12,
        "interior cell must be unmodified, got {}",
        field.vx[[16, 16, 16]]
    );
}
