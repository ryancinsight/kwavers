use super::super::config::NonlinearSWEConfig;
use super::super::material::HyperelasticModel;
use super::NonlinearElasticWaveSolver;
use crate::domain::grid::Grid;
use crate::domain::medium::HomogeneousMedium;
use ndarray::Array3;

#[test]
fn test_nonlinear_solver_creation() {
    let grid = Grid::new(16, 16, 16, 0.001, 0.001, 0.001).unwrap();
    let medium = HomogeneousMedium::new(1000.0, 1500.0, 0.5, 1.0, &grid);
    let material = HyperelasticModel::neo_hookean_soft_tissue();
    let config = NonlinearSWEConfig::default();

    let _solver = NonlinearElasticWaveSolver::new(&grid, &medium, material, config).unwrap();
}

#[test]
fn test_time_step_calculation() {
    let grid = Grid::new(16, 16, 16, 0.001, 0.001, 0.001).unwrap();
    let medium = HomogeneousMedium::soft_tissue(1000.0, 0.49, &grid);
    let material = HyperelasticModel::neo_hookean_soft_tissue();
    let config = NonlinearSWEConfig::default();

    let solver = NonlinearElasticWaveSolver::new(&grid, &medium, material, config).unwrap();
    let dt = solver.calculate_time_step();

    assert!(dt > 0.0, "Time step should be positive");
    assert!(dt < 1e-6, "Time step should be small for stability");
}

#[test]
fn test_wave_propagation() {
    let grid = Grid::new(32, 16, 16, 0.001, 0.001, 0.001).unwrap();
    let medium = HomogeneousMedium::soft_tissue(1000.0, 0.49, &grid);
    let material = HyperelasticModel::neo_hookean_soft_tissue();
    let config = NonlinearSWEConfig {
        nonlinearity_parameter: 0.05,
        enable_harmonics: false,
        max_dt: 1e-7,
        ..Default::default()
    };

    let solver = NonlinearElasticWaveSolver::new(&grid, &medium, material, config).unwrap();

    let mut initial = Array3::zeros((32, 16, 16));
    initial[[16, 8, 8]] = 1e-6;

    let history = solver.propagate_waves(&initial).unwrap();
    assert!(!history.is_empty());
    assert!(history.len() >= 2);
}
