use super::types::ElectromagneticFdtdSolver;
use crate::domain::grid::Grid;
use crate::physics::electromagnetic::equations::{
    EMDimension, EMMaterialDistribution, ElectromagneticWaveEquation,
};

#[test]
fn test_em_fdtd_creation() {
    let grid = Grid::new(10, 10, 10, 1e-3, 1e-3, 1e-3).unwrap();

    // Use canonical domain composition pattern
    let materials = EMMaterialDistribution::vacuum(&[10, 10, 10]);

    let solver = ElectromagneticFdtdSolver::new(grid, materials, 1e-12, 4).unwrap();
    assert_eq!(solver.em_dimension(), EMDimension::Three);
}

#[test]
fn test_maxwell_time_step() {
    let grid = Grid::new(32, 32, 32, 1e-3, 1e-3, 1e-3).unwrap();

    // Use canonical domain composition pattern
    let materials = EMMaterialDistribution::vacuum(&[32, 32, 32]);

    let solver = ElectromagneticFdtdSolver::new(grid, materials, 1e-12, 4).unwrap();

    // Speed of light in vacuum (normalized units)
    let c = 1.0;
    let dt = solver.max_stable_dt(c);

    // Check that time step is reasonable
    assert!(dt > 0.0);
    assert!(dt < 1e-3); // Should be smaller than spatial step
}
