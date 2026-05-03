use super::*;
use crate::domain::grid::Grid;
use crate::solver::forward::helmholtz::BornConfig;

#[test]
fn test_iterative_born_creation() {
    let config = BornConfig::default();
    let grid = Grid::new(16, 16, 16, 0.1, 0.1, 0.1).unwrap();

    let solver = IterativeBornSolver::new(config, grid);
    assert_eq!(solver.config.max_iterations, 50);
    assert_eq!(solver.grid.nx, 16);
}
