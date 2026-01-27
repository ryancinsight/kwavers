use kwavers::domain::grid::Grid;
use kwavers::domain::medium::HomogeneousMedium;
use kwavers::domain::source::GridSource;
use kwavers::solver::forward::pstd::config::{CompatibilityMode, PSTDConfig};
use kwavers::solver::forward::pstd::PSTDSolver;

#[test]
fn test_kwave_compatibility_mode() {
    let config = PSTDConfig {
        compatibility_mode: CompatibilityMode::Reference,
        dt: 1e-8,
        nt: 10,
        ..Default::default()
    };
    let grid = Grid::new(32, 32, 32, 0.001, 0.001, 0.001).unwrap();
    let medium = HomogeneousMedium::new(1000.0, 1500.0, 0.0, 0.0, &grid);
    let source = GridSource::new_empty();
    let mut solver = PSTDSolver::new(config, grid, &medium, source).unwrap();
    let mut result = None;
    for _ in 0..10 {
        result = Some(solver.run_orchestrated(1));
    }
    assert!(result.is_some(), "Solver did not run");
}
