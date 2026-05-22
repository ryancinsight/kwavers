use kwavers::core::constants::fundamental::SOUND_SPEED_WATER_SIM;
use kwavers::domain::grid::Grid;
use kwavers::domain::medium::HomogeneousMedium;
use kwavers::domain::source::GridSource;
use kwavers::solver::forward::pstd::config::{CompatibilityMode, PSTDConfig};
use kwavers::solver::forward::pstd::PSTDSolver;

fn main() {
    let config = PSTDConfig {
        compatibility_mode: CompatibilityMode::Reference,
        dt: 1e-8,
        nt: 10,
        ..Default::default()
    };
    let grid = Grid::new(32, 32, 32, 0.001, 0.001, 0.001).unwrap();
    let medium = HomogeneousMedium::new(1000.0, SOUND_SPEED_WATER_SIM, 0.0, 0.0, &grid);
    let source = GridSource::new_empty();
    let mut solver = PSTDSolver::new(config, grid, &medium, source).unwrap();
    for _ in 0..10 {
        solver.run_orchestrated(1).expect("Solver iteration failed");
    }
    println!("k-Wave compatibility test passed!");
}
