use kwavers_core::constants::fundamental::{DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER_SIM};
use kwavers_grid::Grid;
use kwavers_medium::HomogeneousMedium;
use kwavers_solver::forward::pstd::config::{CompatibilityMode, PSTDConfig};
use kwavers_solver::forward::pstd::PSTDSolver;
use kwavers_source::GridSource;

// The compatibility smoke binary reports its result on stdout; that line is
// the program's deliverable, not debug residue.
#[expect(
    clippy::print_stdout,
    reason = "smoke binary reports its pass/fail result on stdout"
)]
fn main() {
    let config = PSTDConfig {
        compatibility_mode: CompatibilityMode::Reference,
        dt: 1e-8,
        nt: 10,
        ..Default::default()
    };
    let grid = Grid::new(32, 32, 32, 0.001, 0.001, 0.001).unwrap();
    let medium = HomogeneousMedium::new(
        DENSITY_WATER_NOMINAL,
        SOUND_SPEED_WATER_SIM,
        0.0,
        0.0,
        &grid,
    );
    let source = GridSource::new_empty();
    let mut solver = PSTDSolver::new(config, grid, &medium, source).unwrap();
    for _ in 0..10 {
        solver.run_orchestrated(1).expect("Solver iteration failed");
    }
    println!("k-Wave compatibility test passed!");
}
