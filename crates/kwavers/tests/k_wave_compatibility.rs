use kwavers_grid::Grid;
use kwavers_medium::HomogeneousMedium;
use kwavers_solver::forward::pstd::config::{CompatibilityMode, PSTDConfig};
use kwavers_solver::forward::pstd::PSTDSolver;
use kwavers_source::GridSource;

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
    let mut solver = PSTDSolver::new(config, grid, &medium, source)
        .expect("a 32^3 water grid at dt = 10 ns is inside the PSTD parameter range");

    // The previous form asserted `result.is_some()` on a value the loop had
    // just wrapped in `Some`, so it held whether or not a step succeeded and
    // discarded ten `KwaversResult`s on the way. Each step's outcome is the
    // claim, and with an empty source there is nothing to record: reference
    // mode must run without a sensor mask and report no sensor data.
    for step in 0..10 {
        let recorded = solver
            .run_orchestrated(1)
            .unwrap_or_else(|error| panic!("reference-mode step {step} must succeed: {error}"));
        assert!(
            recorded.is_none(),
            "step {step} recorded sensor data without a sensor mask: {recorded:?}"
        );
    }
}
