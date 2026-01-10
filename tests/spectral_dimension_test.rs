use kwavers::domain::grid::Grid;
use kwavers::domain::medium::homogeneous::HomogeneousMedium;
use kwavers::solver::forward::pstd::config::BoundaryConfig;
use kwavers::solver::forward::pstd::{PSTDConfig, PSTDSolver, PSTDSource};
use ndarray::Array3;

#[test]
fn test_spectral_solver_1d_equivalent() {
    // 1D equivalent in 3D: large Y/Z dimensions or just 1 point if supported
    // For now, let's use a 64x1x1 grid if possible, or 64x2x2
    let grid = Grid::new(64, 2, 2, 1e-3, 1e-3, 1e-3).unwrap();
    let medium = HomogeneousMedium::water(&grid);

    let config = PSTDConfig {
        dt: 1e-7,
        nt: 100,
        boundary: BoundaryConfig::None,
        ..Default::default()
    };

    let mut p0 = Array3::zeros((64, 2, 2));
    // Gaussian pulse in X
    for i in 0..64 {
        let x = (i as f64 - 32.0) * 1e-3;
        p0[[i, 0, 0]] = (-x * x / (2.0 * 3e-3_f64.powi(2))).exp() * 1e6;
        p0[[i, 1, 0]] = p0[[i, 0, 0]];
        p0[[i, 0, 1]] = p0[[i, 0, 0]];
        p0[[i, 1, 1]] = p0[[i, 0, 0]];
    }

    let source = PSTDSource {
        p0: Some(p0),
        ..Default::default()
    };

    let mut solver = PSTDSolver::new(config, grid, &medium, source).unwrap();

    // Step forward
    solver.run(10).unwrap();

    let p = &solver.fields.p;
    let max_p = p.iter().fold(0.0f64, |a: f64, &b: &f64| a.max(b.abs()));

    assert!(max_p > 0.0);
    assert!(max_p < 2e6);
}

#[test]
fn test_spectral_solver_2d_equivalent() {
    let grid = Grid::new(32, 32, 2, 1e-3, 1e-3, 1e-3).unwrap();
    let medium = HomogeneousMedium::water(&grid);

    let config = PSTDConfig {
        boundary: BoundaryConfig::None,
        ..Default::default()
    };
    let mut p0 = Array3::zeros((32, 32, 2));
    p0[[16, 16, 0]] = 1e6;
    p0[[16, 16, 1]] = 1e6;

    let source = PSTDSource {
        p0: Some(p0),
        ..Default::default()
    };

    let mut solver = PSTDSolver::new(config, grid, &medium, source).unwrap();
    solver.run(5).unwrap();

    let p = &solver.fields.p;
    let max_p = p.iter().fold(0.0f64, |a: f64, &b: &f64| a.max(b.abs()));
    assert!(max_p > 0.0);
}
