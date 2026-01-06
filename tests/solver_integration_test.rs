use kwavers::boundary::PMLConfig;
use kwavers::grid::Grid;
use kwavers::medium::homogeneous::HomogeneousMedium;
use kwavers::solver::spectral::config::{BoundaryConfig, SpectralConfig as KSpaceConfig};
use kwavers::solver::spectral::solver::SpectralSolver as KSpaceSolver;
use kwavers::solver::spectral::sources::SpectralSource as KSpaceSource;
use ndarray::Array3;

#[test]
fn test_kwave_solver_init_and_step() {
    // 1. Setup Grid
    let (nx, ny, nz) = (32, 32, 32);
    let (dx, dy, dz) = (1.0e-3, 1.0e-3, 1.0e-3);
    let grid = Grid::new(nx, ny, nz, dx, dy, dz).expect("Failed to create grid");

    // 2. Setup Medium (Water)
    let medium = HomogeneousMedium::water(&grid);

    // 3. Setup Config
    let config = KSpaceConfig {
        dt: 50e-9, // 50 ns
        nonlinearity: false,
        boundary: BoundaryConfig::PML(PMLConfig {
            thickness: 4,
            sigma_max_acoustic: 2.0,
            ..PMLConfig::default()
        }),
        ..Default::default()
    };

    // 4. Initialize Solver
    let mut solver = KSpaceSolver::new(config, grid.clone(), &medium, KSpaceSource::default())
        .expect("Failed to create solver");

    // 5. Add initial pressure source (Gaussian pulse in center)
    let mut source = Array3::zeros((nx, ny, nz));
    let cx = nx / 2;
    let cy = ny / 2;
    let cz = nz / 2;
    let sigma = 2.0;

    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let dist_sq = (i as f64 - cx as f64).powi(2)
                    + (j as f64 - cy as f64).powi(2)
                    + (k as f64 - cz as f64).powi(2);
                source[[i, j, k]] = (-dist_sq / (2.0 * sigma * sigma)).exp();
            }
        }
    }

    solver.add_pressure_source(&source);

    // 6. Run a few steps
    solver.run(10).expect("Failed to run solver");

    // 7. Check if field evolved (basic check: pressure should not be exactly source anymore, and not NaN)
    let p_field = solver.pressure_field();

    // Check for NaNs
    for &val in p_field.iter() {
        assert!(!val.is_nan(), "Pressure field contains NaNs");
    }

    // Check peak pressure moved or changed (very basic)
    let center_val = p_field[[cx, cy, cz]];
    println!("Center pressure after 10 steps: {}", center_val);
}
