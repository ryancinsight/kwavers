use kwavers::grid::Grid;
use kwavers::medium::{CoreMedium, HomogeneousMedium};
use kwavers::solver::spectral::config::{
    AbsorptionMode, BoundaryConfig, SpectralConfig as KSpaceConfig,
};
use kwavers::solver::spectral::solver::SpectralSolver as KSpaceSolver;
use kwavers::solver::spectral::sources::{SourceMode, SpectralSource as KSpaceSource};
use ndarray::{Array2, Array3};

#[test]
fn test_plane_wave_propagation_benchmark() {
    // 1. Setup Grid
    let nx = 256;
    let ny = 4;
    let nz = 4;
    let dx = 0.1e-3; // 0.1 mm
    let grid = Grid::new(nx, ny, nz, dx, dx, dx).unwrap();

    // 2. Setup Medium (Water)
    // c0 = 1500 m/s, rho0 = 1000 kg/m^3
    let medium = HomogeneousMedium::water(&grid);
    let c0 = medium.sound_speed(0, 0, 0);

    // 3. Setup Source (Plane Wave Source at x_index = 20)
    let source_x_idx = 20;
    let mut p_mask = Array3::zeros((nx, ny, nz));
    // Create a plane source perpendicular to x-axis
    for k in 0..nz {
        for j in 0..ny {
            p_mask[[source_x_idx, j, k]] = 1.0;
        }
    }

    // Create a Gaussian pulse signal
    let dt = 20e-9; // 20 ns (CFL ~ 0.3)
    let t_end = 10.0e-6; // 10 us
    let steps = (t_end / dt) as usize;

    let mut p_signal = Array2::zeros((1, steps));
    let t_0 = 1.0e-6; // Pulse center at 1 us
    let width = 0.2e-6; // Pulse width

    for n in 0..steps {
        let t = n as f64 * dt;
        let val = (-((t - t_0).powi(2)) / (2.0 * width * width)).exp();
        p_signal[[0, n]] = val;
    }

    let source = KSpaceSource {
        p_mask: Some(p_mask),
        p_signal: Some(p_signal),
        p_mode: SourceMode::Dirichlet, // Use Dirichlet to avoid time-derivative
        ..Default::default()
    };

    // 4. Setup Config
    let config = KSpaceConfig {
        dt,
        absorption_mode: AbsorptionMode::Lossless, // Pure propagation for speed check
        boundary: BoundaryConfig::None,
        ..Default::default()
    };

    // 5. Create Solver
    let mut solver = KSpaceSolver::new(config, grid.clone(), &medium, source).unwrap();

    // 6. Run Simulation
    // We want to check the peak position at the end
    solver.run(steps).unwrap();

    // 7. Analyze Result
    let p_field = solver.pressure_field();

    // Find peak along x-axis (center line)
    let j_center = ny / 2;
    let k_center = nz / 2;

    let mut max_val = 0.0;
    let mut max_idx = 0;

    for i in 0..nx {
        let val = p_field[[i, j_center, k_center]].abs();
        if val > max_val {
            max_val = val;
            max_idx = i;
        }
    }

    let peak_x = max_idx as f64 * dx;

    let propagation_time = t_end - t_0;
    let start_x = source_x_idx as f64 * dx;
    let simulated_speed = (peak_x - start_x) / propagation_time;
    let rel_speed_error = (simulated_speed - c0).abs() / c0;

    println!("Peak position: {:.4} m (index {})", peak_x, max_idx);
    println!("Simulated speed: {:.3} m/s", simulated_speed);
    println!("Expected speed: {:.3} m/s", c0);
    println!("Relative speed error: {:.2}%", 100.0 * rel_speed_error);

    assert!(
        rel_speed_error < 0.10,
        "Wave speed mismatch: expected {}, got {}",
        c0,
        simulated_speed
    );
}
