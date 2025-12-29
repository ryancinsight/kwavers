use kwavers::grid::Grid;
use kwavers::medium::HomogeneousMedium;
use kwavers::solver::kwave_parity::config::{AbsorptionMode, KWaveConfig};
use kwavers::solver::kwave_parity::solver::KWaveSolver;
use kwavers::solver::kwave_parity::sources::{KWaveSource, SourceMode};
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
    let c0 = 1500.0; // Expected speed

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

    let source = KWaveSource {
        p_mask: Some(p_mask),
        p_signal: Some(p_signal),
        p_mode: SourceMode::Dirichlet, // Use Dirichlet to avoid time-derivative
        ..Default::default()
    };

    // 4. Setup Config
    let config = KWaveConfig {
        pml_size: 0, // Disable PML to avoid damping in small Y/Z dimensions
        pml_alpha: 2.0,
        dt,
        absorption_mode: AbsorptionMode::Lossless, // Pure propagation for speed check
        ..Default::default()
    };

    // 5. Create Solver
    let mut solver = KWaveSolver::new(config, grid.clone(), &medium, source).unwrap();

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
        let val = p_field[[i, j_center, k_center]];
        if val > max_val {
            max_val = val;
            max_idx = i;
        }
    }

    let peak_x = max_idx as f64 * dx;

    // Expected distance
    // The pulse was centered at t_0 = 1 us.
    // The simulation ran for t_end = 10 us.
    // The pulse propagated for (10 - 1) = 9 us.
    // Distance = c0 * 9e-6.
    // Start position: source_x_idx * dx = 20 * 0.1e-3 = 2.0 mm.

    let propagation_time = t_end - t_0;
    let expected_dist = c0 * propagation_time;
    let start_x = source_x_idx as f64 * dx;

    let expected_x = start_x + expected_dist;

    println!("Peak position: {:.4} m (index {})", peak_x, max_idx);
    println!("Expected position: {:.4} m", expected_x);
    println!("Error: {:.4} m", (peak_x - expected_x).abs());

    // Allow some error due to discrete grid and pulse width
    // 3.0 * dx tolerance (3 grid points) - allows for minor grid alignment/staggering offsets
    assert!(
        (peak_x - expected_x).abs() < 3.0 * dx,
        "Wave peak position mismatch: expected {}, got {}",
        expected_x,
        peak_x
    );
}
