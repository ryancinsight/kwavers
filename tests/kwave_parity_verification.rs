use kwavers::grid::Grid;
use kwavers::medium::HomogeneousMedium;
use kwavers::solver::kwave_parity::config::KWaveConfig;
use kwavers::solver::kwave_parity::sensors::SensorConfig;
use kwavers::solver::kwave_parity::solver::KWaveSolver;
use kwavers::solver::kwave_parity::sources::KWaveSource;
use ndarray::Array3;

#[test]
fn test_kwave_parity_basic_simulation() {
    // 1. Setup Grid
    let nx = 32;
    let ny = 32;
    let nz = 32;
    let dx = 1e-3;
    let grid = Grid::new(nx, ny, nz, dx, dx, dx).unwrap();

    // 2. Setup Medium (Water)
    let medium = HomogeneousMedium::water(&grid);

    // 3. Setup Source (Initial Pressure)
    let mut p0 = Array3::zeros((nx, ny, nz));
    // Gaussian blob at center
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
                p0[[i, j, k]] = (-dist_sq / (2.0 * sigma * sigma)).exp();
            }
        }
    }

    let source = KWaveSource {
        p0: Some(p0.clone()),
        ..Default::default()
    };

    // 4. Setup Config
    let config = KWaveConfig {
        pml_size: 4,
        pml_alpha: 2.0,
        dt: 1e-7, // Set time step
        ..Default::default()
    };

    // 5. Create Solver
    let mut solver = KWaveSolver::new(config, grid, &medium, source).unwrap();

    // 6. Run a few steps
    for _ in 0..10 {
        solver.step_forward().unwrap();
    }

    // 7. Validate
    let p_field = solver.pressure_field();

    // Check center pressure has decreased (spread out)
    let center_p = p_field[[cx, cy, cz]];
    let initial_center_p = p0[[cx, cy, cz]];

    println!("Initial center p: {}", initial_center_p);
    println!("Final center p: {}", center_p);

    assert!(
        center_p < initial_center_p,
        "Pressure at center should decrease as wave propagates"
    );

    // In very first steps, it might not decrease much, or might oscillate if dt is weird.
    // But with Gaussian initial condition, it should start spreading immediately.

    // Check some neighbor has non-zero pressure
    // The initial condition is wide (sigma=2), so neighbors already have pressure.
    // Let's check a point that was originally near zero.
    // At distance 10, exp(-100/8) is very small ~ 3e-6.
    // Wave speed ~1500 m/s. dt=1e-7. 10 steps = 1e-6 s. Dist = 1.5mm = 1.5 grid cells.
    // So wave hasn't traveled far.

    // Let's just check that physics is happening (values changing).
    assert_ne!(center_p, initial_center_p);

    // Check for NaN/Inf
    for &val in p_field.iter() {
        assert!(val.is_finite(), "Pressure field contains non-finite values");
    }
}

#[test]
fn test_kwave_sensor_records_intensity_and_stats() {
    let nx = 24;
    let ny = 24;
    let nz = 24;
    let dx = 1e-3;
    let grid = Grid::new(nx, ny, nz, dx, dx, dx).unwrap();

    let medium = HomogeneousMedium::water(&grid);

    let mut p0 = Array3::zeros((nx, ny, nz));
    p0[[nx / 2, ny / 2, nz / 2]] = 1.0;

    let source = KWaveSource {
        p0: Some(p0),
        ..Default::default()
    };

    let mut mask = Array3::from_elem((nx, ny, nz), false);
    mask[[nx / 2 + 1, ny / 2, nz / 2]] = true;
    mask[[nx / 2, ny / 2 + 1, nz / 2]] = true;

    let sensor_config = SensorConfig {
        mask,
        record_p: true,
        record_p_max: true,
        record_p_min: true,
        record_p_rms: true,
        record_u: true,
        record_u_max: true,
        record_u_min: true,
        record_u_rms: true,
        record_i: true,
        record_i_avg: true,
        frequency_response: None,
    };

    let config = KWaveConfig {
        dt: 5e-8,
        pml_size: 4,
        pml_alpha: 2.0,
        sensor_config,
        ..Default::default()
    };

    let mut solver = KWaveSolver::new(config, grid, &medium, source).unwrap();
    let data = solver.run(25).unwrap();

    let p = data.p.expect("Expected `p` time series");
    assert_eq!(p.dim().1, 25);
    assert_eq!(p.dim().0, 2);

    let p_max = data.p_max.expect("Expected `p_max`");
    assert!(p_max.iter().all(|v| v.is_finite()));

    let p_min = data.p_min.expect("Expected `p_min`");
    assert!(p_min.iter().all(|v| v.is_finite()));

    let p_rms = data.p_rms.expect("Expected `p_rms`");
    assert!(p_rms.iter().all(|v| v.is_finite()));

    let u = data.u.expect("Expected `u` time series");
    assert_eq!(u.dim().0, 3);
    assert_eq!(u.dim().1, 2);
    assert_eq!(u.dim().2, 25);

    let u_max = data.u_max.expect("Expected `u_max`");
    assert_eq!(u_max.dim().0, 3);

    let i = data.i.expect("Expected `i` time series");
    assert_eq!(i.dim().0, 3);
    assert_eq!(i.dim().1, 2);
    assert_eq!(i.dim().2, 25);
    assert!(i.iter().any(|&v| v != 0.0));

    let i_avg = data.i_avg.expect("Expected `i_avg`");
    assert_eq!(i_avg.dim().0, 3);
    assert!(i_avg.iter().all(|v| v.is_finite()));
}
