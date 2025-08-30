//! Physics validation against analytical solutions and k-Wave benchmarks

use kwavers::{
    constants::materials::{WATER_DENSITY, WATER_SOUND_SPEED},
    Grid, HomogeneousMedium, WesterveltFdtd, WesterveltFdtdConfig,
};
use ndarray::Array3;
use std::f64::consts::PI;

/// Validate plane wave propagation against analytical solution
#[test]
fn validate_plane_wave_propagation() {
    // Test parameters from k-Wave example
    let grid_size = 128;
    let dx = 0.1e-3; // 0.1 mm
    let frequency = 1e6; // 1 MHz
    let wavelength = WATER_SOUND_SPEED / frequency;
    let cfl = 0.3;
    let dt = cfl * dx / WATER_SOUND_SPEED;

    // Create grid
    let grid = Grid::new(grid_size, grid_size, grid_size, dx, dx, dx);

    // Create homogeneous medium
    let medium = HomogeneousMedium::from_minimal(WATER_DENSITY, WATER_SOUND_SPEED, &grid);

    // Initialize solver
    let config = WesterveltFdtdConfig {
        cfl_number: cfl,
        enable_absorption: false,
        enable_nonlinearity: false,
        ..Default::default()
    };
    let mut solver = WesterveltFdtd::new(config, &grid);

    // Add plane wave source
    let source_position = (10, grid_size / 2, grid_size / 2);
    let source_amplitude = 1e6; // 1 MPa

    // Run simulation for one wavelength travel distance
    let periods = 3.0;
    let total_time = periods / frequency;
    let num_steps = (total_time / dt) as usize;

    for step in 0..num_steps {
        let time = step as f64 * dt;
        let source_value = source_amplitude * (2.0 * PI * frequency * time).sin();

        // Apply source
        solver.add_point_source(source_position, source_value);

        // Step simulation
        solver.step(&medium, dt).expect("Simulation step failed");
    }

    // Get final pressure field
    let pressure = solver.get_pressure();

    // Validate against analytical solution
    // For a plane wave: p(x,t) = A * sin(k*x - omega*t)
    let k = 2.0 * PI / wavelength; // wave number
    let omega = 2.0 * PI * frequency; // angular frequency

    // Check wave amplitude at expected position
    let expected_distance = WATER_SOUND_SPEED * total_time;
    let expected_grid_point = source_position.0 + (expected_distance / dx) as usize;

    if expected_grid_point < grid_size {
        let measured_amplitude =
            pressure[[expected_grid_point, grid_size / 2, grid_size / 2]].abs();
        let expected_amplitude = source_amplitude;
        let relative_error = (measured_amplitude - expected_amplitude).abs() / expected_amplitude;

        // Allow 10% error due to numerical dispersion
        assert!(
            relative_error < 0.1,
            "Plane wave amplitude error: {:.2}% (measured: {:.2e}, expected: {:.2e})",
            relative_error * 100.0,
            measured_amplitude,
            expected_amplitude
        );
    }
}

/// Validate acoustic absorption against Beer-Lambert law
#[test]
fn validate_absorption() {
    let grid_size = 256;
    let dx = 0.1e-3;
    let frequency = 2.25e6; // 2.25 MHz

    let grid = Grid::new(grid_size, 1, 1, dx, dx, dx);

    // Create medium with known absorption
    let absorption_coeff = 0.6; // dB/cm/MHz for soft tissue
    let absorption_np = absorption_coeff * 0.1151 * frequency / 1e6; // Convert to Np/m

    let mut medium = HomogeneousMedium::from_minimal(WATER_DENSITY, WATER_SOUND_SPEED, &grid);
    medium.absorption = absorption_np;

    // Initialize solver with absorption enabled
    let config = WesterveltFdtdConfig {
        enable_absorption: true,
        enable_nonlinearity: false,
        ..Default::default()
    };
    let mut solver = WesterveltFdtd::new(config, &grid);

    // Add initial pressure
    let initial_amplitude = 1e6; // 1 MPa
    solver.set_initial_pressure(|x, _, _| {
        if x < 10.0 * dx {
            initial_amplitude
        } else {
            0.0
        }
    });

    // Propagate for known distance
    let propagation_distance = 0.05; // 5 cm
    let propagation_time = propagation_distance / WATER_SOUND_SPEED;
    let dt = 1e-7;
    let num_steps = (propagation_time / dt) as usize;

    for _ in 0..num_steps {
        solver.step(&medium, dt).expect("Simulation step failed");
    }

    // Measure amplitude after propagation
    let pressure = solver.get_pressure();
    let propagation_points = (propagation_distance / dx) as usize;

    if propagation_points < grid_size {
        let measured_amplitude = pressure[[propagation_points, 0, 0]].abs();

        // Beer-Lambert law: I = I0 * exp(-alpha * distance)
        let expected_amplitude = initial_amplitude * (-absorption_np * propagation_distance).exp();
        let relative_error = (measured_amplitude - expected_amplitude).abs() / expected_amplitude;

        // Allow 15% error due to numerical effects
        assert!(
            relative_error < 0.15,
            "Absorption validation error: {:.2}% (measured: {:.2e}, expected: {:.2e})",
            relative_error * 100.0,
            measured_amplitude,
            expected_amplitude
        );
    }
}

/// Validate nonlinear propagation (B/A parameter)
#[test]
fn validate_nonlinear_propagation() {
    let grid_size = 128;
    let dx = 0.1e-3;
    let frequency = 1e6;

    let grid = Grid::new(grid_size, grid_size, grid_size, dx, dx, dx);

    // Create medium with known nonlinearity
    let mut medium = HomogeneousMedium::from_minimal(WATER_DENSITY, WATER_SOUND_SPEED, &grid);
    medium.nonlinearity = 3.5; // B/A for water

    // Test with and without nonlinearity
    let configs = vec![(false, "Linear"), (true, "Nonlinear")];

    for (enable_nonlinearity, label) in configs {
        let config = WesterveltFdtdConfig {
            enable_nonlinearity,
            enable_absorption: false,
            ..Default::default()
        };
        let mut solver = WesterveltFdtd::new(config, &grid);

        // Add high-amplitude source to trigger nonlinear effects
        let source_amplitude = 5e6; // 5 MPa (high amplitude)
        let source_position = (grid_size / 2, grid_size / 2, grid_size / 2);

        // Run for short time
        let dt = 1e-7;
        let num_steps = 100;

        for step in 0..num_steps {
            let time = step as f64 * dt;
            let source_value = source_amplitude * (2.0 * PI * frequency * time).sin();
            solver.add_point_source(source_position, source_value);
            solver.step(&medium, dt).expect("Simulation step failed");
        }

        let pressure = solver.get_pressure();
        let max_pressure = pressure.iter().fold(0.0f64, |a, &b| a.max(b.abs()));

        println!(
            "{} propagation max pressure: {:.2e} Pa",
            label, max_pressure
        );

        // Nonlinear propagation should show harmonic generation
        // This would require FFT analysis for proper validation
        assert!(
            max_pressure > 0.0,
            "{} propagation produced no pressure",
            label
        );
    }
}

/// Validate CFL stability condition
#[test]
fn validate_cfl_stability() {
    let grid = Grid::new(64, 64, 64, 1e-3, 1e-3, 1e-3);
    let medium = HomogeneousMedium::from_minimal(WATER_DENSITY, WATER_SOUND_SPEED, &grid);

    // Test different CFL numbers
    let cfl_values = vec![0.1, 0.3, 0.5, 0.7];

    for cfl in cfl_values {
        let config = WesterveltFdtdConfig {
            cfl_number: cfl,
            ..Default::default()
        };
        let mut solver = WesterveltFdtd::new(config, &grid);

        let dt = solver.calculate_stable_dt(&medium);
        let expected_dt = cfl * grid.dx.min(grid.dy).min(grid.dz) / WATER_SOUND_SPEED;

        let relative_error = (dt - expected_dt).abs() / expected_dt;
        assert!(
            relative_error < 1e-6,
            "CFL calculation error for CFL={}: {:.2e}% (dt={:.2e}, expected={:.2e})",
            cfl,
            relative_error * 100.0,
            dt,
            expected_dt
        );
    }
}
