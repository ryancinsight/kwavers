//! Plane Wave Source Injection Timing Test
//!
//! This test validates that plane wave sources are injected at boundaries
//! with correct timing, not as initial conditions across the domain.
//!
//! Mathematical Specification:
//! - Plane wave traveling in +z direction at speed c = 1500 m/s
//! - Grid: 64×64×64 with spacing 0.1 mm
//! - Source at z=0 boundary, sensor at z=32 (center)
//! - Expected arrival time: distance/c = 3.2e-3 / 1500 = 2.13 µs
//!
//! Author: Ryan Clanton (@ryancinsight)
//! Date: 2025-01-20

use kwavers::domain::grid::Grid;
use kwavers::domain::medium::HomogeneousMedium;
use kwavers::domain::signal::SineWave;
use kwavers::domain::source::{
    InjectionMode, PlaneWaveConfig, PlaneWaveSource, Source, SourceField,
};
use kwavers::physics::acoustics::mechanics::acoustic_wave::SpatialOrder;
use kwavers::simulation::backends::acoustic::backend::AcousticSolverBackend;
use kwavers::simulation::backends::acoustic::fdtd::FdtdBackend;
use std::sync::Arc;

#[test]
fn test_plane_wave_boundary_injection_timing() {
    // Setup: 64³ grid with 0.1 mm spacing
    let nx = 64;
    let ny = 64;
    let nz = 64;
    let dx = 0.1e-3; // 0.1 mm
    let dy = 0.1e-3;
    let dz = 0.1e-3;

    let grid = Grid::new(nx, ny, nz, dx, dy, dz).expect("Failed to create grid");

    // Medium: water (c=1500 m/s, ρ=1000 kg/m³)
    let medium = HomogeneousMedium::new(1000.0, 1500.0, 0.0, 0.0, &grid);

    // Create FDTD backend
    let mut backend =
        FdtdBackend::new(&grid, &medium, SpatialOrder::Second).expect("Failed to create backend");

    // Create plane wave source with BoundaryOnly injection mode
    let signal = Arc::new(SineWave::new(
        1e6, // 1 MHz
        1e5, // 100 kPa
        0.0, // phase
    ));

    let config = PlaneWaveConfig {
        direction: (0.0, 0.0, 1.0), // +z direction
        wavelength: 1500.0 / 1e6,   // c/f = 1.5 mm
        phase: 0.0,
        source_type: SourceField::Pressure,
        injection_mode: InjectionMode::BoundaryOnly,
    };

    let source = Arc::new(PlaneWaveSource::new(config, signal));

    // Check that mask is only at z=0 boundary
    let mask = source.create_mask(&grid);
    let mut num_active_points = 0;
    let mut max_z_with_source = 0;
    let mut min_z_with_source = nz;

    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                if mask[[i, j, k]].abs() > 1e-12 {
                    num_active_points += 1;
                    max_z_with_source = max_z_with_source.max(k);
                    min_z_with_source = min_z_with_source.min(k);
                }
            }
        }
    }

    println!("Mask analysis:");
    println!("  Active points: {}", num_active_points);
    println!("  Min z index: {}", min_z_with_source);
    println!("  Max z index: {}", max_z_with_source);
    println!("  Expected: {} points at z=0", nx * ny);

    // Verify mask is only at boundary
    assert_eq!(
        num_active_points,
        nx * ny,
        "Plane wave mask should have nx×ny points at boundary"
    );
    assert_eq!(
        min_z_with_source, 0,
        "Plane wave should start at z=0 boundary"
    );
    assert_eq!(
        max_z_with_source, 0,
        "Plane wave should only be at z=0 boundary"
    );

    // Add source to backend
    backend
        .add_source(source.clone())
        .expect("Failed to add source");

    let dt = backend.get_dt();
    println!("\nTime step: {:.3e} s", dt);

    // Check initial condition (t=0): pressure should be zero everywhere except possibly at boundary
    let p_initial = backend.get_pressure_field();
    let mut max_p_initial = 0.0;
    let mut max_p_z_index = 0;

    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                let p_val = p_initial[[i, j, k]].abs();
                if p_val > max_p_initial {
                    max_p_initial = p_val;
                    max_p_z_index = k;
                }
            }
        }
    }

    println!("\nInitial pressure (t=0):");
    println!("  Max |p|: {:.3e} Pa", max_p_initial);
    println!("  Location: z={} (index)", max_p_z_index);

    // Initial pressure should be essentially zero (no initial condition)
    assert!(
        max_p_initial < 1e-10,
        "Initial pressure should be zero (no initial condition), found max |p| = {:.3e}",
        max_p_initial
    );

    // Run simulation for expected arrival time
    let c = 1500.0; // m/s
    let sensor_distance = 32.0 * dz; // Distance to center (z=32)
    let expected_arrival_time = sensor_distance / c;
    let arrival_steps = (expected_arrival_time / dt).ceil() as usize;

    println!("\nPropagation parameters:");
    println!("  Sensor distance: {:.3e} m", sensor_distance);
    println!(
        "  Expected arrival time: {:.3e} s ({:.2} µs)",
        expected_arrival_time,
        expected_arrival_time * 1e6
    );
    println!("  Arrival steps: {}", arrival_steps);

    // Step through time and monitor pressure at sensor position
    let sensor_i = nx / 2;
    let sensor_j = ny / 2;
    let sensor_k = 32; // Center in z

    let mut pressure_history = Vec::new();
    let monitor_steps = arrival_steps + 500;

    for step in 0..monitor_steps {
        backend.step().expect("Failed to step");

        let p = backend.get_pressure_field();
        let p_sensor = p[[sensor_i, sensor_j, sensor_k]];
        let t = step as f64 * dt;

        pressure_history.push((t, p_sensor));

        // Print every 100 steps
        if step % 100 == 0 {
            let t_us = t * 1e6;
            println!("  t={:.2} µs: p_sensor={:.3e} Pa", t_us, p_sensor);
        }
    }

    // Find first arrival: when |p| exceeds 10% of expected amplitude
    let threshold = 0.1 * 1e5; // 10% of 100 kPa
    let mut first_arrival_time = None;

    for (t, p) in &pressure_history {
        if p.abs() > threshold {
            first_arrival_time = Some(*t);
            break;
        }
    }

    if let Some(arrival_time) = first_arrival_time {
        let arrival_time_us = arrival_time * 1e6;
        let expected_time_us = expected_arrival_time * 1e6;
        let error_us = (arrival_time - expected_arrival_time).abs() * 1e6;
        let error_percent = error_us / expected_time_us * 100.0;

        println!("\nArrival timing:");
        println!("  Expected: {:.2} µs", expected_time_us);
        println!("  Measured: {:.2} µs", arrival_time_us);
        println!("  Error: {:.2} µs ({:.1}%)", error_us, error_percent);

        // Allow 10% timing error (accounts for rise time, dispersion, etc.)
        assert!(
            error_percent < 10.0,
            "Arrival time error too large: {:.1}% (expected <10%)",
            error_percent
        );

        // Check that arrival is NOT at very early time (would indicate initial condition bug)
        assert!(
            arrival_time > 0.5e-6,
            "Arrival time suspiciously early ({:.2} µs), suggests initial condition bug",
            arrival_time_us
        );
    } else {
        panic!(
            "No wave arrival detected within {} steps ({:.2} µs)",
            monitor_steps,
            monitor_steps as f64 * dt * 1e6
        );
    }

    // Verify that pressure at distant points (before wave arrival) is near zero
    let early_time_idx = arrival_steps / 4; // Well before arrival
    let (t_early, p_early) = pressure_history[early_time_idx];

    println!("\nEarly time check (t={:.2} µs):", t_early * 1e6);
    println!("  p_sensor={:.3e} Pa", p_early);

    assert!(
        p_early.abs() < 1e3,
        "Pressure at sensor should be near zero before wave arrival, found {:.3e} Pa at t={:.2} µs",
        p_early,
        t_early * 1e6
    );

    println!("\n✓ Plane wave injection timing test PASSED");
}

#[test]
fn test_plane_wave_amplitude_scaling() {
    // Test that plane wave amplitude is correct and doesn't scale with number of boundary points

    let nx = 32;
    let ny = 32;
    let nz = 64;
    let dx = 0.1e-3;
    let dy = 0.1e-3;
    let dz = 0.1e-3;

    let grid = Grid::new(nx, ny, nz, dx, dy, dz).expect("Failed to create grid");
    let medium = HomogeneousMedium::new(1000.0, 1500.0, 0.0, 0.0, &grid);

    let mut backend =
        FdtdBackend::new(&grid, &medium, SpatialOrder::Second).expect("Failed to create backend");

    // Create plane wave with known amplitude
    let source_amplitude = 1e5; // 100 kPa
    let signal = Arc::new(SineWave::new(1e6, source_amplitude, 0.0));

    let config = PlaneWaveConfig {
        direction: (0.0, 0.0, 1.0),
        wavelength: 1500.0 / 1e6,
        phase: 0.0,
        source_type: SourceField::Pressure,
        injection_mode: InjectionMode::BoundaryOnly,
    };

    let source = Arc::new(PlaneWaveSource::new(config, signal));
    backend.add_source(source).expect("Failed to add source");

    let dt = backend.get_dt();
    let c = 1500.0;

    // Run to arrival at center
    let sensor_k = nz / 2;
    let distance = sensor_k as f64 * dz;
    let arrival_steps = ((distance / c) / dt * 1.5).ceil() as usize; // 1.5× to ensure arrival

    for _ in 0..arrival_steps {
        backend.step().expect("Failed to step");
    }

    // Check amplitude at sensor
    let p = backend.get_pressure_field();
    let p_sensor = p[[nx / 2, ny / 2, sensor_k]];
    let amplitude_ratio = p_sensor.abs() / source_amplitude;

    println!("\nAmplitude scaling test:");
    println!("  Source amplitude: {:.3e} Pa", source_amplitude);
    println!("  Measured amplitude: {:.3e} Pa", p_sensor.abs());
    println!("  Ratio: {:.2}", amplitude_ratio);

    // Allow factor of 2-3× due to discretization, but not 10× or 100×
    assert!(
        amplitude_ratio < 5.0,
        "Amplitude too large: {:.1}× source (suggests accumulation bug)",
        amplitude_ratio
    );

    assert!(
        amplitude_ratio > 0.1,
        "Amplitude too small: {:.1}× source (suggests normalization bug)",
        amplitude_ratio
    );

    println!("✓ Amplitude scaling test PASSED");
}
