//! Test plane wave source injection with correct boundary semantics
//!
//! This test verifies that:
//! 1. Boundary plane sources use Dirichlet mode (enforce pressure value)
//! 2. Point sources use additive mode with proper normalization
//! 3. Arrival times match theoretical expectations
//! 4. Amplitudes are correct and not accumulated

use kwavers::core::error::KwaversResult;
use kwavers::domain::grid::Grid;
use kwavers::domain::medium::HomogeneousMedium;
use kwavers::domain::signal::SineWave;
use kwavers::domain::source::{
    InjectionMode, PlaneWaveConfig, PlaneWaveSource, PointSource, SourceField,
};
use kwavers::solver::forward::fdtd::{FdtdConfig, FdtdSolver};
use kwavers::solver::forward::pstd::config::KSpaceMethod;
use kwavers::solver::forward::pstd::{PSTDConfig, PSTDSolver};
use kwavers::solver::interface::Solver;
use std::sync::Arc;

/// Test helper to analyze arrival time and amplitude
fn analyze_pressure_signal(pressure: &[f64], dt: f64, threshold_factor: f64) -> (Option<f64>, f64) {
    let max_amp = pressure.iter().map(|&p| p.abs()).fold(0.0, f64::max);
    let threshold = threshold_factor * max_amp;

    // Find first arrival time
    let arrival_time = pressure
        .iter()
        .enumerate()
        .find(|(_, &p)| p.abs() > threshold)
        .map(|(i, _)| i as f64 * dt);

    (arrival_time, max_amp)
}

#[test]
fn test_plane_wave_boundary_injection_fdtd() -> KwaversResult<()> {
    // Setup: 32^3 grid, 0.1mm spacing, 1MHz, 100kPa amplitude
    let nx = 32;
    let ny = 32;
    let nz = 32;
    let dx = 0.1e-3; // 0.1 mm
    let c0 = 1500.0; // m/s (water)
    let rho0 = 1000.0; // kg/m^3
    let frequency = 1e6; // 1 MHz
    let amplitude = 1e5; // 100 kPa

    let grid = Grid::new(nx, ny, nz, dx, dx, dx)?;
    let medium = HomogeneousMedium::new(rho0, c0, 0.0, 0.0, &grid);

    // Create plane wave source propagating in +z direction with boundary-only injection
    let signal = Arc::new(SineWave::new(frequency, amplitude, 0.0));
    let config = PlaneWaveConfig {
        direction: (0.0, 0.0, 1.0),
        wavelength: c0 / frequency, // 1.5 mm
        phase: 0.0,
        source_type: SourceField::Pressure,
        injection_mode: InjectionMode::BoundaryOnly,
    };
    let source = PlaneWaveSource::new(config, signal);

    // Create FDTD solver
    let dt = 0.3 * dx / c0; // CFL = 0.3
    let nt = 300; // Reduced for faster testing
    let fdtd_config = FdtdConfig {
        dt,
        spatial_order: 2,
        ..Default::default()
    };

    let mut solver = FdtdSolver::new(fdtd_config, &grid, &medium, Default::default())?;
    solver.add_source_arc(Arc::new(source))?;

    // Run simulation and record pressure at center point
    let center_idx = (nx / 2, ny / 2, nz / 2);
    let mut pressure_history = Vec::with_capacity(nt);

    for _ in 0..nt {
        solver.step_forward()?;
        let p = solver.fields.p[[center_idx.0, center_idx.1, center_idx.2]];
        pressure_history.push(p);
    }

    // Analyze results
    let (arrival_time, max_amp) = analyze_pressure_signal(&pressure_history, dt, 0.01);

    // Expected arrival time: distance / speed = (nz/2 * dz) / c0
    let distance = (nz / 2) as f64 * dx;
    let expected_arrival = distance / c0;

    println!("Plane Wave Boundary Injection (FDTD):");
    println!("  Expected arrival: {:.3} μs", expected_arrival * 1e6);
    println!(
        "  Actual arrival: {:.3} μs",
        arrival_time.unwrap_or(0.0) * 1e6
    );
    println!("  Expected amplitude: {:.2e} Pa", amplitude);
    println!("  Actual amplitude: {:.2e} Pa", max_amp);
    println!("  Amplitude ratio: {:.3}", max_amp / amplitude.max(1e-10));

    // Assertions
    assert!(
        arrival_time.is_some(),
        "No arrival detected in pressure signal"
    );
    let arrival = arrival_time.unwrap();

    // Arrival time should be within 20% of expected (accounting for numerical dispersion)
    let arrival_error = (arrival - expected_arrival).abs() / expected_arrival;
    assert!(
        arrival_error < 0.20,
        "Arrival time error {:.1}% exceeds 20%",
        arrival_error * 100.0
    );

    // Amplitude should be within factor of 2 of expected (accounting for boundary effects)
    let amp_ratio = max_amp / amplitude;
    assert!(
        amp_ratio > 0.5 && amp_ratio < 2.0,
        "Amplitude ratio {:.2} outside acceptable range [0.5, 2.0]",
        amp_ratio
    );

    Ok(())
}

#[test]
fn test_plane_wave_boundary_injection_pstd() -> KwaversResult<()> {
    // Setup: smaller grid for faster testing
    let nx = 64;
    let ny = 64;
    let nz = 64;
    let dx = 0.1e-3; // 0.1 mm
    let c0 = 1500.0; // m/s (water)
    let rho0 = 1000.0; // kg/m^3
    let frequency = 1e6; // 1 MHz
    let amplitude = 1e5; // 100 kPa

    let grid = Grid::new(nx, ny, nz, dx, dx, dx)?;
    let medium = HomogeneousMedium::new(rho0, c0, 0.0, 0.0, &grid);

    // Create plane wave source propagating in +z direction with boundary-only injection
    let signal = Arc::new(SineWave::new(frequency, amplitude, 0.0));
    let config = PlaneWaveConfig {
        direction: (0.0, 0.0, 1.0),
        wavelength: c0 / frequency, // 1.5 mm
        phase: 0.0,
        source_type: SourceField::Pressure,
        injection_mode: InjectionMode::BoundaryOnly,
    };
    let source = PlaneWaveSource::new(config, signal);

    // Create PSTD solver
    let dt = 0.3 * dx / c0; // CFL = 0.3
    let nt = 300; // Reduced for faster testing
    let pstd_config = PSTDConfig {
        dt,
        nt,
        kspace_method: KSpaceMethod::StandardPSTD,
        boundary: kwavers::solver::forward::pstd::config::BoundaryConfig::None,
        ..Default::default()
    };

    let mut solver = PSTDSolver::new(pstd_config, grid.clone(), &medium, Default::default())?;
    solver.add_source(Box::new(source))?;

    // Run simulation and record pressure at center point and boundary
    let center_idx = (nx / 2, ny / 2, nz / 2);
    let boundary_idx = (nx / 2, ny / 2, 0); // z=0 boundary
    let mut pressure_history = Vec::with_capacity(nt);
    let mut boundary_pressure = Vec::with_capacity(nt);

    println!("\nDEBUG: PSTD Source Timing Analysis");
    println!("  Grid: {}x{}x{}, dx={:.2e} m", nx, ny, nz, dx);
    println!("  c0={} m/s, dt={:.3e} s", c0, dt);
    println!("  Center point: {:?}", center_idx);
    println!("  Boundary point: {:?}", boundary_idx);
    println!("  Expected distance: {:.2e} m", (nz / 2) as f64 * dx);
    println!(
        "  Expected travel time: {:.3} μs",
        ((nz / 2) as f64 * dx / c0) * 1e6
    );

    for i in 0..nt {
        solver.step_forward()?;
        let p_center = solver.fields.p[[center_idx.0, center_idx.1, center_idx.2]];
        let p_boundary = solver.fields.p[[boundary_idx.0, boundary_idx.1, boundary_idx.2]];
        pressure_history.push(p_center);
        boundary_pressure.push(p_boundary);

        // Log first few timesteps
        if i < 10 {
            println!(
                "  t={:.3e} s: p_boundary={:.2e}, p_center={:.2e}",
                i as f64 * dt,
                p_boundary,
                p_center
            );
        }
    }

    // Analyze results
    let (arrival_time, max_amp) = analyze_pressure_signal(&pressure_history, dt, 0.01);
    let (boundary_arrival, boundary_amp) = analyze_pressure_signal(&boundary_pressure, dt, 0.01);

    println!("\nBoundary Signal:");
    println!("  Arrival: {:.3} μs", boundary_arrival.unwrap_or(0.0) * 1e6);
    println!("  Max amplitude: {:.2e} Pa", boundary_amp);

    // Expected arrival time: distance / speed = (nz/2 * dz) / c0
    let distance = (nz / 2) as f64 * dx;
    let expected_arrival = distance / c0;

    println!("Plane Wave Boundary Injection (PSTD):");
    println!("  Expected arrival: {:.3} μs", expected_arrival * 1e6);
    println!(
        "  Actual arrival: {:.3} μs",
        arrival_time.unwrap_or(0.0) * 1e6
    );
    println!("  Expected amplitude: {:.2e} Pa", amplitude);
    println!("  Actual amplitude: {:.2e} Pa", max_amp);
    println!("  Amplitude ratio: {:.3}", max_amp / amplitude.max(1e-10));

    // Assertions
    assert!(
        arrival_time.is_some(),
        "No arrival detected in pressure signal"
    );
    let arrival = arrival_time.unwrap();

    // Arrival time should be within 20% of expected (PSTD is more accurate)
    let arrival_error = (arrival - expected_arrival).abs() / expected_arrival;
    assert!(
        arrival_error < 0.20,
        "Arrival time error {:.1}% exceeds 20%",
        arrival_error * 100.0
    );

    // Amplitude should be within factor of 2 of expected
    let amp_ratio = max_amp / amplitude;
    assert!(
        amp_ratio > 0.5 && amp_ratio < 2.0,
        "Amplitude ratio {:.2} outside acceptable range [0.5, 2.0]",
        amp_ratio
    );

    Ok(())
}

#[test]
fn test_point_source_normalization_fdtd() -> KwaversResult<()> {
    // Setup: smaller grid for point source
    let nx = 32;
    let ny = 32;
    let nz = 32;
    let dx = 0.1e-3; // 0.1 mm
    let c0 = 1500.0; // m/s (water)
    let rho0 = 1000.0; // kg/m^3
    let frequency = 1e6; // 1 MHz
    let amplitude = 1e5; // 100 kPa

    let grid = Grid::new(nx, ny, nz, dx, dx, dx)?;
    let medium = HomogeneousMedium::new(rho0, c0, 0.0, 0.0, &grid);

    // Create point source at center
    let signal = Arc::new(SineWave::new(frequency, amplitude, 0.0));
    let position = (
        (nx / 2) as f64 * dx,
        (ny / 2) as f64 * dx,
        (nz / 2) as f64 * dx,
    );
    let source = PointSource::new(position, signal);

    // Create FDTD solver
    let dt = 0.3 * dx / c0; // CFL = 0.3
    let nt = 200; // Reduced for faster testing
    let fdtd_config = FdtdConfig {
        dt,
        spatial_order: 2,
        ..Default::default()
    };

    let mut solver = FdtdSolver::new(fdtd_config, &grid, &medium, Default::default())?;
    solver.add_source_arc(Arc::new(source))?;

    // Run simulation and record pressure at source location
    let source_idx = (nx / 2, ny / 2, nz / 2);
    let mut pressure_history = Vec::with_capacity(nt);

    for _ in 0..nt {
        solver.step_forward()?;
        let p = solver.fields.p[[source_idx.0, source_idx.1, source_idx.2]];
        pressure_history.push(p);
    }

    // Analyze results
    let (_, max_amp) = analyze_pressure_signal(&pressure_history, dt, 0.01);

    println!("Point Source Normalization (FDTD):");
    println!("  Expected amplitude: {:.2e} Pa", amplitude);
    println!("  Actual amplitude: {:.2e} Pa", max_amp);
    println!("  Amplitude ratio: {:.3}", max_amp / amplitude.max(1e-10));

    // Point source amplitude should be reasonable (not 100× too large)
    // Allow larger range since point source has singularity at center
    let amp_ratio = max_amp / amplitude;
    assert!(
        amp_ratio > 0.1 && amp_ratio < 10.0,
        "Point source amplitude ratio {:.2} outside acceptable range [0.1, 10.0]",
        amp_ratio
    );

    Ok(())
}

#[test]
fn test_boundary_vs_fullgrid_injection() -> KwaversResult<()> {
    // Compare BoundaryOnly vs FullGrid injection modes
    let nx = 32;
    let ny = 32;
    let nz = 32;
    let dx = 0.1e-3;
    let c0 = 1500.0;
    let rho0 = 1000.0;
    let frequency = 1e6;
    let amplitude = 1e5;

    let grid = Grid::new(nx, ny, nz, dx, dx, dx)?;
    let medium = HomogeneousMedium::new(rho0, c0, 0.0, 0.0, &grid);

    let dt = 0.3 * dx / c0;
    let nt = 200; // Reduced for faster testing

    // Test BoundaryOnly mode
    let signal_boundary = Arc::new(SineWave::new(frequency, amplitude, 0.0));
    let config_boundary = PlaneWaveConfig {
        direction: (0.0, 0.0, 1.0),
        wavelength: c0 / frequency,
        phase: 0.0,
        source_type: SourceField::Pressure,
        injection_mode: InjectionMode::BoundaryOnly,
    };
    let source_boundary = PlaneWaveSource::new(config_boundary, signal_boundary);

    let fdtd_config = FdtdConfig {
        dt,
        spatial_order: 2,
        ..Default::default()
    };

    let mut solver_boundary =
        FdtdSolver::new(fdtd_config.clone(), &grid, &medium, Default::default())?;
    solver_boundary.add_source_arc(Arc::new(source_boundary))?;

    let center_idx = (nx / 2, ny / 2, nz / 2);
    let mut pressure_boundary = Vec::with_capacity(nt);

    for _ in 0..nt {
        solver_boundary.step_forward()?;
        let p = solver_boundary.fields.p[[center_idx.0, center_idx.1, center_idx.2]];
        pressure_boundary.push(p);
    }

    // Test FullGrid mode
    let signal_full = Arc::new(SineWave::new(frequency, amplitude, 0.0));
    let config_full = PlaneWaveConfig {
        direction: (0.0, 0.0, 1.0),
        wavelength: c0 / frequency,
        phase: 0.0,
        source_type: SourceField::Pressure,
        injection_mode: InjectionMode::FullGrid,
    };
    let source_full = PlaneWaveSource::new(config_full, signal_full);

    let mut solver_full = FdtdSolver::new(fdtd_config, &grid, &medium, Default::default())?;
    solver_full.add_source_arc(Arc::new(source_full))?;

    let mut pressure_full = Vec::with_capacity(nt);

    for _ in 0..nt {
        solver_full.step_forward()?;
        let p = solver_full.fields.p[[center_idx.0, center_idx.1, center_idx.2]];
        pressure_full.push(p);
    }

    // Analyze both
    let (arrival_boundary, amp_boundary) = analyze_pressure_signal(&pressure_boundary, dt, 0.01);
    let (arrival_full, amp_full) = analyze_pressure_signal(&pressure_full, dt, 0.01);

    let distance = (nz / 2) as f64 * dx;
    let expected_arrival = distance / c0;

    println!("Boundary vs FullGrid Injection:");
    println!("  Expected arrival: {:.3} μs", expected_arrival * 1e6);
    println!(
        "  BoundaryOnly arrival: {:.3} μs",
        arrival_boundary.unwrap_or(0.0) * 1e6
    );
    println!(
        "  FullGrid arrival: {:.3} μs",
        arrival_full.unwrap_or(0.0) * 1e6
    );
    println!("  BoundaryOnly amplitude: {:.2e} Pa", amp_boundary);
    println!("  FullGrid amplitude: {:.2e} Pa", amp_full);

    // BoundaryOnly should have correct arrival time
    assert!(arrival_boundary.is_some(), "BoundaryOnly: no arrival");
    let boundary_error = (arrival_boundary.unwrap() - expected_arrival).abs() / expected_arrival;
    assert!(
        boundary_error < 0.20,
        "BoundaryOnly arrival error {:.1}% exceeds 20%",
        boundary_error * 100.0
    );

    // FullGrid will have early arrival (instantaneous everywhere due to spatial pattern)
    // This demonstrates the difference between the two modes
    if let Some(full_arrival) = arrival_full {
        let full_error = (full_arrival - expected_arrival).abs() / expected_arrival;
        println!(
            "  FullGrid timing error: {:.1}% (expected, due to spatial pre-population)",
            full_error * 100.0
        );
    }

    Ok(())
}

#[test]
fn test_no_amplitude_accumulation() -> KwaversResult<()> {
    // Verify that boundary sources don't accumulate amplitude over time
    let nx = 32;
    let ny = 32;
    let nz = 32;
    let dx = 0.1e-3;
    let c0 = 1500.0;
    let rho0 = 1000.0;
    let frequency = 1e6;
    let amplitude = 1e5;

    let grid = Grid::new(nx, ny, nz, dx, dx, dx)?;
    let medium = HomogeneousMedium::new(rho0, c0, 0.0, 0.0, &grid);

    let signal = Arc::new(SineWave::new(frequency, amplitude, 0.0));
    let config = PlaneWaveConfig {
        direction: (0.0, 0.0, 1.0),
        wavelength: c0 / frequency,
        phase: 0.0,
        source_type: SourceField::Pressure,
        injection_mode: InjectionMode::BoundaryOnly,
    };
    let source = PlaneWaveSource::new(config, signal);

    let dt = 0.3 * dx / c0;
    let nt = 500; // Reduced for faster testing
    let fdtd_config = FdtdConfig {
        dt,
        spatial_order: 2,
        ..Default::default()
    };

    let mut solver = FdtdSolver::new(fdtd_config, &grid, &medium, Default::default())?;
    solver.add_source_arc(Arc::new(source))?;

    // Check pressure at boundary (where source is applied)
    let boundary_idx = (nx / 2, ny / 2, 0);

    let mut max_pressure: f64 = 0.0;
    let mut min_pressure: f64 = 0.0;

    for step in 0..nt {
        solver.step_forward()?;
        let p = solver.fields.p[[boundary_idx.0, boundary_idx.1, boundary_idx.2]];

        if step > nt / 4 {
            // After initial transient
            max_pressure = max_pressure.max(p);
            min_pressure = min_pressure.min(p);
        }
    }

    let peak_pressure = max_pressure.abs().max(min_pressure.abs());

    println!("Amplitude Accumulation Test:");
    println!("  Expected amplitude: {:.2e} Pa", amplitude);
    println!("  Peak pressure at boundary: {:.2e} Pa", peak_pressure);
    println!("  Ratio: {:.3}", peak_pressure / amplitude.max(1e-10));

    // Peak pressure should not grow indefinitely
    // Should be on same order of magnitude as source amplitude
    let ratio = peak_pressure / amplitude;
    assert!(
        ratio < 5.0,
        "Peak pressure ratio {:.2} suggests amplitude accumulation (should be < 5.0)",
        ratio
    );

    Ok(())
}
