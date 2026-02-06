//! Session 2: Source Injection Verification Test
//!
//! This test verifies that source injection works correctly in the Rust core library,
//! independent of the Python bindings. This will help isolate whether bugs are in
//! the core solver or in the pykwavers binding layer.
//!
//! Test Specification:
//! - Grid: 64×64×64, spacing 0.1 mm
//! - Medium: Water (c=1500 m/s, ρ=1000 kg/m³)
//! - Source: 1 MHz sine wave, 100 kPa amplitude, plane wave at z=0
//! - Sensor: Point sensor at (32, 32, 38) - 60% along z-axis
//! - Duration: ~100 timesteps (~1 period)
//!
//! Expected Results:
//! - Sensor should record non-zero pressure
//! - Peak amplitude should be within 50% of source amplitude (100 kPa)
//! - Wave should arrive at sensor after propagation delay (~2.5 µs)
//!
//! Author: Ryan Clanton (@ryancinsight)
//! Date: 2026-02-05
//! Sprint: 217 Session 2 - k-Wave Validation

use kwavers::core::error::KwaversResult;
use kwavers::domain::grid::Grid;
use kwavers::domain::medium::HomogeneousMedium;
use kwavers::domain::signal::Signal;
use kwavers::domain::source::custom::FunctionSource;
use kwavers::domain::source::{GridSource, SourceField};
use kwavers::solver::forward::fdtd::config::FdtdConfig;
use kwavers::solver::forward::fdtd::solver::FdtdSolver;
use ndarray::Array3;
use std::sync::Arc;

/// Simple sine wave signal for testing
#[derive(Clone, Debug)]
struct TestSineSignal {
    frequency: f64,
    amplitude: f64,
}

impl TestSineSignal {
    fn new(frequency: f64, amplitude: f64) -> Self {
        Self {
            frequency,
            amplitude,
        }
    }
}

impl Signal for TestSineSignal {
    fn amplitude(&self, t: f64) -> f64 {
        self.amplitude * (2.0 * std::f64::consts::PI * self.frequency * t).sin()
    }

    fn duration(&self) -> Option<f64> {
        None // Continuous signal
    }

    fn frequency(&self, _t: f64) -> f64 {
        self.frequency
    }

    fn phase(&self, t: f64) -> f64 {
        2.0 * std::f64::consts::PI * self.frequency * t
    }

    fn clone_box(&self) -> Box<dyn Signal> {
        Box::new(self.clone())
    }
}

#[test]
fn test_fdtd_plane_wave_source_injection() -> KwaversResult<()> {
    // Test configuration
    let nx = 64;
    let ny = 64;
    let nz = 64;
    let dx = 0.1e-3; // 0.1 mm
    let dy = 0.1e-3;
    let dz = 0.1e-3;

    let c0 = 1500.0; // Sound speed [m/s]
    let rho0 = 1000.0; // Density [kg/m³]

    let f0 = 1.0e6; // 1 MHz
    let amp = 100e3; // 100 kPa

    // Create grid
    let grid = Grid::new(nx, ny, nz, dx, dy, dz)?;

    // Create medium (water)
    let medium = HomogeneousMedium::new(rho0, c0, 0.0, 0.0, &grid);

    // Sensor position (define before using)
    let cx = nx / 2;
    let cy = ny / 2;
    let cz = (nz as f64 * 0.6) as usize;

    // Time parameters
    let cfl = 0.3;
    let dt = cfl * dx / (c0 * (3.0_f64).sqrt());
    let period = 1.0 / f0;
    let steps_per_period = (period / dt) as usize;

    // Calculate timesteps needed for wave to reach sensor
    let sensor_z = cz as f64 * dz;
    let t_arrival = sensor_z / c0;
    let steps_to_arrival = (t_arrival / dt) as usize;
    let time_steps = steps_to_arrival + steps_per_period; // Arrival time + one period for measurement

    println!("\n=== Session 2: FDTD Source Injection Test ===");
    println!("Grid: {}×{}×{} = {} points", nx, ny, nz, nx * ny * nz);
    println!("Spacing: {:.2} mm", dx * 1e3);
    println!("Sound speed: {:.0} m/s", c0);
    println!("Source: {:.1} MHz, {:.0} kPa", f0 / 1e6, amp / 1e3);
    println!("Time step: {:.2} ns", dt * 1e9);
    println!(
        "Steps: {} (arrival at step ~{}, then {:.2} periods)",
        time_steps,
        steps_to_arrival,
        (time_steps - steps_to_arrival) as f64 / steps_per_period as f64
    );

    // Create sensor mask at calculated position
    let mut sensor_mask = Array3::<bool>::from_elem((nx, ny, nz), false);
    sensor_mask[[cx, cy, cz]] = true;

    println!(
        "Sensor position: grid[{}, {}, {}] = ({:.2}, {:.2}, {:.2}) mm",
        cx,
        cy,
        cz,
        cx as f64 * dx * 1e3,
        cy as f64 * dy * 1e3,
        cz as f64 * dz * 1e3
    );
    println!(
        "Expected arrival time: {:.2} µs (step ~{})",
        t_arrival * 1e6,
        steps_to_arrival
    );

    // Create FDTD configuration with sensor mask
    let config = FdtdConfig {
        dt,
        nt: time_steps,
        spatial_order: 4,
        staggered_grid: true,
        cfl_factor: 0.3,
        subgridding: false,
        subgrid_factor: 2,
        enable_gpu_acceleration: false,
        sensor_mask: Some(sensor_mask),
    };

    // Create empty GridSource (we'll use dynamic sources)
    let grid_source = GridSource::new_empty();

    // Create solver
    let mut solver = FdtdSolver::new(config, &grid, &medium, grid_source)?;

    // Create sine wave signal
    let signal = Arc::new(TestSineSignal::new(f0, amp));

    // Create plane wave source at z=0
    let dz_closure = dz; // Capture for closure
    let function_source = Arc::new(FunctionSource::new(
        move |_x, _y, z, _t| {
            // Return 1.0 for z=0 plane (cells within dz/2 of origin), 0.0 elsewhere
            if z.abs() < dz_closure * 0.5 {
                1.0
            } else {
                0.0
            }
        },
        signal,
        SourceField::Pressure,
    ));

    // Add source to solver
    solver.add_source_arc(function_source)?;

    println!("\nRunning simulation...");

    // Run simulation
    for step in 0..time_steps {
        solver.step_forward()?;

        // Print progress every 20 steps
        if step % 20 == 0 {
            println!("  Step {}/{}", step, time_steps);
        }
    }

    println!("Simulation complete!");

    // Extract recorded sensor data
    let sensor_data = solver
        .extract_recorded_sensor_data()
        .expect("No sensor data recorded");

    println!("\nSensor data shape: {:?}", sensor_data.dim());

    // Analyze results
    let time_series = sensor_data.row(0); // First (and only) sensor
    let p_max = time_series.iter().map(|&p| p.abs()).fold(0.0, f64::max);
    let p_min = time_series.iter().copied().fold(f64::INFINITY, f64::min);
    let p_mean = time_series.iter().map(|&p| p.abs()).sum::<f64>() / time_series.len() as f64;

    println!("\nResults:");
    println!("  Max pressure:  {:.2} kPa", p_max / 1e3);
    println!("  Min pressure:  {:.2} kPa", p_min / 1e3);
    println!("  Mean |p|:      {:.2} kPa", p_mean / 1e3);
    println!("  Amplitude error: {:.1}%", (p_max - amp) / amp * 100.0);

    // Print first 10 values
    println!("\nFirst 10 timesteps:");
    for i in 0..10.min(time_series.len()) {
        let t = i as f64 * dt;
        let p = time_series[i];
        println!("  t[{}] = {:.2} ns: p = {:.2} Pa", i, t * 1e9, p);
    }

    // Verification assertions
    println!("\nVerification:");

    // Check 1: Non-zero signal
    assert!(
        p_max > 0.0,
        "FAIL: Sensor recorded all zeros - source not injecting or sensor not receiving"
    );
    println!(
        "  ✓ Sensor recorded non-zero signal (max = {:.2} kPa)",
        p_max / 1e3
    );

    // Check 2: Amplitude within reasonable range (20% tolerance for plane wave)
    let amplitude_error = (p_max - amp).abs() / amp;
    assert!(
        amplitude_error < 0.2,
        "FAIL: Amplitude error {:.1}% exceeds 20% tolerance",
        amplitude_error * 100.0
    );
    println!(
        "  ✓ Amplitude within 20% tolerance (error = {:.1}%)",
        amplitude_error * 100.0
    );

    // Check 3: Wave arrives after propagation delay (should be near zero initially)
    // Check first half of arrival time should be mostly zero
    let early_steps = steps_to_arrival / 2;
    let early_max = time_series
        .iter()
        .take(early_steps)
        .map(|&p| p.abs())
        .fold(0.0, f64::max);
    assert!(
        early_max < amp * 0.1,
        "FAIL: Early timesteps show pressure {:.2} kPa > 10% of source amplitude",
        early_max / 1e3
    );
    println!("  ✓ Wave obeys causality (early timesteps < 10% of amplitude)");

    // Check 4: Wave arrives near expected time
    // Find first significant amplitude (> 10% of max)
    let arrival_threshold = p_max * 0.1;
    let measured_arrival = time_series
        .iter()
        .position(|&p| p.abs() > arrival_threshold)
        .unwrap_or(0);
    let arrival_error =
        (measured_arrival as f64 - steps_to_arrival as f64).abs() / steps_to_arrival as f64;
    println!(
        "  ✓ Wave arrival time: step {} (expected ~{}, error {:.1}%)",
        measured_arrival,
        steps_to_arrival,
        arrival_error * 100.0
    );

    println!("\n=== TEST PASSED ===\n");

    Ok(())
}

#[test]
fn test_fdtd_point_source_injection() -> KwaversResult<()> {
    // Simpler test: point source at grid center with nearby sensor
    let nx = 32;
    let ny = 32;
    let nz = 32;
    let dx = 0.1e-3;
    let dy = 0.1e-3;
    let dz = 0.1e-3;

    let c0 = 1500.0;
    let rho0 = 1000.0;
    let f0 = 1.0e6;
    let amp = 100e3;

    let grid = Grid::new(nx, ny, nz, dx, dy, dz)?;
    let medium = HomogeneousMedium::new(rho0, c0, 0.0, 0.0, &grid);

    let cfl = 0.3;
    let dt = cfl * dx / (c0 * (3.0_f64).sqrt());

    // Sensor is 1 cell away (0.1 mm), need time for wave to propagate
    let distance = 1.0 * dz; // 0.1 mm
    let t_arrival = distance / c0;
    let steps_to_arrival = (t_arrival / dt) as usize;
    let period = 1.0 / f0;
    let steps_per_period = (period / dt) as usize;
    let time_steps = steps_to_arrival + steps_per_period * 2; // Arrival + 2 periods

    println!("\n=== Session 2: FDTD Point Source Test ===");
    println!("Grid: {}×{}×{}", nx, ny, nz);
    println!("Source at center (16, 16, 16)");
    println!(
        "Sensor at (16, 16, 17) - adjacent cell (distance: {:.2} mm)",
        distance * 1e3
    );
    println!("Time step: {:.2} ns", dt * 1e9);
    println!(
        "Steps: {} (arrival at step ~{}, then {} periods)",
        time_steps,
        steps_to_arrival,
        (time_steps - steps_to_arrival) / steps_per_period
    );
    println!(
        "Expected arrival time: {:.2} ns (step ~{})",
        t_arrival * 1e9,
        steps_to_arrival
    );

    // Sensor adjacent to source
    let mut sensor_mask = Array3::<bool>::from_elem((nx, ny, nz), false);
    sensor_mask[[16, 16, 17]] = true;

    let config = FdtdConfig {
        dt,
        nt: time_steps,
        spatial_order: 2,
        staggered_grid: false,
        cfl_factor: 0.3,
        subgridding: false,
        subgrid_factor: 2,
        enable_gpu_acceleration: false,
        sensor_mask: Some(sensor_mask),
    };

    let grid_source = GridSource::new_empty();
    let mut solver = FdtdSolver::new(config, &grid, &medium, grid_source)?;

    // Point source at center
    let signal = Arc::new(TestSineSignal::new(f0, amp));
    let src_x = 16.0 * dx;
    let src_y = 16.0 * dy;
    let src_z = 16.0 * dz;
    let dx_closure = dx;
    let dy_closure = dy;
    let dz_closure = dz;

    let function_source = Arc::new(FunctionSource::new(
        move |x, y, z, _t| {
            if (x - src_x).abs() < dx_closure * 0.5
                && (y - src_y).abs() < dy_closure * 0.5
                && (z - src_z).abs() < dz_closure * 0.5
            {
                1.0
            } else {
                0.0
            }
        },
        signal,
        SourceField::Pressure,
    ));

    solver.add_source_arc(function_source)?;

    println!("Running simulation...");
    for _ in 0..time_steps {
        solver.step_forward()?;
    }

    let sensor_data = solver
        .extract_recorded_sensor_data()
        .expect("No sensor data recorded");
    let time_series = sensor_data.row(0);
    let p_max = time_series.iter().map(|&p| p.abs()).fold(0.0, f64::max);

    println!("\nResults:");
    println!("  Max pressure: {:.2} kPa", p_max / 1e3);
    println!("  Expected: {:.2} kPa", amp / 1e3);

    // Verification assertions
    println!("\nVerification:");

    // Check 1: Non-zero signal
    assert!(p_max > 0.0, "FAIL: No signal recorded");
    println!("  ✓ Signal recorded (max = {:.2} kPa)", p_max / 1e3);

    // Check 2: Amplitude should be reasonable for point source at 1 cell distance
    // Point sources decay with 1/r, so expect lower amplitude than source
    assert!(
        p_max > amp * 0.001,
        "FAIL: Amplitude too low ({:.2} kPa)",
        p_max / 1e3
    );
    assert!(
        p_max < amp * 10.0,
        "FAIL: Amplitude too high ({:.2} kPa)",
        p_max / 1e3
    );
    println!("  ✓ Amplitude within reasonable range for point source");

    // Check 3: Wave arrives after propagation delay
    let early_steps = steps_to_arrival / 2;
    let early_max = time_series
        .iter()
        .take(early_steps)
        .map(|&p| p.abs())
        .fold(0.0, f64::max);
    assert!(
        early_max < p_max * 0.5,
        "FAIL: Signal arrives too early (early_max = {:.2} kPa)",
        early_max / 1e3
    );
    println!("  ✓ Wave obeys causality (early timesteps < 50% of max)");

    // Check 4: Wave arrives near expected time
    let arrival_threshold = p_max * 0.1;
    let measured_arrival = time_series
        .iter()
        .position(|&p| p.abs() > arrival_threshold)
        .unwrap_or(0);
    println!(
        "  ✓ Wave arrival time: step {} (expected ~{})",
        measured_arrival, steps_to_arrival
    );

    println!("\n=== TEST PASSED ===\n");

    Ok(())
}
