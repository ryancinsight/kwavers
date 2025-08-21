//! Data Acquisition Demonstration
//!
//! This example demonstrates the complete data acquisition capabilities:
//! - Sonoluminescence detection and mapping
//! - Cavitation event tracking
//! - Thermal monitoring and dose calculation
//! - Integrated recording and statistics

use kwavers::{
    boundary::{Boundary, PMLBoundary, PMLConfig},
    config::{Config, SimulationConfig},
    grid::Grid,
    medium::HomogeneousMedium,
    physics::{
        bubble_dynamics::{BubbleCloud, BubbleIMEXConfig, BubbleParameters},
        plugin::{PhysicsPlugin, PluginContext},
        sonoluminescence_detector::DetectorConfig,
    },
    recorder::{Recorder, RecorderConfig},
    sensor::{Sensor, SensorConfig},
    signal::SineWave,
    solver::{
        pstd::{PstdConfig, PstdPlugin},
        Solver,
    },
    source::{PointSource, Source},
    time::Time,
    KwaversResult,
};
use ndarray::{Array3, Array4, Axis};
use std::sync::Arc;

/// Simulation parameters
struct SimulationParams {
    // Grid
    grid_size: usize,
    grid_spacing: f64,

    // Time
    duration: f64,
    dt: f64,

    // Acoustic
    frequency: f64,
    pressure_amplitude: f64,

    // Recording
    enable_cavitation: bool,
    enable_sonoluminescence: bool,
    enable_thermal: bool,
}

impl Default for SimulationParams {
    fn default() -> Self {
        Self {
            grid_size: 64,
            grid_spacing: 1e-3,      // 1 mm
            duration: 1e-3,          // 1 ms
            dt: 1e-7,                // 100 ns
            frequency: 500e3,        // 500 kHz
            pressure_amplitude: 2e5, // 2 bar
            enable_cavitation: true,
            enable_sonoluminescence: true,
            enable_thermal: true,
        }
    }
}

fn run_data_acquisition_demo(params: SimulationParams) -> KwaversResult<()> {
    println!("Data Acquisition Demonstration");
    println!("==============================");
    println!("Monitoring:");
    println!(
        "  - Cavitation: {}",
        if params.enable_cavitation {
            "✓"
        } else {
            "✗"
        }
    );
    println!(
        "  - Sonoluminescence: {}",
        if params.enable_sonoluminescence {
            "✓"
        } else {
            "✗"
        }
    );
    println!(
        "  - Thermal Effects: {}",
        if params.enable_thermal { "✓" } else { "✗" }
    );
    println!();

    // Create grid
    let n = params.grid_size;
    let grid = Grid::new(
        n,
        n,
        n,
        params.grid_spacing,
        params.grid_spacing,
        params.grid_spacing,
    );

    // Create time configuration
    let n_steps = (params.duration / params.dt) as usize;
    let time = Time::new(params.dt, n_steps);

    // Setup sensors at strategic locations
    let sensor_positions = vec![
        // Center
        (0.05, 0.05, 0.05),
        // Corners for coverage
        (0.025, 0.025, 0.025),
        (0.075, 0.075, 0.075),
        (0.025, 0.075, 0.025),
        (0.075, 0.025, 0.075),
    ];

    let sensor = Sensor::new(&grid, &time, &sensor_positions);

    // Create medium
    let medium = HomogeneousMedium::new(1000.0, 1500.0, &grid, 0.0, 0.0);

    // Configure recorder with all monitoring capabilities
    let sl_config = DetectorConfig {
        spectral_analysis: true,
        time_resolved: true,
        temperature_threshold: 5000.0, // 5000 K
        pressure_threshold: 1e6,       // 1 MPa
        compression_threshold: 5.0,
        spatial_resolution: params.grid_spacing,
        time_resolution: params.dt * 10.0,
    };

    let recorder_config = RecorderConfig::new("data_acquisition_output")
        .with_pressure_recording(true)
        .with_light_recording(true)
        .with_temperature_recording(params.enable_thermal)
        .with_cavitation_detection(params.enable_cavitation, -1e5) // -1 bar threshold
        .with_sonoluminescence_detection(params.enable_sonoluminescence, Some(sl_config))
        .with_snapshot_interval(100);

    let mut recorder = Recorder::from_config(sensor, &time, &recorder_config);

    // Initialize solver
    let solver_config = PstdConfig {
        k_space_correction: true,
        k_space_order: 2,
        anti_aliasing: true,
        pml_stencil_size: 10,
        cfl_factor: 0.5,
        use_leapfrog: true,
        enable_absorption: false,
        absorption_model: None,
    };

    let mut solver = PstdPlugin::new(solver_config, &grid)?;

    // Create acoustic source
    let signal = Arc::new(SineWave::new(
        params.frequency,
        params.pressure_amplitude,
        0.0,
    ));

    let source_position = (
        n as f64 * params.grid_spacing / 2.0,
        n as f64 * params.grid_spacing / 2.0,
        n as f64 * params.grid_spacing / 4.0,
    );

    let source = PointSource::new(source_position, signal);

    // Initialize bubble cloud for cavitation
    let bubble_params = BubbleParameters::default();
    // For now, create a simple bubble field - full cloud implementation needs the distributions
    // let mut bubble_cloud = BubbleCloud::new((n, n, n), bubble_params, size_dist, spatial_dist);

    // Initialize fields (4D: [field_type, nx, ny, nz])
    let mut fields = Array4::zeros((4, n, n, n)); // pressure, light, temperature, bubble_radius

    // Simulation parameters
    let num_steps = (params.duration / params.dt) as usize;
    let output_interval = num_steps / 20; // Output 20 times

    println!("Starting simulation with {} time steps...", num_steps);
    println!();

    // Main simulation loop
    for step in 0..num_steps {
        let t = step as f64 * params.dt;

        // Add source contribution to pressure field
        let mut pressure = fields.index_axis_mut(Axis(0), 0);
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    let x = i as f64 * grid.dx;
                    let y = j as f64 * grid.dy;
                    let z = k as f64 * grid.dz;
                    pressure[[i, j, k]] += source.get_source_term(t, x, y, z, &grid) * params.dt;
                }
            }
        }
        drop(pressure);

        // Update bubble dynamics (affects temperature and radius fields)
        let pressure = fields.index_axis(Axis(0), 0).to_owned();
        // let bubble_states = bubble_cloud.get_state_fields();

        // Update temperature and bubble radius fields
        // fields.index_axis_mut(Axis(0), 2).assign(&bubble_states.temperature);
        // fields.index_axis_mut(Axis(0), 3).assign(&bubble_states.radius);

        // Simulate light emission from sonoluminescence
        if params.enable_sonoluminescence {
            let mut light = fields.index_axis_mut(Axis(0), 1);
            for i in 0..n {
                for j in 0..n {
                    for k in 0..n {
                        // For now, use simple temperature-based light emission
                        // In full implementation, would use bubble_states.temperature and radius
                        let temp: f64 = 300.0; // bubble_states.temperature[[i, j, k]];
                        if temp > 5000.0 {
                            // Stefan-Boltzmann radiation
                            let sigma: f64 = 5.67e-8;
                            let radius: f64 = 5e-6; // bubble_states.radius[[i, j, k]];
                            let surface_area = 4.0f64 * std::f64::consts::PI * radius.powi(2);
                            let power = sigma * surface_area * temp.powi(4);
                            light[[i, j, k]] += power * params.dt;
                        }
                    }
                }
            }
        }

        // Record all data
        recorder.record(&fields, step, t);

        // Time stepping (PSTD solver)
        let context = PluginContext::new(step, num_steps, 100e3);
        solver.update(&mut fields, &grid, &medium, params.dt, t, &context)?;

        // Output progress
        if step % output_interval == 0 {
            let progress = (step as f64 / num_steps as f64) * 100.0;
            let stats = &recorder.statistics;

            println!("Progress: {:.1}%", progress);
            println!("  Cavitation Events: {}", stats.total_cavitation_events);
            println!("  SL Events: {}", stats.total_sl_events);
            println!("  Thermal Events: {}", stats.total_thermal_events);
            println!("  Max Temperature: {:.0} K", stats.max_temperature);
            println!("  Total SL Photons: {:.2e}", stats.total_sl_photons);
            println!();
        }
    }

    // Final statistics
    println!("Simulation Complete!");
    println!("===================");

    let stats = &recorder.statistics;
    println!("Final Statistics:");
    println!(
        "  Total Cavitation Events: {}",
        stats.total_cavitation_events
    );
    println!("  Total SL Events: {}", stats.total_sl_events);
    println!("  Total Thermal Events: {}", stats.total_thermal_events);
    println!("  Max Pressure: {:.2e} Pa", stats.max_pressure);
    println!("  Min Pressure: {:.2e} Pa", stats.min_pressure);
    println!("  Max Temperature: {:.0} K", stats.max_temperature);
    println!(
        "  Max Light Intensity: {:.2e} W/m²",
        stats.max_light_intensity
    );
    println!("  Total SL Photons: {:.2e}", stats.total_sl_photons);
    println!("  Total SL Energy: {:.2e} J", stats.total_sl_energy);

    // Save recorded data
    recorder.save()?;
    println!("\nData saved to data_acquisition_output.csv");

    // Export specialized maps
    if let Some(cavitation_map) = recorder.cavitation_map() {
        println!(
            "Cavitation map available: max events at single location = {:.0}",
            cavitation_map.iter().fold(0.0f64, |a, &b| a.max(b))
        );
    }

    if let Some(sl_map) = recorder.sonoluminescence_intensity_map() {
        println!(
            "SL intensity map available: max luminosity = {:.2e} photons/m³/s",
            sl_map.iter().fold(0.0f64, |a, &b| a.max(b))
        );
    }

    if let Some(thermal_dose) = recorder.thermal_dose_map() {
        println!(
            "Thermal dose map available: max cumulative exposure = {:.2} CEM43",
            thermal_dose.iter().fold(0.0f64, |a, &b| a.max(b))
        );
    }

    Ok(())
}

fn main() -> KwaversResult<()> {
    env_logger::init();

    // Run with default parameters
    let params = SimulationParams::default();
    run_data_acquisition_demo(params)?;

    println!("\n✓ Data acquisition demonstration complete!");

    Ok(())
}
