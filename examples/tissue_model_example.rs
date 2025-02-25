// examples/tissue_model_example.rs
//! Example demonstrating the tissue-specific absorption model with a layered tissue structure.
//! This simulation models a focused ultrasound beam passing through multiple layers of tissue,
//! showing the different acoustic properties and absorption characteristics of each tissue type.

use kwavers::{
    boundary::PMLBoundary,
    config::Config,
    grid::Grid,
    init_logging,
    medium::heterogeneous::tissue::HeterogeneousTissueMedium,
    physics::mechanics::acoustic_wave::NonlinearWave,
    save_pressure_data, save_light_data, generate_summary,
    source::{LinearArray, Source, HanningApodization},
    solver::Solver,
    time::Time,
};
use ndarray::{Array3, Array4, Axis};
use log::{info, debug};
use std::sync::Arc;
use std::time::Instant;
use std::fs::create_dir_all;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logger
    init_logging()?;
    
    // Create output directory
    create_dir_all("output")?;
    
    // Tissue model parameters
    let domain_size_x = 0.1f64;    // 10 cm in x direction
    let domain_size_y = 0.06f64;   // 6 cm in y direction
    let domain_size_z = 0.06f64;   // 6 cm in z direction
    let dx = 0.0005f64;            // 0.5 mm grid spacing
    let dy = 0.0005f64;
    let dz = 0.0005f64;
    
    // Transducer parameters
    let frequency = 1.0e6f64;       // 1 MHz focused ultrasound
    let source_position = (0.01f64, 0.03f64, 0.03f64);
    let focus_position = (0.05f64, 0.03f64, 0.03f64);  // Focus in center of domain
    let aperture_radius = 0.015f64; // 1.5 cm aperture
    
    info!("Starting tissue model simulation with {} MHz transducer", frequency/1.0e6);
    
    // Create grid with appropriate dimensions
    let nx = (domain_size_x / dx).round() as usize;
    let ny = (domain_size_y / dy).round() as usize;
    let nz = (domain_size_z / dz).round() as usize;
    
    let grid = Grid::new(nx, ny, nz, dx, dy, dz);
    info!("Created grid with dimensions: {}x{}x{}", nx, ny, nz);
    
    // Create heterogeneous tissue medium with predefined layers
    let mut medium = HeterogeneousTissueMedium::new_layered(&grid);
    info!("Created layered tissue medium model");
    
    // Create a linear array transducer instead of FocusedTransducer
    let amplitude = 1.0e5f64; // 0.1 MPa amplitude
    let mut signal = kwavers::SineWave::new(frequency, amplitude, 0.0);
    
    let num_elements = 16;
    let source = LinearArray::with_focus(
        aperture_radius * 2.0, // length
        num_elements,
        source_position.1,
        source_position.2,
        Box::new(signal),
        &medium,
        &grid,
        frequency,
        focus_position.0,
        focus_position.1,
        focus_position.2,
        HanningApodization,
    );
    
    info!("Created linear array at position ({:.3}, {:.3}, {:.3}) m focusing at ({:.3}, {:.3}, {:.3}) m", 
          source_position.0, source_position.1, source_position.2,
          focus_position.0, focus_position.1, focus_position.2);
    
    // Create nonlinear wave solver with enhanced physics
    let mut nonlinear_wave = NonlinearWave::new(&grid);
    nonlinear_wave.set_nonlinearity_scaling(2.0); // Enhance nonlinearity
    nonlinear_wave.set_k_space_correction_order(3); // Higher-order correction
    info!("Configured nonlinear wave solver with enhanced physics");
    
    // Configure solver
    let dt = 0.2f64 * dx / 1600.0f64; // CFL condition for numerical stability
    
    // Configure simulation time
    let num_cycles = 5.0f64;
    let simulation_time = num_cycles / frequency;
    let num_steps = (simulation_time / dt).round() as usize;
    
    let time = Time::new(dt, num_steps);
    let medium_arc = Arc::new(medium);
    
    // Configure PML boundary conditions
    let pml_thickness = 10;
    let boundary = PMLBoundary::new(
        pml_thickness, 
        100.0,             // sigma_max_acoustic
        10.0,              // sigma_max_light
        medium_arc.as_ref(),
        &grid,
        frequency,
        Some(2),           // polynomial_order
        Some(0.000001)     // target_reflection
    );
    
    // Create sensor array and recorder
    let sensor_positions = vec![
        (focus_position.0, focus_position.1, focus_position.2),               // At focus
        (focus_position.0 + 0.01, focus_position.1, focus_position.2),        // 1cm beyond focus
        (focus_position.0 - 0.01, focus_position.1, focus_position.2),        // 1cm before focus
    ];
    
    let sensor = kwavers::Sensor::new(&grid, &time, &sensor_positions);
    let mut recorder = kwavers::Recorder::new(
        sensor,
        &time,
        "output/tissue_model",
        true,
        true,
        20 // snapshot interval
    );
    
    // Create solver
    let mut solver = Solver::new(
        grid.clone(),
        time,
        medium_arc.clone(),
        Box::new(source),
        Box::new(boundary)
    );
    
    // Run simulation
    let start_time = Instant::now();
    info!("Starting simulation with {} time steps", num_steps);
    
    solver.run(&mut recorder, frequency);
    
    let elapsed = start_time.elapsed();
    info!("Simulation completed in {:.2?}", elapsed);
    
    // Save results
    save_pressure_data(&recorder, &solver.time, "output/tissue_pressure.csv")?;
    save_light_data(&recorder, &solver.time, "output/tissue_light.csv")?;
    generate_summary(&recorder, "output/tissue_summary.csv")?;
    info!("Results saved to output directory");
    
    Ok(())
} 