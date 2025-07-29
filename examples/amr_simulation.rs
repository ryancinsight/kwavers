//! Example demonstrating Adaptive Mesh Refinement (AMR) in Kwavers
//! 
//! This example shows how to enable and configure AMR for efficient
//! simulation of focused ultrasound with adaptive resolution.

use kwavers::{
    KwaversResult, SimulationBuilder,
    solver::amr::{AMRConfig, WaveletType, InterpolationScheme},
};
use std::time::Instant;

fn main() -> KwaversResult<()> {
    // Initialize logging
    env_logger::init();
    
    println!("=== Kwavers AMR Simulation Example ===\n");
    
    // Create simulation with focused transducer
    let mut builder = SimulationBuilder::new();
    
    // Configure grid - start with moderate resolution
    builder
        .grid_size(128, 128, 128)
        .grid_spacing(0.5e-3, 0.5e-3, 0.5e-3) // 0.5mm spacing
        .time_steps(1000)
        .cfl_number(0.3);
    
    // Configure medium
    builder
        .homogeneous_medium(1000.0, 1500.0) // Water: density=1000 kg/m³, speed=1500 m/s
        .attenuation(0.0022, 1.05); // Typical water attenuation
    
    // Configure focused transducer source
    builder
        .focused_transducer_source()
        .frequency(1e6) // 1 MHz
        .focal_length(50e-3) // 50mm focal length
        .aperture(30e-3) // 30mm aperture
        .pressure(1e6); // 1 MPa source pressure
    
    // Configure physics models
    builder
        .enable_nonlinear_acoustics()
        .enable_thermal_effects()
        .enable_cavitation();
    
    // Configure recording
    builder
        .record_pressure()
        .record_temperature()
        .recording_interval(10);
    
    // Build simulation
    let (mut solver, mut recorder) = builder.build()?;
    
    println!("Grid size: {}x{}x{}", 
             solver.grid.nx, solver.grid.ny, solver.grid.nz);
    println!("Time steps: {}", solver.time.n_steps);
    println!("CFL number: {:.3}", solver.time.cfl);
    
    // Configure and enable AMR
    let amr_config = AMRConfig {
        max_level: 4,                    // Up to 4 refinement levels
        min_level: 0,                    // No coarsening below base grid
        refine_threshold: 1e-4,          // Refine when error > 0.0001
        coarsen_threshold: 1e-5,         // Coarsen when error < 0.00001
        refinement_ratio: 2,             // Standard 2:1 refinement
        buffer_cells: 3,                 // 3-cell buffer around refined regions
        wavelet_type: WaveletType::Daubechies4,
        interpolation_scheme: InterpolationScheme::Conservative,
    };
    
    // Enable AMR with adaptation every 20 steps
    solver.enable_amr(amr_config, 20)?;
    
    println!("\nAMR Configuration:");
    println!("  Max refinement level: {}", amr_config.max_level);
    println!("  Refine threshold: {:.2e}", amr_config.refine_threshold);
    println!("  Coarsen threshold: {:.2e}", amr_config.coarsen_threshold);
    println!("  Adaptation interval: 20 steps");
    
    // Run simulation
    println!("\nRunning simulation with AMR...\n");
    let start_time = Instant::now();
    
    solver.run(&mut recorder, 1e6)?;
    
    let elapsed = start_time.elapsed();
    println!("\nSimulation completed in {:.2} seconds", elapsed.as_secs_f64());
    
    // Report final AMR statistics
    if let Some(ref amr_manager) = solver.amr_manager {
        let stats = amr_manager.memory_stats();
        println!("\nFinal AMR Statistics:");
        println!("  Total cells: {}", stats.total_cells);
        println!("  Active cells: {}", stats.active_cells);
        println!("  Memory saved: {:.1}%", stats.memory_saved_percent);
        println!("  Compression ratio: {:.2}x", stats.compression_ratio);
    }
    
    // Calculate average timing
    let avg_step_time = solver.step_times.iter().sum::<f64>() / solver.step_times.len() as f64;
    let avg_amr_time = solver.physics_times[8].iter().sum::<f64>() / solver.physics_times[8].len().max(1) as f64;
    
    println!("\nTiming Statistics:");
    println!("  Average step time: {:.4} s", avg_step_time);
    println!("  Average AMR time: {:.4} s ({:.1}% of step time)", 
             avg_amr_time, avg_amr_time / avg_step_time * 100.0);
    
    // Save results
    println!("\nSaving results...");
    recorder.save_to_file("amr_simulation_results.h5")?;
    
    println!("\nResults saved to amr_simulation_results.h5");
    println!("\nSimulation complete!");
    
    Ok(())
}

/// Alternative: Manual AMR configuration for existing simulation
#[allow(dead_code)]
fn configure_amr_manually() -> KwaversResult<()> {
    use kwavers::{
        Grid, Time, HomogeneousMedium, PointSource, NullBoundary,
        AcousticWaveModel, NullCavitation, NullLight, NullThermal,
        NullChemical, NullStreaming, NullScattering, NullHeterogeneity,
        Solver, Recorder,
    };
    use std::sync::Arc;
    
    // Create components manually
    let grid = Grid::new(256, 256, 256, 0.25e-3, 0.25e-3, 0.25e-3);
    let time = Time::from_grid(&grid, 0.3, 2000);
    let medium = Arc::new(HomogeneousMedium::new(1000.0, 1500.0, &grid, 0.0, 0.0));
    
    // Create solver
    let mut solver = Solver::new(
        grid.clone(),
        time,
        medium,
        Box::new(PointSource::new(128, 128, 50, 1e6, 1e6)),
        Box::new(NullBoundary),
        Box::new(AcousticWaveModel::new(&grid, 1e6)?),
        Box::new(NullCavitation),
        Box::new(NullLight),
        Box::new(NullThermal),
        Box::new(NullChemical),
        Box::new(NullStreaming),
        Box::new(NullScattering),
        Box::new(NullHeterogeneity),
        13, // Number of fields
    );
    
    // Configure AMR for high-resolution focusing
    let amr_config = AMRConfig {
        max_level: 5,                    // Very high refinement
        min_level: 0,
        refine_threshold: 5e-5,          // Very tight threshold
        coarsen_threshold: 1e-5,
        refinement_ratio: 2,
        buffer_cells: 4,                 // Larger buffer for wave propagation
        wavelet_type: WaveletType::Daubechies6, // Higher-order wavelet
        interpolation_scheme: InterpolationScheme::WENO5, // High-order interpolation
    };
    
    // Enable AMR with frequent adaptation for rapidly changing fields
    solver.enable_amr(amr_config, 5)?; // Adapt every 5 steps
    
    // Create recorder
    let mut recorder = Recorder::new(grid.clone());
    recorder.add_field("pressure", 0);
    
    // Run simulation
    solver.run(&mut recorder, 1e6)?;
    
    Ok(())
}