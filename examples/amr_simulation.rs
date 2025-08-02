//! Example demonstrating Adaptive Mesh Refinement (AMR) in Kwavers
//! 
//! This example shows how to use AMR for efficient simulation
//! of focused ultrasound with adaptive resolution.

use kwavers::{
    KwaversResult,
    Grid,
    HomogeneousMedium,
    medium::Medium,
    Time,
    SineWave,
    PMLBoundary,
    PMLConfig,
    solver::amr::{AMRManager, AMRConfig, WaveletType, InterpolationScheme},
};
use std::time::Instant;

fn main() -> KwaversResult<()> {
    println!("=== Kwavers AMR Simulation Example ===\n");
    
    // Create grid - start with moderate resolution
    let grid = Grid::new(64, 64, 64, 1e-3, 1e-3, 1e-3); // 1mm spacing
    
    // Create medium (water)
    let medium = HomogeneousMedium::new(1000.0, 1500.0, &grid, 0.0, 0.0);
    
    // Create time parameters
    let dt = grid.cfl_timestep_default(medium.sound_speed(0.0, 0.0, 0.0, &grid));
    let time = Time::new(dt, 1000);
    
    // Create AMR configuration
    let amr_config = AMRConfig {
        max_level: 3,                               // Up to 3 refinement levels
        min_level: 0,                               // Minimum level (coarsest)
        refine_threshold: 1e4,                      // Refinement threshold
        coarsen_threshold: 1e3,                     // Coarsening threshold
        refinement_ratio: 2,                        // 2:1 refinement
        buffer_cells: 2,                            // 2-cell buffer
        wavelet_type: WaveletType::Daubechies4,    // Wavelet for error estimation
        interpolation_scheme: InterpolationScheme::Linear,
    };
    
    // Create AMR manager
    let mut amr_manager = AMRManager::new(amr_config.clone(), &grid);
    
    // Create PML boundary
    let pml_config = PMLConfig {
        thickness: 10,
        ..Default::default()
    };
    let boundary = PMLBoundary::new(pml_config);
    
    // Create solver
    let mut solver = Solver::new(grid.clone(), time.clone());
    
    // Source configuration
    let source_signal = SineWave::new(1e6, 1e5); // 1 MHz, 100 kPa
    let source_position = (grid.nx / 2, grid.ny / 2, grid.nz / 4);
    
    println!("Initial grid: {}x{}x{}", grid.nx, grid.ny, grid.nz);
    println!("AMR Configuration:");
    println!("  Max refinement levels: {}", amr_config.max_level);
    println!("  Refinement ratio: {}", amr_config.refinement_ratio);
    println!("  Wavelet type: {:?}", amr_config.wavelet_type);
    
    // Simulation parameters
    let num_steps = 1000;
    let adapt_interval = 20; // Adapt mesh every 20 steps
    
    println!("\nRunning simulation with AMR...");
    println!("Time step: {:.2e} s", time.dt);
    println!("Total steps: {}", num_steps);
    
    let start = Instant::now();
    let mut refinement_count = 0;
    
    for step in 0..num_steps {
        // Apply source
        let source_value = source_signal.evaluate(step as f64 * time.dt);
        solver.fields.fields[[0, source_position.0, source_position.1, source_position.2]] = source_value;
        
        // Update fields
        solver.update_fields(&medium, &boundary)?;
        
        // Adapt mesh periodically
        if step % adapt_interval == 0 && step > 0 {
            // Get pressure field
            let pressure = solver.fields.fields.index_axis(ndarray::Axis(0), 0);
            
            // Estimate error and adapt mesh
            let error = amr_manager.estimate_error(&pressure);
            let max_error = error.iter().cloned().fold(0.0, f64::max);
            
            if max_error > amr_config.refine_threshold {
                amr_manager.adapt_mesh(&pressure)?;
                refinement_count += 1;
                
                println!(
                    "Step {}: Adapted mesh, max error: {:.2e}",
                    step, max_error
                );
            }
        }
        
        // Progress report
        if step % 100 == 0 {
            let pressure = solver.fields.fields.index_axis(ndarray::Axis(0), 0);
            let max_pressure = pressure.iter().cloned().map(f64::abs).fold(0.0, f64::max);
            println!("Step {}/{}: max pressure = {:.2e} Pa", step, num_steps, max_pressure);
        }
    }
    
    let elapsed = start.elapsed();
    println!("\nSimulation completed in {:.2?}", elapsed);
    println!("Total mesh adaptations: {}", refinement_count);
    
    Ok(())
}