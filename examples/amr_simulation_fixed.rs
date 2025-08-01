//! Example demonstrating Adaptive Mesh Refinement (AMR) in Kwavers
//! 
//! This example shows how to use AMR for efficient simulation
//! of focused ultrasound with adaptive resolution.

use kwavers::{
    KwaversResult,
    Grid,
    medium::homogeneous::HomogeneousMedium,
    solver::{
        Solver,
        amr::{AMRManager, AMRConfig, RefinementCriterion, InterpolationMethod},
    },
    source::PointSource,
    signal::GaussianPulse,
    boundary::pml::{PMLBoundary, PMLConfig},
};
use ndarray::Array3;
use std::time::Instant;

fn main() -> KwaversResult<()> {
    // Initialize logging
    env_logger::init();
    
    println!("=== Kwavers AMR Simulation Example ===\n");
    
    // Create grid - start with moderate resolution
    let grid = Grid::new(64, 64, 64, 1e-3, 1e-3, 1e-3); // 1mm spacing
    
    // Create medium (water)
    let medium = HomogeneousMedium::new(1000.0, 1500.0, &grid, 0.0, 0.0);
    
    // Create AMR configuration
    let amr_config = AMRConfig {
        max_levels: 3,                              // Up to 3 refinement levels
        refinement_ratio: 2,                        // 2:1 refinement
        buffer_cells: 2,                            // 2-cell buffer
        refinement_criterion: RefinementCriterion::GradientBased {
            threshold: 1e4,                         // Pressure gradient threshold
        },
        coarsening_factor: 0.1,                     // Coarsen when gradient < 0.1 * threshold
        interpolation_method: InterpolationMethod::Linear,
        min_cells_per_dimension: 4,                 // Minimum 4 cells per dimension
        max_memory_mb: 1024,                        // 1GB memory limit
    };
    
    // Create AMR manager
    let mut amr_manager = AMRManager::new(amr_config.clone(), &grid);
    
    // Create source
    let source_position = (grid.nx as f64 * grid.dx / 2.0,
                          grid.ny as f64 * grid.dy / 2.0,
                          grid.nz as f64 * grid.dz / 4.0);
    let signal = GaussianPulse::new(1e6, 1e5, 3e-6); // 1 MHz, 100 kPa, 3 μs pulse
    let source = PointSource::new(source_position, Box::new(signal));
    
    // Create boundary (PML)
    let pml_config = PMLConfig {
        size: 10,
        max_value: 2.0,
        alpha_max: 0.0,
    };
    let boundary = PMLBoundary::new(pml_config)?;
    
    // Create solver
    let mut solver = Solver::new(grid.clone(), Box::new(medium), Box::new(source), Box::new(boundary));
    
    println!("Initial grid: {}x{}x{}", grid.nx, grid.ny, grid.nz);
    println!("AMR Configuration:");
    println!("  Max refinement levels: {}", amr_config.max_levels);
    println!("  Refinement ratio: {}", amr_config.refinement_ratio);
    
    // Initialize pressure field
    let mut pressure = Array3::zeros((grid.nx, grid.ny, grid.nz));
    
    // Simulation parameters
    let dt = 5e-8; // 50 ns
    let num_steps = 1000;
    let adapt_interval = 20; // Adapt mesh every 20 steps
    
    println!("\nRunning simulation with AMR...");
    let start_time = Instant::now();
    
    for step in 0..num_steps {
        let t = step as f64 * dt;
        
        // Update fields
        solver.step(dt)?;
        
        // Get pressure field
        if let Ok(p) = solver.get_field("pressure") {
            pressure.assign(p);
        }
        
        // Adapt mesh periodically
        if step > 0 && step % adapt_interval == 0 {
            let octree = amr_manager.get_octree();
            if let Ok(stats) = amr_manager.adapt_mesh(&pressure) {
                if stats.cells_refined > 0 || stats.cells_coarsened > 0 {
                    println!("  Step {}: Refined {} cells, coarsened {} cells",
                            step, stats.cells_refined, stats.cells_coarsened);
                    println!("    Active cells: {}", octree.count_active_cells());
                }
            }
        }
        
        // Progress update
        if step % 100 == 0 {
            let max_pressure = pressure.iter().fold(0.0f64, |a, &b| a.max(b.abs()));
            println!("Step {}/{}: t = {:.2e} s, max |p| = {:.2e} Pa",
                    step, num_steps, t, max_pressure);
        }
    }
    
    let elapsed = start_time.elapsed();
    println!("\nSimulation completed in {:.2} seconds", elapsed.as_secs_f64());
    
    // Final statistics
    let octree = amr_manager.get_octree();
    println!("\nFinal AMR statistics:");
    println!("  Total active cells: {}", octree.count_active_cells());
    println!("  Memory usage: {:.2} MB", amr_manager.memory_usage_mb());
    
    // Analyze refinement efficiency
    let base_cells = grid.nx * grid.ny * grid.nz;
    let compression_ratio = base_cells as f64 / octree.count_active_cells() as f64;
    println!("  Compression ratio: {:.2}x", compression_ratio);
    
    Ok(())
}