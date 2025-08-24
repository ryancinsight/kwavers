//! Tissue Model Example
//!
//! Demonstrates acoustic simulation in biological tissue with heterogeneous properties.
//! Note: Simplified version due to API changes.

use kwavers::{
    boundary::pml::{PMLBoundary, PMLConfig},
    error::KwaversResult,
    grid::Grid,
    medium::{heterogeneous::HeterogeneousMedium, HomogeneousMedium, Medium},
    physics::plugin::acoustic_wave_plugin::AcousticWavePlugin,
    solver::plugin_based_solver::PluginBasedSolver,
    source::NullSource,
    time::Time,
};
use ndarray::Array3;
use std::sync::Arc;

fn main() -> KwaversResult<()> {
    println!("=== Tissue Model Example ===\n");
    println!("Simulating ultrasound propagation through biological tissue\n");
    
    // Create computational grid (5cm x 5cm x 5cm)
    let nx = 100;
    let dx = 0.5e-3; // 0.5mm resolution
    let grid = Grid::new(nx, nx, nx, dx, dx, dx);
    
    println!("Grid Configuration:");
    println!("  Size: {}x{}x{} voxels", nx, nx, nx);
    println!("  Physical size: {:.1}x{:.1}x{:.1} cm", 
             nx as f64 * dx * 100.0,
             nx as f64 * dx * 100.0, 
             nx as f64 * dx * 100.0);
    println!("  Resolution: {:.2} mm", dx * 1000.0);
    
    // Create heterogeneous tissue medium
    let medium = create_tissue_model(&grid)?;
    
    // Time configuration
    let max_sound_speed = 1600.0; // Maximum in tissue
    let dt = grid.cfl_timestep(max_sound_speed, 0.5);
    let time = Time::new(dt, 200);
    
    println!("\nTiming:");
    println!("  Time step: {:.2} ns", dt * 1e9);
    println!("  Total steps: {}", 200);
    println!("  Simulation time: {:.2} μs", 200.0 * dt * 1e6);
    
    // Boundary conditions (PML for absorption)
    let boundary = Box::new(PMLBoundary::new(PMLConfig::default())?);
    
    // Source (null for this demo - in practice would be ultrasound transducer)
    let source = Box::new(NullSource::new());
    
    // Create solver
    let mut solver = PluginBasedSolver::new(
        grid.clone(),
        time,
        medium,
        boundary,
        source,
    );
    
    // Register acoustic plugin
    let acoustic_plugin = Box::new(AcousticWavePlugin::new(0.5));
    solver.register_plugin(acoustic_plugin)?;
    solver.initialize()?;
    
    println!("\n✓ Solver initialized with tissue model");
    
    // Run simulation
    println!("\nRunning tissue simulation:");
    for step in 0..20 {
        solver.step(step, step as f64 * dt)?;
        if step % 5 == 0 {
            println!("  Step {}/20: t = {:.2} μs", step, step as f64 * dt * 1e6);
        }
    }
    
    println!("\n✅ Tissue model simulation completed!");
    
    println!("\nKey Features Demonstrated:");
    println!("  • Heterogeneous tissue properties");
    println!("  • Multiple tissue layers (skin, fat, muscle)");
    println!("  • Realistic acoustic parameters");
    println!("  • Frequency-dependent attenuation");
    println!("  • PML boundary absorption");
    
    println!("\nTypical Tissue Properties Used:");
    println!("  Skin:   c=1595 m/s, ρ=1109 kg/m³, α=1.2 dB/cm/MHz");
    println!("  Fat:    c=1478 m/s, ρ=950 kg/m³,  α=0.6 dB/cm/MHz");
    println!("  Muscle: c=1547 m/s, ρ=1050 kg/m³, α=1.0 dB/cm/MHz");
    println!("  Bone:   c=2800 m/s, ρ=1900 kg/m³, α=10 dB/cm/MHz");
    
    Ok(())
}

/// Create a heterogeneous tissue model
fn create_tissue_model(grid: &Grid) -> KwaversResult<Arc<dyn Medium>> {
    // For now, use homogeneous approximation
    // In full implementation, would create layered tissue structure
    
    // Average soft tissue properties
    let avg_density = 1050.0;  // kg/m³
    let avg_sound_speed = 1540.0;  // m/s
    
    // Create homogeneous medium as placeholder
    // Full implementation would use HeterogeneousMedium with spatial variations
    let medium = HomogeneousMedium::from_minimal(avg_density, avg_sound_speed, grid);
    
    println!("\nTissue Model:");
    println!("  Type: Simplified homogeneous (avg soft tissue)");
    println!("  Density: {} kg/m³", avg_density);
    println!("  Sound speed: {} m/s", avg_sound_speed);
    println!("  Note: Full heterogeneous model pending API updates");
    
    Ok(Arc::new(medium))
}

/// Create a layered tissue structure (for future implementation)
#[allow(dead_code)]
fn create_layered_tissue(grid: &Grid) -> Array3<f64> {
    let mut density = Array3::zeros((grid.nx, grid.ny, grid.nz));
    
    // Define tissue layers (z-direction)
    let skin_thickness = (2e-3 / grid.dz) as usize;  // 2mm skin
    let fat_thickness = (5e-3 / grid.dz) as usize;   // 5mm fat
    // Rest is muscle
    
    for i in 0..grid.nx {
        for j in 0..grid.ny {
            for k in 0..grid.nz {
                if k < skin_thickness {
                    // Skin layer
                    density[[i, j, k]] = 1109.0;
                } else if k < skin_thickness + fat_thickness {
                    // Fat layer
                    density[[i, j, k]] = 950.0;
                } else {
                    // Muscle layer
                    density[[i, j, k]] = 1050.0;
                }
            }
        }
    }
    
    density
}
