//! Wave Simulation Example
//!
//! Demonstrates basic wave propagation simulation

use kwavers::{
    error::KwaversResult,
    grid::{Grid, stability::StabilityCalculator},
    medium::{core::CoreMedium, HomogeneousMedium},
    solver::plugin_based::PluginBasedSolver,
    source::NullSource,
};
use std::sync::Arc;

fn main() -> KwaversResult<()> {
    println!("=== Wave Simulation Example ===\n");

    // Create computational grid
    let grid = Grid::new(64, 64, 64, 1e-3, 1e-3, 1e-3)?;
    println!("Grid: {}x{}x{}", grid.nx, grid.ny, grid.nz);
    println!("Spacing: {:.1}mm", grid.dx * 1000.0);

    // Create homogeneous medium (water)
    let medium = Arc::new(HomogeneousMedium::water(&grid));

    // Access properties through the Medium trait
    let sound_speed = medium.sound_speed(0, 0, 0);
    let density = medium.density(0, 0, 0);
    println!("Medium: Water (c={} m/s, ρ={} kg/m³)", sound_speed, density);

    // Calculate time step using CFL condition
    let dt = StabilityCalculator::cfl_timestep_fdtd(&grid, sound_speed);
    let num_steps = 100;
    let total_time = dt * num_steps as f64;

    // For now, use a null source (no source)
    // In a real application, you would implement a proper source
    let source = Box::new(NullSource::new());
    println!("Source: Using null source for demonstration");

    // Create time configuration
    use kwavers::time::Time;
    let time = Time::new(dt, num_steps);

    // Create boundary (using PML for absorption)
    use kwavers::boundary::pml::{PMLBoundary, PMLConfig};
    let pml_config = PMLConfig::default();
    let boundary: Box<dyn kwavers::boundary::Boundary> = Box::new(PMLBoundary::new(pml_config)?);

    // Create solver
    let mut solver = PluginBasedSolver::new(grid.clone(), time, medium.clone(), boundary, source);

    // Register acoustic wave plugin
    use kwavers::physics::plugin::acoustic_wave_plugin::AcousticWavePlugin;
    let acoustic_plugin = Box::new(AcousticWavePlugin::new(0.95)); // CFL number
    solver.add_plugin(acoustic_plugin)?;

    println!("\nSimulation parameters:");
    println!("  Time step: {:.2} ns", dt * 1e9);
    println!("  Steps: {}", num_steps);
    println!("  Total time: {:.2} μs", total_time * 1e6);

    // Initialize the solver
    solver.initialize()?;

    // Run simulation
    println!("\nRunning simulation...");
    for step in 0..num_steps {
        solver.step()?;

        if step % 20 == 0 {
            println!("  Step {}/{}", step, num_steps);
        }
    }

    println!("\n✅ Wave simulation completed successfully!");
    println!("Note: This example uses a null source for simplicity.");
    println!("In practice, you would implement a proper acoustic source.");

    Ok(())
}
