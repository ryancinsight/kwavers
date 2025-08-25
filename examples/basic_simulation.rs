//! Basic simulation example demonstrating current Kwavers API
//!
//! This example shows the simplest way to set up and run a simulation.

use kwavers::{
    medium::{core::CoreMedium, Medium},
    Grid, HomogeneousMedium, KwaversResult, Time,
};
use std::time::Instant;

fn main() -> KwaversResult<()> {
    println!("=== Basic Kwavers Simulation ===\n");

    // 1. Create computational grid
    let grid = Grid::new(
        64, 64, 64, // Grid points (nx, ny, nz)
        1e-3, 1e-3, 1e-3, // Grid spacing in meters
    );

    println!("Grid created: {}x{}x{} points", grid.nx, grid.ny, grid.nz);
    println!(
        "Domain size: {:.1}x{:.1}x{:.1} mm",
        grid.nx as f64 * grid.dx * 1000.0,
        grid.ny as f64 * grid.dy * 1000.0,
        grid.nz as f64 * grid.dz * 1000.0
    );

    // 2. Create medium (water)
    let medium = HomogeneousMedium::new(
        1000.0, // Density (kg/m³)
        1500.0, // Sound speed (m/s)
        0.0,    // Optical absorption
        0.0,    // Optical scattering
        &grid,
    );

    println!("Medium: water (density=1000 kg/m³, c=1500 m/s)");

    // 3. Create time parameters
    let dt = grid.cfl_timestep_default(1500.0); // CFL-based time step
    let num_steps = 100;
    let time = Time::new(dt, num_steps);

    println!("Time step: {:.2e} s", time.dt);
    println!("Total steps: {}", num_steps);
    println!("Simulation duration: {:.2} ms", time.t_max * 1000.0);

    // 4. Run a simple test
    println!("\nRunning basic test...");
    let start = Instant::now();

    // Just demonstrate the grid and time stepping
    for step in 0..10 {
        let current_time = step as f64 * dt;
        println!("Step {}: t = {:.3} ms", step, current_time * 1000.0);
    }

    let elapsed = start.elapsed();
    println!("\nTest completed in {:.2?}", elapsed);

    // 5. Show some grid properties
    println!("\nGrid properties:");
    println!("  CFL timestep: {:.2e} s", dt);
    println!("  Grid points: {}", grid.nx * grid.ny * grid.nz);
    println!(
        "  Memory estimate: {:.1} MB",
        grid.nx as f64 * grid.ny as f64 * grid.nz as f64 * 8.0 * 10.0 / 1e6
    );

    // 6. Test medium properties at center
    let center_x = grid.nx as f64 / 2.0 * grid.dx;
    let center_y = grid.ny as f64 / 2.0 * grid.dy;
    let center_z = grid.nz as f64 / 2.0 * grid.dz;

    let density = medium.density(center_x, center_y, center_z, &grid);
    let sound_speed = medium.sound_speed(center_x, center_y, center_z, &grid);

    println!("\nMedium properties at center:");
    println!(
        "  Position: ({:.1}, {:.1}, {:.1}) mm",
        center_x * 1000.0,
        center_y * 1000.0,
        center_z * 1000.0
    );
    println!("  Density: {} kg/m³", density);
    println!("  Sound speed: {} m/s", sound_speed);

    Ok(())
}
