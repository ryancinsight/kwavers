//! Basic simulation example demonstrating current Kwavers API
//!
//! This example shows the simplest way to set up and run a simulation.

use kwavers_core::error::KwaversResult;
use kwavers_core::time::Time;
use kwavers_grid::Grid;
use kwavers_medium::CoreMedium;
use kwavers_medium::HomogeneousMedium;
use std::io::Write;
use std::time::Instant;

fn main() -> KwaversResult<()> {
    let _ = writeln!(
        std::io::stdout().lock(),
        "=== Basic Kwavers Simulation ===\n"
    );

    // 1. Create computational grid
    let grid = Grid::new(
        64, 64, 64, // Grid points (nx, ny, nz)
        1e-3, 1e-3, 1e-3, // Grid spacing in meters
    )?;

    let _ = writeln!(
        std::io::stdout().lock(),
        "Grid created: {}x{}x{} points",
        grid.nx,
        grid.ny,
        grid.nz
    );
    let _ = writeln!(
        std::io::stdout().lock(),
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

    let _ = writeln!(
        std::io::stdout().lock(),
        "Medium: water (density=1000 kg/m³, c=1500 m/s)"
    );

    // 3. Create time parameters
    let dt = grid.cfl_timestep(1500.0); // CFL-based time step
    let num_steps = 100;
    let time = Time::new(dt, num_steps);

    let _ = writeln!(std::io::stdout().lock(), "Time step: {:.2e} s", time.dt);
    let _ = writeln!(std::io::stdout().lock(), "Total steps: {}", num_steps);
    let _ = writeln!(
        std::io::stdout().lock(),
        "Simulation duration: {:.2} ms",
        time.t_max * 1000.0
    );

    // 4. Run a simple test
    let _ = writeln!(std::io::stdout().lock(), "\nRunning basic test...");
    let start = Instant::now();

    // Just demonstrate the grid and time stepping
    for step in 0..10 {
        let current_time = step as f64 * dt;
        let _ = writeln!(
            std::io::stdout().lock(),
            "Step {}: t = {:.3} ms",
            step,
            current_time * 1000.0
        );
    }

    let elapsed = start.elapsed();
    let _ = writeln!(
        std::io::stdout().lock(),
        "\nTest completed in {:.2?}",
        elapsed
    );

    // 5. Show some grid properties
    let _ = writeln!(std::io::stdout().lock(), "\nGrid properties:");
    let _ = writeln!(std::io::stdout().lock(), "  CFL timestep: {:.2e} s", dt);
    let _ = writeln!(
        std::io::stdout().lock(),
        "  Grid points: {}",
        grid.nx * grid.ny * grid.nz
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "  Memory estimate: {:.1} MB",
        grid.nx as f64 * grid.ny as f64 * grid.nz as f64 * 8.0 * 10.0 / 1e6
    );

    // 6. Test medium properties at center
    let center_i = grid.nx / 2;
    let center_j = grid.ny / 2;
    let center_k = grid.nz / 2;

    let density = medium.density(center_i, center_j, center_k);
    let sound_speed = medium.sound_speed(center_i, center_j, center_k);

    let _ = writeln!(std::io::stdout().lock(), "\nMedium properties at center:");
    let _ = writeln!(
        std::io::stdout().lock(),
        "  Position: ({:.1}, {:.1}, {:.1}) mm",
        center_i as f64 * grid.dx * 1000.0,
        center_j as f64 * grid.dy * 1000.0,
        center_k as f64 * grid.dz * 1000.0
    );
    let _ = writeln!(std::io::stdout().lock(), "  Density: {} kg/m³", density);
    let _ = writeln!(
        std::io::stdout().lock(),
        "  Sound speed: {} m/s",
        sound_speed
    );

    Ok(())
}
