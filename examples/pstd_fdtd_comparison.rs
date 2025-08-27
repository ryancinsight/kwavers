//! PSTD vs FDTD Comparison Example
//!
//! This example compares pseudo-spectral (PSTD) and finite-difference (FDTD) methods.
//! Note: Full comparison functionality temporarily simplified due to API changes.

use kwavers::{
    boundary::pml::{PMLBoundary, PMLConfig},
    error::KwaversResult,
    grid::Grid,
    medium::HomogeneousMedium,
    physics::plugin::acoustic_wave_plugin::AcousticWavePlugin,
    solver::plugin_based::PluginBasedSolver,
    source::NullSource,
    time::Time,
};
use std::sync::Arc;

fn main() -> KwaversResult<()> {
    println!("=== PSTD vs FDTD Comparison ===\n");
    println!("Note: This is a simplified demonstration.\n");

    // Create computational grid
    let nx = 128;
    let dx = 1e-3;
    let grid = Grid::new(nx, nx, nx, dx, dx, dx);

    // Create medium
    let medium = Arc::new(HomogeneousMedium::water(&grid));
    let sound_speed = 1500.0; // Water sound speed

    // Time configuration
    let dt = grid.cfl_timestep(sound_speed); // Conservative CFL
    let time = Time::new(dt, 100);

    // Boundary conditions
    let boundary = Box::new(PMLBoundary::new(PMLConfig::default())?);

    println!("Grid Configuration:");
    println!("  Size: {}x{}x{}", nx, nx, nx);
    println!("  Spacing: {:.2} mm", dx * 1000.0);
    println!("  Time step: {:.2} ns", dt * 1e9);

    // Create FDTD-like solver (using plugin system)
    println!("\n1. FDTD-like Solver:");
    println!("   Using finite-difference approximations");

    // Source (null for this demo)
    let source = Box::new(NullSource::new());

    let mut fdtd_solver = PluginBasedSolver::new(
        grid.clone(),
        time.clone(),
        medium.clone(),
        boundary.clone(),
        source,
    );

    // Register acoustic plugin (uses finite differences internally)
    let fdtd_plugin = Box::new(AcousticWavePlugin::new(0.5));
    fdtd_solver.add_plugin(fdtd_plugin)?;
    fdtd_solver.initialize()?;

    println!("   ✓ FDTD solver initialized");

    // Create PSTD-like solver
    println!("\n2. PSTD-like Solver:");
    println!("   Would use spectral methods (k-space)");
    println!("   Note: Full PSTD implementation pending API updates");

    // In a full implementation, we would:
    // - Use spectral derivatives via FFT
    // - Apply k-space corrections
    // - Compare accuracy and dispersion

    println!("\nKey Differences:");
    println!("  FDTD:");
    println!("    - 2nd/4th order spatial accuracy");
    println!("    - Numerical dispersion increases with frequency");
    println!("    - Local operations (good parallelization)");
    println!("    - CFL condition limits time step");

    println!("\n  PSTD:");
    println!("    - Spectral accuracy in space");
    println!("    - Minimal numerical dispersion");
    println!("    - Global operations (FFT required)");
    println!("    - Less restrictive stability condition");

    // Run a few steps to demonstrate
    println!("\nRunning demonstration (FDTD):");
    for step in 0..10 {
        fdtd_solver.step()?;
        if step % 5 == 0 {
            println!("  Step {}: t = {:.2} μs", step, step as f64 * dt * 1e6);
        }
    }

    println!("\n✅ Comparison demonstration completed!");
    println!("\nFor full PSTD vs FDTD comparison with accuracy metrics,");
    println!("see the integration tests and documentation.");

    Ok(())
}
