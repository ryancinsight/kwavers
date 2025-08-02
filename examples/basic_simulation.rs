//! Basic simulation example demonstrating current Kwavers API
//! 
//! This example shows the simplest way to set up and run a simulation.

use kwavers::{
    KwaversResult,
    Grid,
    HomogeneousMedium,
    Solver,
    TimeParameters,
    SineWave,
    Sensor,
    Recorder,
    PMLBoundary,
    PMLConfig,
};
use std::time::Instant;

fn main() -> KwaversResult<()> {
    println!("=== Basic Kwavers Simulation ===\n");
    
    // 1. Create computational grid
    let grid = Grid::new(
        128, 128, 128,  // Grid points (nx, ny, nz)
        0.5e-3, 0.5e-3, 0.5e-3  // Grid spacing in meters
    );
    
    println!("Grid created: {}x{}x{} points", grid.nx, grid.ny, grid.nz);
    
    // 2. Create medium (water)
    let medium = HomogeneousMedium::new(
        1000.0,  // Density (kg/m³)
        1500.0,  // Sound speed (m/s)
        &grid,
        0.0,     // Nonlinearity coefficient
        0.0,     // Attenuation coefficient
    );
    
    println!("Medium: water (density=1000 kg/m³, c=1500 m/s)");
    
    // 3. Create time parameters
    let time = TimeParameters::from_cfl(0.3, &grid, &medium);
    let num_steps = 1000;
    
    println!("Time step: {:.2e} s, Total steps: {}", time.dt, num_steps);
    
    // 4. Create PML boundary conditions
    let pml_config = PMLConfig {
        thickness: 10,
        max_damping: 200.0,
        power: 2.0,
    };
    let boundary = PMLBoundary::new(pml_config, &grid);
    
    // 5. Create solver
    let mut solver = Solver::new(grid.clone(), time.clone());
    
    // 6. Create source - sine wave at center
    let source_signal = SineWave::new(1e6, 1e5); // 1 MHz, 100 kPa amplitude
    let source_position = (grid.nx / 2, grid.ny / 2, grid.nz / 2);
    
    // 7. Create sensor for recording
    let sensor_positions = vec![
        (64.0e-3, 64.0e-3, 64.0e-3),  // Center position in meters
    ];
    let sensor = Sensor::new(&grid, &time, &sensor_positions);
    
    // 8. Create recorder
    let mut recorder = Recorder::new(
        sensor,
        &time,
        "output/basic_simulation",
        true,  // Record pressure
        false, // Don't record light
        10,    // Record every 10 steps
    );
    
    // 9. Run simulation
    println!("\nRunning simulation...");
    let start = Instant::now();
    
    for step in 0..num_steps {
        // Apply source
        let source_value = source_signal.evaluate(step as f64 * time.dt);
        solver.fields.fields[[0, source_position.0, source_position.1, source_position.2]] = source_value;
        
        // Update fields
        solver.update_fields(&medium, &boundary)?;
        
        // Record data
        recorder.record(&solver.fields, &grid, step)?;
        
        // Progress report
        if step % 100 == 0 {
            println!("Step {}/{}", step, num_steps);
        }
    }
    
    let elapsed = start.elapsed();
    println!("\nSimulation completed in {:.2?}", elapsed);
    
    // 10. Save results
    recorder.save()?;
    println!("Results saved to output/basic_simulation");
    
    Ok(())
}