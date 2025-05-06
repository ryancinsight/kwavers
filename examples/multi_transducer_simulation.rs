use kwavers::{
    boundary::PMLBoundary,
    config::Config,
    grid::Grid,
    init_logging,
    medium::heterogeneous::tissue::HeterogeneousTissueMedium,
    physics::mechanics::acoustic_wave::NonlinearWave,
    save_pressure_data, generate_summary,
    source::{
        FocusedTransducer, LinearArray, MatrixArray, Source,
        HanningApodization, GaussianApodization,
    },
    signal::sine_wave::SineWave,
    solver::Solver,
    time::Time,
};
use std::sync::Arc;
use log::info;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    init_logging()?;
    
    // Create simulation grid
    let domain_size = 0.05f64;  // 5 cm cubic domain
    let dx = 0.0002f64;         // 0.2 mm grid spacing
    let n = (domain_size / dx).round() as usize;
    let grid = Grid::new(n, n, n, dx, dx, dx);
    
    info!("Created {}x{}x{} grid with {} mm spacing", n, n, n, dx * 1000.0);
    
    // Create heterogeneous tissue medium
    let mut medium = HeterogeneousTissueMedium::new(&grid);
    let medium = Arc::new(medium);
    
    // Create three different transducer types
    
    // 1. Focused transducer for main beam
    let focused_signal = Box::new(SineWave::new(1.0e6, 1.0e5, 0.0));  // 1 MHz, 0.1 MPa
    let focused = Box::new(FocusedTransducer::new(
        0.01, 0.025, 0.025,     // position
        0.03, 0.025, 0.025,     // focus point
        0.01,                   // radius
        focused_signal,
        Box::new(HanningApodization),
        medium.clone(),         // Add medium reference
    )) as Box<dyn Source>;
    
    // 2. Linear array for steering
    let linear_signal = Box::new(SineWave::new(1.0e6, 0.8e5, 0.0));  // 1 MHz, 0.08 MPa
    let linear = Box::new(LinearArray::with_focus(
        0.02,              // length
        32,                // elements
        0.025,             // y position
        0.04,              // z position
        linear_signal,
        medium.as_ref(),
        &grid,
        1.0e6,            // frequency
        0.03,             // focus x
        0.025,            // focus y
        0.02,             // focus z
        HanningApodization,
    )) as Box<dyn Source>;
    
    // 3. Matrix array for wide coverage
    let matrix_signal = Box::new(SineWave::new(1.0e6, 0.5e5, 0.0));  // 1 MHz, 0.05 MPa
    let matrix = Box::new(MatrixArray::new(
        0.01,              // width
        0.01,              // height
        16,                // rows
        16,                // columns
        0.04,              // x position
        0.025,             // y position
        0.025,             // z position
        matrix_signal,
        GaussianApodization,
    )) as Box<dyn Source>;
    
    // Combine sources into a composite source
    let sources: Vec<Box<dyn Source>> = vec![focused, linear, matrix];
    
    // Configure solver with PML boundary
    let pml = PMLBoundary::new(10, 2.0, &grid);
    let mut solver = Solver::new(&grid, Box::new(pml));
    
    // Time stepping parameters
    let dt = 0.1 * dx / 1500.0;  // CFL = 0.1
    let simulation_duration = 40.0e-6f64;  // 40 microseconds
    let num_time_steps = (simulation_duration / dt).round() as usize;
    let time = Time::new(dt, num_time_steps);
    
    info!("Starting multi-transducer simulation for {} time steps", num_time_steps);
    
    // Run simulation for each source
    for (i, source) in sources.iter().enumerate() {
        info!("Simulating source {}", i + 1);
        solver.simulate(&time, source.as_ref(), Some(&[
            Box::new(save_pressure_data),
        ]))?;
    }
    
    generate_summary("multi_transducer_simulation")?;
    info!("Simulation complete");
    
    Ok(())
}
