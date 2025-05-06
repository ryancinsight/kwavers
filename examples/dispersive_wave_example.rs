use kwavers::{
    boundary::PMLBoundary,
    config::Config,
    grid::Grid,
    init_logging,
    medium::dispersion::PowerLawDispersiveMedium,
    physics::mechanics::acoustic_wave::NonlinearWave,
    save_pressure_data, generate_summary,
    source::{LinearArray, Source, HanningApodization},
    solver::Solver,
    time::Time,
};
use ndarray::Array3;
use log::info;
use std::sync::Arc;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    init_logging()?;
    
    // Create high-resolution grid for dispersive effects
    let domain_size = 0.05f64;  // 5 cm
    let dx = 0.0001f64;         // 0.1 mm spacing
    let n = (domain_size / dx).round() as usize;
    let grid = Grid::new(n, n, n, dx, dx, dx);
    
    info!("Created {}x{}x{} grid with {} mm spacing", n, n, n, dx * 1000.0);
    
    // Create dispersive medium (frequency-dependent attenuation)
    let density = Array3::<f64>::ones((n, n, n)) * 1000.0;  // Water density
    let sound_speed = Array3::<f64>::ones((n, n, n)) * 1500.0;  // Base sound speed
    
    let medium = PowerLawDispersiveMedium::new(
        density,
        sound_speed,
        0.5,    // α₀: power law coefficient
        1.1,    // y: power law exponent (typical for biological tissue)
    );
    
    let medium = Arc::new(medium);
    
    // Create broadband pulse source to observe dispersion
    let center_freq = 2.0e6;  // 2 MHz center frequency
    let bandwidth = 1.0e6;    // 1 MHz bandwidth
    let amplitude = 1.0e5;    // Initial pressure amplitude
    
    let signal = Box::new(kwavers::signal::chirp::ChirpSignal::new(
        center_freq - bandwidth/2.0,
        center_freq + bandwidth/2.0,
        5.0e-6,  // 5 μs duration
        amplitude,
    ));
    
    // Create linear array source
    let source = LinearArray::new(
        0.01,              // 1 cm array length
        32,                // elements
        domain_size/2.0,   // y position (center)
        domain_size/2.0,   // z position (center)
        signal,
        medium.clone(),
        &grid,
        center_freq,
        HanningApodization,
    );
    
    // Configure solver with PML boundary
    let pml = PMLBoundary::new(20, 2.0, &grid);  // 20 point PML
    let mut solver = Solver::new(&grid, Box::new(pml));
    
    // Configure solver for dispersive wave propagation
    let mut wave = NonlinearWave::new(&grid);
    wave.set_nonlinearity_scaling(0.0);  // Linear propagation to observe dispersion
    wave.set_k_space_correction_order(4); // High-order correction for better dispersion handling
    
    // Time stepping parameters
    let dt = 0.1 * dx / 1500.0;  // CFL = 0.1
    let simulation_duration = 20.0e-6;  // 20 microseconds
    let num_time_steps = (simulation_duration / dt).round() as usize;
    let time = Time::new(dt, num_time_steps);
    
    info!("Starting dispersive wave simulation for {} time steps", num_time_steps);
    
    // Run simulation
    solver.simulate(&time, &source, Some(&[
        Box::new(save_pressure_data),
    ]))?;
    
    generate_summary("dispersive_wave_simulation")?;
    info!("Simulation complete");
    
    Ok(())
}
