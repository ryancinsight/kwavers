use kwavers::{
    boundary::PMLBoundary,
    config::Config,
    grid::Grid,
    init_logging,
    medium::heterogeneous::HeterogeneousMedium,
    physics::mechanics::acoustic_wave::NonlinearWave,
    save_pressure_data, generate_summary,
    source::{LinearArray, Source, HanningApodization},
    solver::Solver,
    time::Time,
};
use ndarray::Array3;
use log::{info, debug};
use std::sync::Arc;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    init_logging()?;
    
    // Domain size and grid parameters
    let domain_size_x = 0.05f64;    // 5 cm
    let domain_size_y = 0.03f64;    // 3 cm
    let domain_size_z = 0.03f64;    // 3 cm
    let dx = 0.0002f64;             // 0.2 mm grid spacing for higher resolution
    let dy = dx;
    let dz = dx;
    
    // Create grid
    let nx = (domain_size_x / dx).round() as usize;
    let ny = (domain_size_y / dy).round() as usize;
    let nz = (domain_size_z / dz).round() as usize;
    let grid = Grid::new(nx, ny, nz, dx, dy, dz);
    
    info!("Created grid with dimensions: {}x{}x{}", nx, ny, nz);
    
    // Create heterogeneous medium with an acoustic barrier
    let mut medium = HeterogeneousMedium::new(&grid);
    
    // Define material properties
    let water_density = 1000.0;      // Water density (kg/m³)
    let water_speed = 1500.0;        // Water sound speed (m/s)
    let barrier_density = 7800.0;    // Steel density (kg/m³)
    let barrier_speed = 5900.0;      // Steel sound speed (m/s)
    
    // Add a thin barrier in the middle of the domain
    let barrier_x = nx / 2;
    let barrier_thickness = 5;        // 5 grid points thick
    let aperture_size = ny / 10;     // Size of the aperture
    let aperture_center = ny / 2;
    
    let mut density = Array3::<f64>::ones((nx, ny, nz)) * water_density;
    let mut sound_speed = Array3::<f64>::ones((nx, ny, nz)) * water_speed;
    
    // Create a barrier with a small aperture
    for i in barrier_x..(barrier_x + barrier_thickness) {
        for j in 0..ny {
            for k in 0..nz {
                // Create an aperture in the middle
                if j < (aperture_center - aperture_size) || j > (aperture_center + aperture_size) {
                    density[[i, j, k]] = barrier_density;
                    sound_speed[[i, j, k]] = barrier_speed;
                }
            }
        }
    }
    
    // Configure medium properties
    medium.set_density(density);
    medium.set_sound_speed(sound_speed);
    
    // Set absorption and thermal properties
    let absorption_coeff = 0.1;      // Absorption coefficient (dB/MHz/cm)
    let specific_heat = 4186.0;      // Specific heat of water (J/kg/K)
    let thermal_cond = 0.6;          // Thermal conductivity of water (W/m/K)
    
    medium.set_absorption_coefficient(absorption_coeff);
    medium.set_specific_heat(specific_heat);
    medium.set_thermal_conductivity(thermal_cond);
    
    // Source parameters
    let frequency = 1.0e6f64;       // 1 MHz
    let amplitude = 1.0e5f64;       // 0.1 MPa
    let source_position = (0.01f64, domain_size_y/2.0, domain_size_z/2.0);
    
    // Create planar transducer
    let signal = Box::new(kwavers::signal::sine_wave::SineWave::new(frequency, amplitude, 0.0));
    let source = LinearArray::new(
        0.02,  // 2 cm array length
        32,    // number of elements
        source_position.1,
        source_position.2,
        signal,
        Arc::new(medium),
        &grid,
        frequency,
        HanningApodization,
    );
    
    // Configure solver with PML boundary conditions
    let pml = PMLBoundary::new(10, 2.0, &grid);  // 10 point PML
    let mut solver = Solver::new(&grid, Box::new(pml));
    
    // Time stepping parameters
    let cfl = 0.2f64;  // Courant-Friedrichs-Lewy number
    let dt = cfl * dx / 3000.0f64;  // Time step based on fastest sound speed
    let simulation_duration = 20.0e-6f64;  // 20 microseconds
    let num_time_steps = (simulation_duration / dt).round() as usize;
    let time = Time::new(dt, num_time_steps);
    
    info!("Starting diffraction simulation for {} time steps", num_time_steps);
    let start_time = Instant::now();
    
    // Run simulation
    solver.simulate(&time, source.as_ref(), Some(&[
        Box::new(save_pressure_data),
    ]))?;
    
    info!("Simulation completed in {:?}", start_time.elapsed());
    generate_summary("diffraction_simulation")?;
    
    Ok(())
}
