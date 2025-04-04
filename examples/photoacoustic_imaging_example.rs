use kwavers::{
    boundary::PMLBoundary,
    config::Config,
    grid::Grid,
    init_logging,
    medium::heterogeneous::tissue::HeterogeneousTissueMedium,
    physics::{
        mechanics::acoustic_wave::NonlinearWave,
        optics::photoacoustic::PhotoacousticSource,
    },
    save_pressure_data, save_light_data, generate_summary,
    sensor::Sensor,
    solver::Solver,
    time::Time,
};
use ndarray::Array3;
use log::{info, debug};
use std::sync::Arc;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    init_logging()?;
    
    // Domain parameters
    let domain_size_x = 0.03f64;    // 3 cm
    let domain_size_y = 0.03f64;    // 3 cm
    let domain_size_z = 0.02f64;    // 2 cm
    let dx = 0.0001f64;             // 0.1 mm grid spacing for high resolution
    let dy = dx;
    let dz = dx;
    
    // Create grid
    let nx = (domain_size_x / dx).round() as usize;
    let ny = (domain_size_y / dy).round() as usize;
    let nz = (domain_size_z / dz).round() as usize;
    let grid = Grid::new(nx, ny, nz, dx, dy, dz);
    
    info!("Created grid with dimensions: {}x{}x{}", nx, ny, nz);
    
    // Create tissue medium with blood vessels and surrounding tissue
    let mut medium = HeterogeneousTissueMedium::new(&grid);
    
    // Set tissue properties
    let blood_absorption = 0.4;      // Blood absorption coefficient (mm⁻¹)
    let tissue_absorption = 0.1;     // Background tissue absorption coefficient (mm⁻¹)
    let blood_scattering = 35.0;     // Blood reduced scattering coefficient (mm⁻¹)
    let tissue_scattering = 10.0;    // Background tissue reduced scattering coefficient (mm⁻¹)
    
    // Set thermal and mechanical properties
    medium.set_specific_heat(3600.0);           // Blood specific heat (J/kg/K)
    medium.set_thermal_conductivity(0.52);      // Blood thermal conductivity (W/m/K)
    medium.set_thermal_expansion(4.0e-4);       // Blood thermal expansion coefficient (K⁻¹)
    
    // Create initial pressure distribution (simulating absorbed light)
    let mut initial_pressure = Array3::<f64>::zeros((nx, ny, nz));
    let mut absorption_map = Array3::<f64>::ones((nx, ny, nz)) * tissue_absorption;
    let mut scattering_map = Array3::<f64>::ones((nx, ny, nz)) * tissue_scattering;
    
    // Add two blood vessel-like structures
    let vessel_radius = (0.0005 / dx) as usize;  // 0.5 mm radius
    
    // Horizontal vessel
    let vessel1_x = nx / 3;
    let vessel1_y = ny / 2;
    for i in (vessel1_x-vessel_radius)..(vessel1_x+vessel_radius) {
        for j in (vessel1_y-vessel_radius)..(vessel1_y+vessel_radius) {
            for k in 0..nz {
                let dx = (i as f64 - vessel1_x as f64) * dx;
                let dy = (j as f64 - vessel1_y as f64) * dy;
                if dx*dx + dy*dy <= (vessel_radius as f64 * dx).powi(2) {
                    initial_pressure[[i, j, k]] = 1.0;  // 1 MPa initial pressure
                    absorption_map[[i, j, k]] = blood_absorption;
                    scattering_map[[i, j, k]] = blood_scattering;
                }
            }
        }
    }
    
    // Vertical vessel
    let vessel2_x = 2 * nx / 3;
    let vessel2_z = nz / 2;
    for i in (vessel2_x-vessel_radius)..(vessel2_x+vessel_radius) {
        for j in 0..ny {
            for k in (vessel2_z-vessel_radius)..(vessel2_z+vessel_radius) {
                let dx = (i as f64 - vessel2_x as f64) * dx;
                let dz = (k as f64 - vessel2_z as f64) * dz;
                if dx*dx + dz*dz <= (vessel_radius as f64 * dx).powi(2) {
                    initial_pressure[[i, j, k]] = 1.0;
                    absorption_map[[i, j, k]] = blood_absorption;
                    scattering_map[[i, j, k]] = blood_scattering;
                }
            }
        }
    }
    
    // Set optical properties in the medium
    medium.set_absorption_coefficient_light(absorption_map);
    medium.set_reduced_scattering_coefficient_light(scattering_map);
    
    // Create photoacoustic source with tissue-specific initial pressure
    let mut source = PhotoacousticSource::new_with_medium(initial_pressure, medium.clone());
    
    // Set tissue-specific Grüneisen parameters
    // Blood has higher Grüneisen parameter than surrounding tissue
    // Reference: Ref: Cox, B. T., et al. (2009). Photoacoustic tomography with a single detector in a reverberant cavity
    source.set_gruneisen_parameter(0.24);  // Higher value for blood vessels
    
    // Place linear sensor array for detection
    let num_sensors = 64;
    let sensor_positions: Vec<(f64, f64, f64)> = (0..num_sensors)
        .map(|i| {
            let y = (i as f64 / num_sensors as f64) * domain_size_y;
            (0.0, y, domain_size_z / 2.0)
        })
        .collect();
    
    let sensor = Sensor::new(&sensor_positions, &grid);
    
    // Configure solver with PML boundary conditions
    let pml = PMLBoundary::new(20, 2.0, &grid);  // 20 point PML for better absorption
    let mut solver = Solver::new(&grid, Box::new(pml));
    
    // Time stepping parameters
    let dt = 0.1 * dx / 1500.0;  // CFL number = 0.1
    let simulation_duration = 20.0e-6f64;  // 20 microseconds
    let num_time_steps = (simulation_duration / dt).round() as usize;
    let time = Time::new(dt, num_time_steps);
    
    info!("Starting photoacoustic simulation for {} time steps", num_time_steps);
    let start_time = Instant::now();
    
    // Run simulation
    solver.simulate(&time, &source, Some(&[
        Box::new(save_pressure_data),
        Box::new(save_light_data),
        Box::new(sensor),
    ]))?;
    
    info!("Simulation completed in {:?}", start_time.elapsed());
    generate_summary("photoacoustic_simulation")?;
    
    Ok(())
}
