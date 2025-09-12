use kwavers::{
    boundary::{PMLBoundary, PMLConfig},
    configuration::Configuration,
    grid::Grid,
    init_logging,
    medium::heterogeneous::tissue::{HeterogeneousTissueMedium, TissueType},
    physics::mechanics::{
        acoustic_wave::NonlinearWave,
        elastic_wave::ElasticWave,
    },
    save_pressure_data,
    source::{BowlTransducer, BowlConfig, HanningApodization},
    solver::plugin_based::Solver,
    time::Time,
};
use ndarray::{Array3, s};
use log::info;
use std::sync::Arc;
use std::io::Write;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    init_logging()?;
    
    // Create high-resolution grid for accurate shear wave tracking
    let domain_size = 0.04f64;  // 4 cm cubic domain
    let dx = 0.0002f64;         // 0.2 mm spacing
    let n = (domain_size / dx).round() as usize;
    let grid = Grid::new(n, n, n, dx, dx, dx)?;
    
    info!("Created {}x{}x{} grid with {} mm spacing", n, n, n, dx * 1000.0);
    
    // Create tissue medium with inclusions of different stiffness
    let mut medium = HeterogeneousTissueMedium::new(grid.clone()?, TissueType::Muscle);
    
    // Create shear modulus distribution with stiff inclusion
    let mut mu = Array3::<f64>::ones((n, n, n)) * 3.0e3;  // 3 kPa background (soft tissue)
    
    // Add stiffer inclusion (simulating tumor)
    let center = n/2;
    let radius_f64 = 0.005 / dx;
    let radius = if radius_f64 > usize::MAX as f64 {
        usize::MAX
    } else {
        radius_f64 as usize
    };
    
    for i in (center-radius)..(center+radius) {
        for j in (center-radius)..(center+radius) {
            for k in (center-radius)..(center+radius) {
                let dx = (i as f64 - center as f64) * dx;
                let dy = (j as f64 - center as f64) * dx;
                let dz = (k as f64 - center as f64) * dx;
                
                if dx*dx + dy*dy + dz*dz <= (radius as f64 * dx).powi(2) {
                    mu[[i, j, k]] = 15.0e3;  // 15 kPa (stiffer region)
                }
            }
        }
    }
    
    // Initialize elastic wave solver
    let mut elastic_wave = ElasticWave::new(&grid);
    elastic_wave.set_shear_modulus(mu);
    elastic_wave.initialize_from_medium(&medium, &grid);
    
    // Create acoustic radiation force pulse
    let push_freq = 2.0e6;    // 2 MHz push frequency
    let push_amplitude = 2.0e6;  // 2 MPa push amplitude
    let push_duration = 100e-6;  // 100 μs push duration
    
    let signal = Box::new(kwavers::signal::sine_wave::SineWave::new(
        push_freq,
        push_amplitude,
        0.0,
    ));
    
    // Create focused transducer for push beam
    let bowl_config = BowlConfig {
        center: [0.01, domain_size/2.0, domain_size/2.0],
        focus: [0.02, domain_size/2.0, domain_size/2.0],
        diameter: 0.02,
        radius_of_curvature: 0.01,
        frequency: push_freq,
        amplitude: push_amplitude,
        phase: 0.0,
        ..Default::default()
    };
    let push_source = BowlTransducer::new(bowl_config)?;
    
    // Configure acoustic solver for push beam
    let pml_config = PMLConfig {
        thickness: 10,
        sigma_max_acoustic: 2.0,
        ..Default::default()
    };
    let pml = PMLBoundary::new(pml_config)?;
    let mut solver = Solver::new(&grid, Box::new(pml));
    
    // Time parameters for push beam
    let dt_acoustic = 0.1 * dx / 1500.0;  // CFL = 0.1 for acoustics
    let num_push_steps = (push_duration / dt_acoustic).round() as usize;
    let push_time = Time::new(dt_acoustic, num_push_steps);
    
    info!("Simulating acoustic radiation force push...");
    
    // Simulate push beam
    solver.simulate(&push_time, &push_source, Some(&[
        Box::new(save_pressure_data),
    ]))?;
    
    // Convert acoustic pressure to body force
    let pressure = solver.get_pressure();
    let force = pressure.mapv(|p| p.powi(2) / (1500.0 * 1000.0));  // F = αI/c, I = p²/ρc
    
    info!("Starting shear wave tracking...");
    
    // Time parameters for shear wave propagation
    let dt_elastic = 0.1 * dx / 10.0;  // CFL for shear waves (c_s ≈ 1-10 m/s)
    let tracking_duration = 10e-3;      // 10 ms tracking time
    let num_tracking_steps = (tracking_duration / dt_elastic).round() as usize;
    
    // Initialize displacement from radiation force
    let mut displacement = elastic_wave.get_displacement(0);  // x-component
    displacement += &force * dt_elastic;
    
    // Track shear wave propagation
    for step in 0..num_tracking_steps {
        elastic_wave.step(dt_elastic, &grid);
        
        if step % 100 == 0 {
            // Save displacement field every 100 steps
            let disp = elastic_wave.get_displacement(0);
            save_displacement_snapshot(&disp, step)?;
            info!("Tracking step {}/{}", step, num_tracking_steps);
        }
    }
    
    // Note: generate_summary requires a Recorder instance
    // For now, we'll skip summary generation in this example
    // generate_summary(&recorder, "elastography_simulation")?;
    info!("Simulation complete");
    
    Ok(())
}

fn save_displacement_snapshot(displacement: &Array3<f64>, step: usize) -> Result<(), Box<dyn std::error::Error>> {
    let filename = format!("displacement_step_{}.csv", step);
    let mut file = std::fs::File::create(&filename)?;
    
    for ((i, j, k), &value) in displacement.indexed_iter() {
        if let Err(e) = writeln!(file, "{},{},{},{}", i, j, k, value) {
            eprintln!("Error writing to file: {}", e);
            return Err(e.into());
        }
    }
    
    Ok(())
}