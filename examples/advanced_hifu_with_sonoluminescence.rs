// examples/advanced_hifu_with_sonoluminescence.rs
use kwavers::{
    init_logging, plot_simulation_outputs, Config, HanningApodization, HomogeneousMedium,
    LinearArray, PMLBoundary, Recorder, Sensor, SineWave, Solver, NonlinearWave,
};
use std::error::Error;
use std::fs::File;
use std::io::Write;
use std::sync::Arc;

fn main() -> Result<(), Box<dyn Error>> {
    init_logging()?;

    // Create a scientifically optimized config file for HIFU simulations
    // Parameters are based on typical ultrasound physics for HIFU applications
    let config_content = r#"
        domain_size_x = 0.06
        domain_size_yz = 0.04
        points_per_wavelength = 6
        frequency = 1000000.0  # 1 MHz - typical HIFU frequency
        amplitude = 1.0e6      # Higher amplitude for nonlinear effects
        num_cycles = 5.0
        pml_thickness = 10
        pml_sigma_acoustic = 100.0
        pml_sigma_light = 10.0
        pml_polynomial_order = 2
        pml_reflection = 0.000001
        kspace_alpha = 0.5     # k-space correction coefficient
        
        # HIFU transducer parameters
        num_elements = 32
        signal_type = "sine"
        focus_x = 0.03
        focus_y = 0.0
        focus_z = 0.0

        pressure_file = "hifu_pressure.csv"
        light_file = "hifu_light.csv"
        summary_file = "hifu_summary.csv"
        snapshot_interval = 10
        enable_visualization = true
    "#;
    let mut file = File::create("hifu_config.toml")?;
    file.write_all(config_content.as_bytes())?;

    let config = Config::from_file("hifu_config.toml")?;
    let grid = config.grid().clone();
    let time = config.time().clone();
    
    // Create a more physically accurate water medium for HIFU
    // Configure medium properties before wrapping in Arc
    let mut medium_obj = HomogeneousMedium::new(
        998.0,   // Density (kg/m³) for water
        1482.0,  // Sound speed (m/s) for water at body temperature
        &grid, 
        0.3,     // mu_a (absorption coefficient for light)
        10.0,    // mu_s_prime (reduced scattering coefficient)
    );
    
    // Configure the medium for improved absorption modeling
    medium_obj.alpha0 = 0.3;     // Power law absorption coefficient
    medium_obj.delta = 1.1;      // Power law exponent
    medium_obj.b_a = 5.2;        // Nonlinearity parameter (B/A) for water
    
    // Wrap in Arc after configuration
    let medium = Arc::new(medium_obj);
    
    let signal = Box::new(SineWave::new(
        config.simulation.frequency,
        config.simulation.amplitude,
        0.0,
    ));
    
    // Create a focused transducer for HIFU
    let source = Box::new(LinearArray::with_focus(
        0.04, // Length (smaller)
        config.source.num_elements,
        0.0,
        0.0, // y0, z0
        signal,
        medium.as_ref(),
        &grid,
        config.simulation.frequency,
        config.source.focus_x.unwrap_or(0.03),
        config.source.focus_y.unwrap_or(0.0),
        config.source.focus_z.unwrap_or(0.0),
        HanningApodization,
    )) as Box<dyn kwavers::Source>;
    
    // Use the PML from the config with proper thickness for HIFU
    let boundary = Box::new(config.pml().clone());
    
    // Place sensors strategically at focus and surrounding areas
    let sensor_positions: Vec<(f64, f64, f64)> = vec![
        (0.03, 0.0, 0.0),   // Focal point
        (0.025, 0.0, 0.0),  // Pre-focal region
        (0.035, 0.0, 0.0),  // Post-focal region
        (0.03, 0.005, 0.0), // Off-axis near focus
    ];
    
    let sensor = Sensor::new(&grid, &time, &sensor_positions);
    let mut recorder = Recorder::new(
        sensor,
        &time,
        "hifu_sonoluminescence",
        true,
        true,
        config.output.snapshot_interval,
    );

    // Create the solver with our optimized medium
    let mut solver = Solver::new(grid.clone(), time.clone(), medium, source, boundary);
    
    // Access and configure the nonlinear wave solver for better HIFU physics
    {
        let wave = &mut solver.wave;
        // Set stronger nonlinearity scaling for HIFU (high pressure nonlinear effects)
        wave.set_nonlinearity_scaling(2.0);
        // Use 3rd order k-space correction for better dispersion handling
        wave.set_k_space_correction_order(3);
    }
    
    // Run the simulation
    solver.run(&mut recorder, config.simulation.frequency);
    recorder.save()?;
    
    // Create visualizations
    plot_simulation_outputs(&recorder, &grid, &time, solver.source.as_ref());

    Ok(())
}
