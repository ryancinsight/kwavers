// examples/advanced_hifu_with_sonoluminescence.rs
use kwavers::{
    init_logging, plot_simulation_outputs, Config, HomogeneousMedium, PMLBoundary, Recorder, Sensor, Solver, NonlinearWave, // NonlinearWave for concrete type
    physics::{ // Import physics models and traits
        mechanics::cavitation::CavitationModel,
        mechanics::streaming::StreamingModel,
        chemistry::ChemicalModel,
        optics::diffusion::LightDiffusion as LightDiffusionModel,
        scattering::acoustic::AcousticScatteringModel,
        thermodynamics::heat_transfer::ThermalModel,
        heterogeneity::HeterogeneityModel,
        traits::*, // Import all traits
    },
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
        [simulation]
        domain_size_x = 0.06
        domain_size_yz = 0.04
        points_per_wavelength = 6
        frequency = 1000000.0  # 1 MHz - typical HIFU frequency
        # amplitude is now part of [source]
        num_cycles = 5.0
        pml_thickness = 10
        pml_sigma_acoustic = 100.0
        pml_sigma_light = 10.0
        pml_polynomial_order = 2
        pml_reflection = 0.000001
        kspace_alpha = 0.5     # k-space correction coefficient
        
        [source]
        # HIFU transducer parameters
        num_elements = 32
        signal_type = "sine"
        frequency = 1000000.0  # Source signal frequency
        amplitude = 1.0e6      # Source signal amplitude
        focus_x = 0.03
        focus_y = 0.0
        focus_z = 0.0
        # phase is optional

        [output]
        pressure_file = "hifu_pressure.csv"
        light_file = "hifu_light.csv"
        summary_file = "hifu_summary.csv"
        snapshot_interval = 10
        enable_visualization = true
    "#;
    let mut file = File::create("hifu_config.toml")?;
    file.write_all(config_content.as_bytes())?;

    let config = Config::from_file("hifu_config.toml")?;
    
    // Initialize grid and time from simulation config
    let grid = config.simulation.initialize_grid()?;
    let time = config.simulation.initialize_time(&grid)?;

    // Create a custom, more physically accurate water medium for HIFU
    // This medium will be used to initialize the source and boundary
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
    let custom_medium = Arc::new(medium_obj); // Renamed to avoid conflict if config.medium() was used
    
    // Initialize source using the custom medium and source config
    let source = config.source.initialize_source(custom_medium.as_ref(), &grid)?;
    
    // Initialize PML boundary using the custom medium and simulation config for PML params
    let boundary = Box::new(PMLBoundary::new(
        config.simulation.pml_thickness,
        config.simulation.pml_sigma_acoustic,
        config.simulation.pml_sigma_light,
        custom_medium.as_ref(), // Use the custom medium
        &grid,
        config.simulation.frequency, // Simulation frequency for PML
        Some(config.simulation.pml_polynomial_order),
        Some(config.simulation.pml_reflection),
    ));
    
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

    // Instantiate physics models
    let grid_clone = grid.clone(); // Clone grid for model instantiation

    let mut acoustic_wave_model = NonlinearWave::new(&grid_clone);
    // Configure the nonlinear wave solver for better HIFU physics
    acoustic_wave_model.set_nonlinearity_scaling(2.0);
    acoustic_wave_model.set_k_space_correction_order(3);

    let wave: Box<dyn AcousticWaveModel> = Box::new(acoustic_wave_model);
    let cavitation: Box<dyn CavitationModelBehavior> = Box::new(CavitationModel::new(&grid_clone, 10e-6));
    let light: Box<dyn LightDiffusionModelTrait> = Box::new(LightDiffusionModel::new(&grid_clone, true, true, true));
    let thermal: Box<dyn ThermalModelTrait> = Box::new(ThermalModel::new(&grid_clone, 293.15, 1e-6, 1e-6));
    let chemical: Box<dyn ChemicalModelTrait> = Box::new(ChemicalModel::new(&grid_clone, true, true));
    let streaming: Box<dyn StreamingModelTrait> = Box::new(StreamingModel::new(&grid_clone));
    let scattering: Box<dyn AcousticScatteringModelTrait> = Box::new(AcousticScatteringModel::new(&grid_clone));
    let heterogeneity: Box<dyn HeterogeneityModelTrait> = Box::new(HeterogeneityModel::new(&grid_clone, 1500.0, 0.05));

    // Create the solver with our optimized medium and injected physics models
    let mut solver = Solver::new(
        grid.clone(), // or just grid
        time.clone(),
        custom_medium, // Use the custom_medium initialized above
        source,
        boundary,
        wave,
        cavitation,
        light,
        thermal,
        chemical,
        streaming,
        scattering,
        heterogeneity,
    );
    
    // Run the simulation
    solver.run(&mut recorder, config.simulation.frequency);
    recorder.save()?;
    
    // Create visualizations
    plot_simulation_outputs(&recorder, &grid, &time, solver.source.as_ref());

    Ok(())
}
