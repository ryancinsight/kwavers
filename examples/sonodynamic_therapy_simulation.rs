// examples/sonodynamic_therapy_simulation.rs
use kwavers::{
    generate_summary, init_logging, plot_simulation_outputs, save_light_data, save_pressure_data,
    Config, PMLBoundary, Recorder, Sensor,
    Solver, NonlinearWave, // Added NonlinearWave for concrete type
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
use log::info;
use std::fs::File;
use std::io::{Write};
use kwavers::boundary::pml::PMLConfig;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    init_logging()?;

    // Write config file with reduced parameters for troubleshooting
    let config_content = r#"
        [simulation]
        domain_size_x = 0.05
        domain_size_yz = 0.03
        points_per_wavelength = 6
        frequency = 180000.0 # Overall simulation frequency
        amplitude = 1.0e5    # Overall simulation amplitude (can be default for source)
        num_cycles = 2.0
        pml_thickness = 5
        pml_sigma_acoustic = 100.0
        pml_sigma_light = 10.0
        pml_polynomial_order = 2
        pml_reflection = 0.000001

        [source]
        num_elements = 4
        signal_type = "sine"
        frequency = 180000.0 # Source-specific signal frequency
        amplitude = 1.0e5    # Source-specific signal amplitude
        focus_x = 0.025
        focus_y = 0.0
        focus_z = 0.015
        # phase is optional, defaults to 0.0

        [output]
        snapshot_interval = 10
        light_file = "sdt_light_output.csv"
        # pressure_file and summary_file will use defaults
    "#;
    let mut file = File::create("sdt_config.toml")?;
    file.write_all(config_content.as_bytes())?;

    // Load configuration
    let config = Config::from_file("sdt_config.toml")?;

    // Initialize components from config
    let grid = config.simulation.initialize_grid()?;
    let time = config.simulation.initialize_time(&grid)?;
    let medium = config.simulation.initialize_medium(&grid)?; // Returns Arc<dyn Medium>
    
    let source = config.source.initialize_source(medium.as_ref(), &grid)?;
    
    let pml_config = PMLConfig {
        thickness: config.simulation.pml_thickness,
        sigma_max_acoustic: 2.0,
        sigma_max_light: 1.0,
        alpha_max_acoustic: 0.0,
        alpha_max_light: 0.0,
        kappa_max_acoustic: 1.0,
        kappa_max_light: 1.0,
        target_reflection: Some(1e-6),
    };
    let boundary = Box::new(PMLBoundary::new(pml_config).expect("Failed to create PML boundary"));

    // Reduced sensor positions (just 4 instead of 25)
    let sensor_positions = vec![
        (0.025, 0.0, 0.0),
        (0.025, 0.01, 0.0),
        (0.025, 0.0, 0.01),
        (0.025, 0.01, 0.01),
    ];
    
    let sensor = Sensor::new(&grid, &time, &sensor_positions);
    let mut recorder = Recorder::new(sensor, &time, "sdt_sensor_data", true, true, 10);

    // Instantiate physics models
    let grid_clone = grid.clone(); // Clone grid for model instantiation

    // Default physics models (as Solver::new used to create them)
    let wave: Box<dyn AcousticWaveModel> = Box::new(NonlinearWave::new(&grid_clone));
    let cavitation: Box<dyn CavitationModelBehavior> = Box::new(CavitationModel::new(&grid_clone, 10e-6));
    let light: Box<dyn LightDiffusionModelTrait> = Box::new(LightDiffusionModel::new(&grid_clone, true, true, true));
    let thermal: Box<dyn ThermalModelTrait> = Box::new(ThermalModel::new(&grid_clone, 293.15, 1e-6, 1e-6));
    let chemical: Box<dyn ChemicalModelTrait> = Box::new(ChemicalModel::new(&grid_clone, true, true));
    let streaming: Box<dyn StreamingModelTrait> = Box::new(StreamingModel::new(&grid_clone));
    let scattering: Box<dyn AcousticScatteringModelTrait> = Box::new(AcousticScatteringModel::new(&grid_clone));
    let heterogeneity: Box<dyn HeterogeneityModelTrait> = Box::new(HeterogeneityModel::new(&grid_clone, 1500.0, 0.05));

    // Run simulation
    let mut solver = Solver::new(
        grid.clone(), // or just grid
        time.clone(),
        medium,
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
        4, // num_simulation_fields for acoustic + light + temp + bubble_radius
    );
    solver.run(&mut recorder, config.simulation.frequency);

    // Save outputs
    save_pressure_data(&recorder, &time, "sdt_pressure_output.csv")?;
    save_light_data(&recorder, &time, "sdt_light_output.csv")?;
    generate_summary(&recorder, "sdt_summary.csv")?;
    recorder.save()?;

    // Generate plots
    plot_simulation_outputs(&recorder, &grid, &time, solver.source.as_ref());

    info!("SDT simulation completed. Outputs saved and plotted as HTML files.");
    Ok(())
}
