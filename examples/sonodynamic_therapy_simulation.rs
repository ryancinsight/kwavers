// examples/sonodynamic_therapy_simulation.rs
use kwavers::{
    generate_summary, init_logging, plot_simulation_outputs, save_light_data, save_pressure_data,
    Config, HanningApodization, HomogeneousMedium, MatrixArray, PMLBoundary, Recorder, Sensor,
    Solver, SineWave,
};
use log::info;
use std::fs::File;
use std::io::{Write};
use std::sync::Arc;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    init_logging()?;

    // Write config file with reduced parameters for troubleshooting
    let config_content = r#"
        domain_size_x = 0.05
        domain_size_yz = 0.03
        points_per_wavelength = 6
        frequency = 180000.0
        amplitude = 1.0e5
        num_cycles = 2.0
        pml_thickness = 5
        pml_sigma_acoustic = 100.0
        pml_sigma_light = 10.0
        pml_polynomial_order = 2
        pml_reflection = 0.000001

        num_elements = 4
        signal_type = "sine"
        focus_x = 0.025
        focus_y = 0.0
        focus_z = 0.015

        snapshot_interval = 10
        light_file = "sdt_light_output.csv"
    "#;
    let mut file = File::create("sdt_config.toml")?;
    file.write_all(config_content.as_bytes())?;

    // Load configuration
    let config = Config::from_file("sdt_config.toml")?;
    let grid = config.grid().clone();
    let time = config.time().clone();
    
    // Create medium with Arc
    let medium = Arc::new(HomogeneousMedium::water(&grid));
    
    // Simple sine wave signal for troubleshooting
    let signal = Box::new(SineWave::new(config.simulation.frequency, config.simulation.amplitude, 0.0));
    
    // Simpler transducer configuration
    let source = Box::new(MatrixArray::with_focus(
        0.05,
        0.02, // Width, Height (smaller)
        4,
        2,   // num_x, num_y (fewer elements)
        0.0, // z_pos
        signal,
        medium.as_ref(),
        &grid,
        config.simulation.frequency,
        config.source.focus_x.unwrap_or(0.025),
        config.source.focus_y.unwrap_or(0.0),
        config.source.focus_z.unwrap_or(0.015),
        HanningApodization,
    )) as Box<dyn kwavers::Source>;

    // Use the PML from the config
    let boundary = Box::new(config.pml().clone());

    // Reduced sensor positions (just 4 instead of 25)
    let sensor_positions = vec![
        (0.025, 0.0, 0.0),
        (0.025, 0.01, 0.0),
        (0.025, 0.0, 0.01),
        (0.025, 0.01, 0.01),
    ];
    
    let sensor = Sensor::new(&grid, &time, &sensor_positions);
    let mut recorder = Recorder::new(sensor, &time, "sdt_sensor_data", true, true, 10);

    // Run simulation
    let mut solver = Solver::new(grid.clone(), time.clone(), medium, source, boundary);
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
