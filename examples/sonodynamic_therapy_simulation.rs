// examples/sonodynamic_therapy_simulation.rs
use kwavers::{
    generate_summary, init_logging, plot_simulation_outputs, save_light_data, save_pressure_data,
    Config, HanningApodization, HomogeneousMedium, MatrixArray, PMLBoundary, Recorder, Sensor,
    Solver,
};
use log::info;
use std::fs::File;
use std::io::{Write};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    init_logging()?;

    // Write config file
    let config_content = r#"
        domain_size_x = 0.1
        domain_size_yz = 0.05
        points_per_wavelength = 10
        frequency = 180000.0
        amplitude = 1.0
        num_cycles = 5.0
        pml_thickness = 10

        num_elements = 0
        signal_type = "chirp"
        start_freq = 160000.0
        end_freq = 200000.0
        signal_duration = 0.0000555555
        focus_x = 0.05
        focus_y = 0.0
        focus_z = 0.025

        snapshot_interval = 5
        light_file = "sdt_light_output.csv"
    "#;
    let mut file = File::create("sdt_config.toml")?;
    file.write_all(config_content.as_bytes())?;

    // Load configuration
    let config = Config::from_file("sdt_config.toml")?;
    let grid = config.grid().clone();
    let time = config.time().clone();
    let medium = Box::new(HomogeneousMedium::water(&grid));
    let source = Box::new(MatrixArray::with_focus(
        0.1,
        0.05, // Width, Height
        16,
        8,   // num_x, num_y
        0.0, // z_pos
        config.source().signal().clone_box(),
        medium.as_ref(),
        &grid,
        config.simulation.frequency,
        config.source.focus_x.unwrap(),
        config.source.focus_y.unwrap(),
        config.source.focus_z.unwrap(),
        HanningApodization, // Default apodization
    )) as Box<dyn kwavers::Source>;

    let boundary = Box::new(PMLBoundary::new(
        config.simulation.pml_thickness,
        100.0, // alpha_max
        10.0,  // sigma_max
        medium.as_ref(),
        &grid,
        config.simulation.frequency,
    ));

    // Define sensor positions (grid along focal plane)
    let mut sensor_positions = Vec::new();
    for i in 0..5 {
        for j in 0..5 {
            sensor_positions.push((0.05, i as f64 * 0.01 - 0.02, j as f64 * 0.01 - 0.02));
        }
    }
    let sensor = Sensor::new(&grid, &time, &sensor_positions);
    let mut recorder = Recorder::new(sensor, &time, "sdt_sensor_data", true, true, 5);

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
