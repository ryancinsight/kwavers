// examples/advanced_hifu_with_sonoluminescence.rs
use kwavers::{
    init_logging, plot_simulation_outputs, Config, HanningApodization, HomogeneousMedium,
    LinearArray, PMLBoundary, Recorder, Sensor, SineWave, Solver,
};
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    init_logging()?;
    let config = Config::from_file("hifu_config.toml")?;
    let grid = config.grid().clone();
    let time = config.time().clone();
    let medium = Box::new(HomogeneousMedium::new(
        998.0,  // Density (kg/m³) for water
        1500.0, // Sound speed (m/s) for water
        &grid, 1.0,  // mu_a (absorption coefficient for light)
        50.0, // mu_s_prime (reduced scattering coefficient)
    ));
    let signal = Box::new(SineWave::new(
        config.simulation.frequency,
        config.simulation.amplitude,
        0.0,
    ));
    let source = Box::new(LinearArray::with_focus(
        0.1, // Length
        config.source.num_elements,
        0.0,
        0.0, // y0, z0
        signal,
        medium.as_ref(),
        &grid,
        config.simulation.frequency,
        config.source.focus_x.unwrap_or(0.05),
        config.source.focus_y.unwrap_or(0.0),
        config.source.focus_z.unwrap_or(0.025),
        HanningApodization, // Default apodization
    )) as Box<dyn kwavers::Source>;
    let boundary = Box::new(PMLBoundary::new(
        config.simulation.pml_thickness,
        0.5, // alpha_max (example value)
        1.0, // sigma_max (example value)
        medium.as_ref(),
        &grid,
        config.simulation.frequency,
    ));
    let sensor_positions: Vec<(f64, f64, f64)> = vec![
        (
            grid.nx as f64 / 2.0 * grid.dx,
            grid.ny as f64 / 2.0 * grid.dy,
            grid.nz as f64 / 2.0 * grid.dz,
        ),
        (
            (grid.nx / 2 + 10) as f64 * grid.dx,
            grid.ny as f64 / 2.0 * grid.dy,
            grid.nz as f64 / 2.0 * grid.dz,
        ),
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

    let mut solver = Solver::new(grid.clone(), time.clone(), medium, source, boundary);
    solver.run(&mut recorder, config.simulation.frequency);
    recorder.save()?;
    plot_simulation_outputs(&recorder, &grid, &time, solver.source.as_ref());

    Ok(())
}
