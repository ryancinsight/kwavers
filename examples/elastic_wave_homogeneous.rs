use kwavers::{
    grid::Grid,
    HomogeneousMedium, // Corrected: HomogeneousMedium is directly under kwavers due to pub use
    physics::mechanics::elastic_wave::ElasticWave,
    source::Source,
    signal::{SineWave, Signal},
    solver::{TOTAL_FIELDS, VX_IDX, VY_IDX, VZ_IDX, SZZ_IDX},
    time::Time,
    boundary::{Boundary, PMLBoundary},
    recorder::Recorder,
    sensor::Sensor, // Added Sensor
    log::init_logging,
    physics::traits::AcousticWaveModel,
};
use ndarray::{Array3, Array4, Axis};
use std::sync::Arc;
use std::error::Error;
use std::io::Write; // For manual CSV writing
use log::info;

// --- Simple PointSource (if not existing) ---
#[derive(Debug)]
struct SimplePointSource {
    position: (f64, f64, f64),
    signal: Box<dyn Signal>,
    magnitude: f64,
}

impl SimplePointSource {
    #[allow(dead_code)]
    pub fn new(position: (f64, f64, f64), signal: Box<dyn Signal>, magnitude: f64) -> Self {
        Self { position, signal, magnitude }
    }
}

impl Source for SimplePointSource {
    fn get_source_term(&self, t: f64, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        let dx = (x - self.position.0).abs();
        let dy = (y - self.position.1).abs();
        let dz = (z - self.position.2).abs();
        if dx < grid.dx / 2.0 && dy < grid.dy / 2.0 && dz < grid.dz / 2.0 {
            self.signal.amplitude(t) * self.magnitude
        } else {
            0.0
        }
    }
    fn positions(&self) -> Vec<(f64, f64, f64)> { vec![self.position] }
    fn signal(&self) -> &dyn Signal { self.signal.as_ref() }
}
// --- End Simple PointSource ---

fn main() -> Result<(), Box<dyn Error>> {
    init_logging()?;

    info!("Starting 3D Homogeneous Elastic Wave Simulation Example");

    let output_dir_name = "output_elastic_wave_homogeneous";
    std::fs::create_dir_all(output_dir_name)?;

    // --- 1. Define Grid ---
    let (nx, ny, nz) = (64, 64, 64);
    let (dx, dy, dz) = (0.001, 0.001, 0.001);
    let grid = Grid::new(nx, ny, nz, dx, dy, dz);
    info!("Grid: {}x{}x{} points, spacing: {}x{}x{} m", nx, ny, nz, dx, dy, dz);

    // --- 2. Define Medium (Homogeneous Elastic) ---
    let density: f64 = 1000.0;
    let lame_mu: f64 = 0.357e6;
    let lame_lambda: f64 = 1.428e6;

    let mut medium_props = HomogeneousMedium::new(density, 1500.0, &grid, 0.0, 0.0);
    medium_props.density = density;
    medium_props.lame_lambda_val = lame_lambda;
    medium_props.lame_mu_val = lame_mu;

    let cp: f64 = ((lame_lambda + 2.0 * lame_mu) / density).sqrt();
    let cs: f64 = (lame_mu / density).sqrt();
    info!("Medium: Density={} kg/m^3, Lambda={} Pa, Mu={} Pa", density, lame_lambda, lame_mu);
    info!("Expected Speeds: Cp={} m/s, Cs={} m/s", cp, cs);

    let medium = Arc::new(medium_props);

    // --- 3. Define Time Settings ---
    let cfl_dt = dx / cp;
    let dt = cfl_dt * 0.3;
    let num_steps = 300;
    let t_end = num_steps as f64 * dt;
    let time = Time::new(dt, num_steps);
    info!("Time: dt={} s, num_steps={}, t_end={} s (CFL dt ~{:.2e} s)", dt, num_steps, t_end, cfl_dt);

    // --- 4. Define Source ---
    let source_freq = 0.5e6;
    let source_mag = 1e6;
    let source_pos_x = grid.x_coordinates()[nx/2];
    let source_pos_y = grid.y_coordinates()[ny/2];
    let source_pos_z = dz * 10.0;
    let source_pos = (source_pos_x, source_pos_y, source_pos_z);
    let sine_signal = SineWave::new(source_freq, source_mag, 0.0); // Corrected
    let source: Box<dyn Source> = Box::new(SimplePointSource::new(source_pos, Box::new(sine_signal), 1.0));
    info!("Source: Freq={} Hz, Pos=({:.3}, {:.3}, {:.3}) m", source_freq, source_pos.0, source_pos.1, source_pos.2);

    // --- 5. Define Boundary Conditions ---
    let pml_thickness = 10;
    let _boundary: Box<dyn Boundary> = Box::new(PMLBoundary::new(pml_thickness, 0.0, 0.0, medium.as_ref(), &grid, source_freq, None, None));
    info!("Boundary: PML, thickness={} points", pml_thickness);

    // --- 6. Initialize Recorder (for snapshots primarily) & Manual Data Collection ---
    let dummy_sensor_positions: Vec<(f64, f64, f64)> = vec![(0.0,0.0,0.0)]; // Minimal sensor for Recorder
    let sensor_for_recorder = Sensor::new(&grid, &time, &dummy_sensor_positions);
    let snapshot_interval = 20; // Save snapshots less frequently
    let mut recorder = Recorder::new(
        sensor_for_recorder,
        &time,
        &format!("{}/snapshot_metadata", output_dir_name),
        false,
        false,
        snapshot_interval
    );
    info!("Recorder initialized for snapshots (interval: {} steps).", snapshot_interval);

    // Manual data collection for specific elastic fields at a sensor point
    let sensor_grid_idx = (nx/2, ny/2, nz/2);
    let mut center_vz_data: Vec<(f64, f64)> = Vec::with_capacity(num_steps);
    let mut center_szz_data: Vec<(f64, f64)> = Vec::with_capacity(num_steps);
    info!("Manual sensor point for vz, szz at grid index: ({}, {}, {})", sensor_grid_idx.0, sensor_grid_idx.1, sensor_grid_idx.2);

    // --- 7. Simulation Loop ---
    let mut fields_array = Array4::zeros((TOTAL_FIELDS, nx, ny, nz));
    let prev_pressure_dummy = Array3::zeros((nx, ny, nz));

    let mut wave_solver_mut = ElasticWave::new(&grid);

    info!("Starting simulation loop (direct wave model update)...");
    for step in 0..num_steps {
        let current_time = step as f64 * dt;

        AcousticWaveModel::update_wave(
            &mut wave_solver_mut,
            &mut fields_array,
            &prev_pressure_dummy,
            source.as_ref(),
            &grid,
            medium.as_ref(),
            dt,
            current_time,
        );

        let mut pml_boundary_mut = PMLBoundary::new(pml_thickness, 0.0, 0.0, medium.as_ref(), &grid, source_freq, None, None);
        let elastic_velocity_indices = [VX_IDX, VY_IDX, VZ_IDX];
        for &field_idx in elastic_velocity_indices.iter() {
            if field_idx < fields_array.shape()[0] {
                let mut component = fields_array.index_axis(Axis(0), field_idx).to_owned();
                pml_boundary_mut.apply_acoustic(&mut component, &grid, step);
                fields_array.index_axis_mut(Axis(0), field_idx).assign(&component);
            }
        }

        recorder.record(&fields_array, step, current_time);

        center_vz_data.push((current_time, fields_array[[VZ_IDX, sensor_grid_idx.0, sensor_grid_idx.1, sensor_grid_idx.2]]));
        center_szz_data.push((current_time, fields_array[[SZZ_IDX, sensor_grid_idx.0, sensor_grid_idx.1, sensor_grid_idx.2]]));

        if step % 50 == 0 || step == num_steps -1 {
            info!("Step {}/{} (Time: {:.3e} s)", step, num_steps, current_time);
             let vz_max = fields_array.index_axis(Axis(0), VZ_IDX).iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
             info!("Max Vz in domain: {:.3e}", vz_max);
        }
    }
    AcousticWaveModel::report_performance(&wave_solver_mut);
    info!("Simulation loop finished.");

    // --- 8. Save Results ---
    recorder.save()?;
    info!("Snapshot metadata (and standard field snapshots) saved via Recorder to directory: {}", output_dir_name);

    let mut vz_file = std::fs::File::create(format!("{}/center_vz.csv", output_dir_name))?;
    writeln!(vz_file, "Time,Vz")?;
    for (t_val, vz_val) in center_vz_data {
        writeln!(vz_file, "{:.6e},{:.6e}", t_val, vz_val)?;
    }
    info!("Manually saved Vz sensor data to {}/center_vz.csv", output_dir_name);

    let mut szz_file = std::fs::File::create(format!("{}/center_szz.csv", output_dir_name))?;
    writeln!(szz_file, "Time,Szz")?;
    for (t_val, szz_val) in center_szz_data {
        writeln!(szz_file, "{:.6e},{:.6e}", t_val, szz_val)?;
    }
    info!("Manually saved Szz sensor data to {}/center_szz.csv", output_dir_name);

    info!("--- Validation Notes (Manual Check) ---");
    info!("Expected P-wave speed: {:.2} m/s", cp);
    info!("Expected S-wave speed: {:.2} m/s", cs);
    let sensor_world_z = sensor_grid_idx.2 as f64 * dz;
    let distance_to_sensor_z = (sensor_world_z - source_pos.2).abs();
    info!("Distance to sensor (z-axis): {:.4} m", distance_to_sensor_z);
    if cp > 0.0 {
        info!("Expected P-wave arrival at sensor (z-component): {:.3e} s", distance_to_sensor_z / cp);
    }
    if cs > 0.0 {
        info!("Expected S-wave arrival at sensor (z-component, if generated effectively by Fz source): {:.3e} s", distance_to_sensor_z / cs);
    }
    info!("Check output CSVs for wavefronts and arrival times.");

    Ok(())
}
