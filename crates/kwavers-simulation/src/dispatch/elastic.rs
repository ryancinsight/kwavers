//! Elastic wave solver dispatch.

use kwavers_core::error::{KwaversError, KwaversResult};
use crate::types::{SimulationRunRequest, SimulationRunResult};
use kwavers_solver::forward::elastic::swe::{ElasticWaveConfig, ElasticWaveField, ElasticWaveSolver};

/// Run an elastic-wave simulation.
pub fn run(req: &SimulationRunRequest<'_>) -> KwaversResult<SimulationRunResult> {
    let (nx, ny, nz) = req.grid.dimensions();
    let u0_opt = match &req.grid_source.p0 {
        Some(u0) => {
            if u0.dim() != (nx, ny, nz) {
                return Err(KwaversError::InvalidInput("Elastic initial displacement shape mismatch".into()));
            }
            Some(u0.clone())
        }
        None => None,
    };

    let sensor_mask = req.sensor_mask.clone();
    let pml_thickness = req.pml.map(|p| p.size.unwrap_or(10)).unwrap_or(10);
    let config = ElasticWaveConfig {
        time_step: req.dt,
        simulation_time: req.dt * (req.time_steps as f64),
        pml_thickness,
        save_every: 1,
        sensor_mask,
        ..ElasticWaveConfig::default()
    };

    let mut solver = ElasticWaveSolver::new(req.grid, req.medium, config)?;
    let mut initial_field = ElasticWaveField::new(nx, ny, nz);
    if let Some(ref u0) = u0_opt {
        initial_field.uz.assign(u0);
    }

    let duration = req.dt * (req.time_steps as f64);
    let _final_field = solver.propagate(&initial_field, duration, None)?;

    let recorded_p = solver.extract_recorded_data();
    let sensor_data = recorded_p.unwrap_or_else(|| ndarray::Array2::zeros((1, 0)));
    let (ux_data, uy_data, uz_data) = solver.extract_recorded_velocity_components();

    Ok(SimulationRunResult {
        sensor_data, stats: None,
        ux_data, uy_data, uz_data,
        ix_data: None, iy_data: None, iz_data: None,
        i_avg_x: None, i_avg_y: None, i_avg_z: None,
        velocity_stats: None, full_grid_stats: None,
        thermal_temperature: None, thermal_dose: None,
    })
}
